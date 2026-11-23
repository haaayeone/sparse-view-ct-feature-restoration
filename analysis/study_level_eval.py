"""
study_level_eval.py — 검사(study) 단위 AUROC + 고정 특이도에서의 민감도 (피드백 3.2). 재학습 없음.

analysis/dump_preds.py 가 저장한 슬라이스별 예측(preds/{tag}_{split}.npz: p_full / p_under / p_rest)을 그대로 쓴다.

1) 검사 단위 AUROC
   검사 점수 = 그 검사 슬라이스 확률의 최댓값(--agg max, 기본) 또는 상위 3개 평균(--agg top3),
   검사 라벨 = 슬라이스 라벨 중 하나라도 양성이면 양성. E_f · E_u · E_r 각각, 라벨별 + 6라벨 평균.
   검사 3,262건 paired bootstrap(B, seed 42 → bootstrap_study.py와 같은 검사 표본) 95% CI,
   Δ = AUROC(E_r) − AUROC(E_u)의 z 검정 p, 180개 라벨 단위 비교 Holm 보정.
2) 고정 특이도(0.90, 0.95)에서의 민감도 — any, epidural, 슬라이스 단위와 검사 단위 모두
   mode=roc    : 각 표현의 test ROC에서 특이도가 목표값이 되는 점의 민감도 (bootstrap 표본마다 다시 계산)
   mode=fixed  : 임계값을 val 세트의 E_f(기준) 확률에서 목표 특이도로 한 번 정하고, test의 E_f/E_u/E_r에 그대로 적용.
                 프로브와 함께 임계값도 고정한 실제 사용 상황 → 민감도와 실제 특이도를 함께 보고.
                 val 예측 파일(preds/{tag}_val.npz)이 있을 때만 계산 (없으면 아래 Usage의 dump_preds --split val).
   민감도 정의: 점수 > 임계값이면 양성 판정. 임계값 = 음성 중 점수 ≤ 임계값 비율이 처음으로 목표 특이도 이상이 되는 점수.

출력 (--out_dir):
  study_auroc_perlabel.csv   인코더 × view × 라벨: 검사 단위 AUROC f/u/r (+CI), Δ (+CI, SE), p_z, p_holm, 양성 검사 수
  study_auroc_summary.csv    인코더 × view: 6라벨 평균 검사 단위 AUROC f/u/r (+CI), Δ (+CI), R% (+CI), p_z
  sens_at_spec.csv           인코더 × view × 단위(slice/study) × 라벨(any/epidural) × 목표 특이도 × mode × 표현:
                             민감도 (+CI), 실제 특이도 (+CI), 임계값, Δ민감도(E_r − E_u) (+CI, p_z)

실행 위치: 서버 또는 로컬 (CPU, numpy만). 조건당 B=2000에서 1–3분.
Usage (서버):
  conda activate hayeon
  cd ~/hayeon/ich_cf_clean
  python analysis/dump_preds.py --all --split val
  python analysis/study_level_eval.py --pred_dir preds --B 2000 --out_dir .
  python analysis/study_level_eval.py --pred_dir preds --B 200 --encoders medsiglip_after_r360 --views 6    # 빠른 확인
결과 가져오기 (로컬 PowerShell, 한 줄씩):
  scp neumous-lab@100.73.97.101:/home/neumous-lab/hayeon/ich_cf_clean/study_auroc_*.csv "C:\\Users\\user\\Desktop\\MIGT\\01. sparse ct project\\result"
  scp neumous-lab@100.73.97.101:/home/neumous-lab/hayeon/ich_cf_clean/sens_at_spec.csv "C:\\Users\\user\\Desktop\\MIGT\\01. sparse ct project\\result"
"""
import argparse
import csv
import glob
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_study import weighted_auc, ci, p_two_sided, holm, NAME, ORDER  # noqa: E402

FNAME = re.compile(r"^(?P<enc>.+)_nv(?P<v>\d+)_(?P<split>[a-z]+)\.npz$")
REPS = {"f": "p_full", "u": "p_under", "r": "p_rest"}


def load(path):
    d = np.load(path, allow_pickle=True)
    return dict(y=d["y"].astype(bool), study=d["study_id"].astype(str), labels=[str(x) for x in d["label_cols"]],
                p={k: d[v].astype(np.float64) for k, v in REPS.items()})


def to_study(study, y, p, agg):
    """슬라이스 → 검사. 반환: sidx (N,) 검사 번호, 검사 라벨 (S,L), 검사 점수 {k: (S,L)}"""
    uniq, sidx = np.unique(study, return_inverse=True)
    order = np.argsort(sidx, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(sidx[order])) + 1]
    ys = np.maximum.reduceat(y[order].astype(np.int8), starts, axis=0).astype(bool)
    ps = {}
    for k, v in p.items():
        vo = v[order]
        if agg == "max":
            ps[k] = np.maximum.reduceat(vo, starts, axis=0)
        else:                                                      # top3: 상위 3 슬라이스 평균
            ends = np.r_[starts[1:], len(vo)]
            ps[k] = np.stack([np.sort(vo[s:e], axis=0)[-3:].mean(0) for s, e in zip(starts, ends)])
    return sidx, ys, ps


def threshold_at_spec(neg_scores, spec, w=None):
    """(가중) 음성 점수에서 '점수 ≤ thr 비율 ≥ spec'인 최소 thr"""
    o = np.argsort(neg_scores, kind="mergesort")
    s = neg_scores[o]
    if w is None:
        cw = np.arange(1, len(s) + 1, dtype=float)
        k = int(np.argmax(cw >= spec * len(s) - 1e-9))
        return float(s[k])
    cw = np.cumsum(w[:, o], axis=1)
    k = np.argmax(cw >= spec * cw[:, -1:] - 1e-9, axis=1)
    return s[k]


def sens_spec_w(score, y, W, thr):
    """W (b,N) 가중치, thr 스칼라 또는 (b,) → 민감도 (b,), 특이도 (b,)"""
    thr = np.broadcast_to(np.asarray(thr, float), (W.shape[0],))[:, None]
    Wp, Wn = W[:, y], W[:, ~y]
    with np.errstate(invalid="ignore", divide="ignore"):        # 복원추출 표본에 양성이 하나도 없으면 nan
        sens = (Wp * (score[y][None, :] > thr)).sum(1) / Wp.sum(1)
        spec = (Wn * (score[~y][None, :] <= thr)).sum(1) / Wn.sum(1)
    return sens, spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default="preds")
    ap.add_argument("--split", default="test")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--agg", choices=["max", "top3"], default="max")
    ap.add_argument("--specs", nargs="+", type=float, default=[0.90, 0.95])
    ap.add_argument("--sens_labels", nargs="+", default=["any", "epidural"])
    ap.add_argument("--encoders", nargs="+", default=None, help=f"tag 앞부분 (기본 전부: {ORDER})")
    ap.add_argument("--views", nargs="+", type=int, default=None)
    ap.add_argument("--out_dir", default=".")
    a = ap.parse_args()

    files = {}
    for f in glob.glob(os.path.join(a.pred_dir, f"*_{a.split}.npz")):
        m = FNAME.match(os.path.basename(f))
        if not m or m["enc"] not in NAME:
            continue
        if (a.encoders and m["enc"] not in a.encoders) or (a.views and int(m["v"]) not in a.views):
            continue
        files[(m["enc"], int(m["v"]))] = f
    if not files:
        raise SystemExit(f"{a.pred_dir}에 *_{a.split}.npz 없음 (먼저 analysis/dump_preds.py)")
    keys = sorted(files, key=lambda k: (ORDER.index(k[0]), -k[1]))

    perl, summ, sens_rows = [], [], []
    for enc, v in keys:
        t0 = time.time()
        name = NAME[enc]
        d = load(files[(enc, v)])
        labels, L = d["labels"], len(d["labels"])
        sidx, ys, ps = to_study(d["study"], d["y"], d["p"], a.agg)
        S = ys.shape[0]
        val_path = os.path.join(a.pred_dir, f"{enc}_nv{v}_val.npz")
        dv = load(val_path) if os.path.exists(val_path) else None
        if dv is not None:
            _, ys_v, ps_v = to_study(dv["study"], dv["y"], dv["p"], a.agg)

        # 민감도 대상: (단위, 라벨, 점수 dict, 라벨 벡터) 와 fixed 임계값
        targets = []
        for lab in a.sens_labels:
            if lab not in labels:
                continue
            c = labels.index(lab)
            for level in ("slice", "study"):
                sc = {k: (d["p"][k][:, c] if level == "slice" else ps[k][:, c]) for k in REPS}
                yy = d["y"][:, c] if level == "slice" else ys[:, c]
                thr_fixed = {}
                if dv is not None:
                    ref_s = dv["p"]["f"][:, c] if level == "slice" else ps_v["f"][:, c]
                    ref_y = dv["y"][:, c] if level == "slice" else ys_v[:, c]
                    thr_fixed = {sp: threshold_at_spec(ref_s[~ref_y], sp) for sp in a.specs}
                targets.append(dict(level=level, label=lab, sc=sc, y=yy, thr_fixed=thr_fixed))

        rng = np.random.default_rng(a.seed)                        # bootstrap_study.py 와 같은 검사 표본
        auc = {k: np.empty((a.B, L)) for k in REPS}
        sb = {}                                                    # (ti, spec, mode, rep) → (sens(B,), spec(B,))
        for b0 in range(0, a.B, a.chunk):
            b1 = min(a.B, b0 + a.chunk)
            counts = np.stack([np.bincount(rng.integers(0, S, S), minlength=S) for _ in range(b1 - b0)]).astype(float)
            Wslice = counts[:, sidx]
            for k in REPS:
                for c in range(L):
                    auc[k][b0:b1, c] = weighted_auc(ps[k][:, c], ys[:, c], counts)
            for ti, t in enumerate(targets):
                W = Wslice if t["level"] == "slice" else counts
                for sp in a.specs:
                    for k in REPS:
                        modes = {"roc": threshold_at_spec(t["sc"][k][~t["y"]], sp, W[:, ~t["y"]])}
                        if sp in t["thr_fixed"]:
                            modes["fixed"] = t["thr_fixed"][sp]
                        for mode, thr in modes.items():
                            se_, sp_ = sens_spec_w(t["sc"][k], t["y"], W, thr)
                            key = (ti, sp, mode, k)
                            if key not in sb:
                                sb[key] = (np.empty(a.B), np.empty(a.B))
                            sb[key][0][b0:b1], sb[key][1][b0:b1] = se_, sp_

        ones = np.ones((1, S))
        pt = {k: np.array([weighted_auc(ps[k][:, c], ys[:, c], ones)[0] for c in range(L)]) for k in REPS}
        for c, lab in enumerate(labels):
            row = dict(encoder=name, views=v, label=lab, n_studies=S, n_pos_studies=int(ys[:, c].sum()), agg=a.agg)
            for k in REPS:
                lo, hi = ci(auc[k][:, c])
                row.update({f"auc_{k}": round(pt[k][c], 4), f"auc_{k}_lo": round(lo, 4), f"auc_{k}_hi": round(hi, 4)})
            db, dl = auc["r"][:, c] - auc["u"][:, c], pt["r"][c] - pt["u"][c]
            se, p_z, _ = p_two_sided(dl, db)
            lo, hi = ci(db)
            row.update(delta=round(dl, 4), delta_lo=round(lo, 4), delta_hi=round(hi, 4), delta_se=round(se, 5), p_z=p_z)
            perl.append(row)
        mean = {k: np.nanmean(auc[k], 1) for k in REPS}
        mpt = {k: float(np.nanmean(pt[k])) for k in REPS}
        db, dl = mean["r"] - mean["u"], mpt["r"] - mpt["u"]
        se, p_z, _ = p_two_sided(dl, db)
        with np.errstate(invalid="ignore", divide="ignore"):
            Rb = 100 * db / (mean["f"] - mean["u"])
        row = dict(encoder=name, views=v, n_studies=S, agg=a.agg)
        for k in REPS:
            lo, hi = ci(mean[k])
            row.update({f"auc_{k}": round(mpt[k], 4), f"auc_{k}_lo": round(lo, 4), f"auc_{k}_hi": round(hi, 4)})
        lo, hi = ci(db)
        Rlo, Rhi = ci(Rb)
        row.update(delta=round(dl, 4), delta_lo=round(lo, 4), delta_hi=round(hi, 4), p_z=p_z,
                   R_pct=round(100 * dl / (mpt["f"] - mpt["u"]), 1) if mpt["f"] > mpt["u"] else float("nan"),
                   R_lo=round(Rlo, 1), R_hi=round(Rhi, 1))
        summ.append(row)

        N1 = {"slice": np.ones((1, len(sidx))), "study": ones}
        for ti, t in enumerate(targets):
            for sp in a.specs:
                for mode in ("roc", "fixed"):
                    if mode == "fixed" and sp not in t["thr_fixed"]:
                        continue
                    point = {}
                    for k in REPS:
                        thr = (threshold_at_spec(t["sc"][k][~t["y"]], sp) if mode == "roc" else t["thr_fixed"][sp])
                        se_, sp_ = sens_spec_w(t["sc"][k], t["y"], N1[t["level"]], thr)
                        point[k] = (float(se_[0]), float(sp_[0]), float(thr))
                    for k in REPS:
                        bs_, bp_ = sb[(ti, sp, mode, k)]
                        slo, shi = ci(bs_)
                        plo, phi = ci(bp_)
                        r = dict(encoder=name, views=v, level=t["level"], label=t["label"], spec_target=sp, mode=mode,
                                 rep={"f": "E_f", "u": "E_u", "r": "E_r"}[k], n_pos=int(t["y"].sum()),
                                 n_neg=int((~t["y"]).sum()), threshold=round(point[k][2], 6),
                                 sens=round(point[k][0], 4), sens_lo=round(slo, 4), sens_hi=round(shi, 4),
                                 spec=round(point[k][1], 4), spec_lo=round(plo, 4), spec_hi=round(phi, 4))
                        if k == "r":
                            dsb = bs_ - sb[(ti, sp, mode, "u")][0]
                            dsp = point["r"][0] - point["u"][0]
                            se, p_z, _ = p_two_sided(dsp, dsb)
                            dlo, dhi = ci(dsb)
                            r.update(dsens_r_minus_u=round(dsp, 4), dsens_lo=round(dlo, 4), dsens_hi=round(dhi, 4),
                                     dsens_p_z=p_z)
                        sens_rows.append(r)
        msg = (f"{name:<12}{v:>5}  studies={S}  study AUROC f/u/r={mpt['f']:.3f}/{mpt['u']:.3f}/{mpt['r']:.3f}  "
               f"Δ={dl:+.3f} [{row['delta_lo']:+.3f}, {row['delta_hi']:+.3f}]")
        for r in sens_rows:
            if (r["encoder"], r["views"]) == (name, v) and r["level"] == "study" and r["mode"] == "roc" \
                    and r["spec_target"] == a.specs[0] and r["rep"] != "E_f":
                msg += f"  | study {r['label']} sens@{a.specs[0]:.2f} {r['rep']}={r['sens']:.3f}"
        print(msg + f"  ({time.time() - t0:.0f}s){'' if dv is not None else '  [val preds 없음 → fixed 생략]'}")

    adj = holm([r["p_z"] for r in perl])
    for r, p in zip(perl, adj):
        r["p_holm"] = p
    print(f"\n검사 단위 라벨 비교 {len(perl)}개 중 Holm 보정 p < 0.05: {int(np.sum(adj < 0.05))}개")
    os.makedirs(a.out_dir, exist_ok=True)
    for fname, rows in (("study_auroc_perlabel.csv", perl), ("study_auroc_summary.csv", summ),
                        ("sens_at_spec.csv", sens_rows)):
        if not rows:
            continue
        cols = list(rows[0])
        for r in rows[1:]:
            cols += [c for c in r if c not in cols]
        path = os.path.join(a.out_dir, fname)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"saved {path}  ({len(rows)} rows, B={a.B}, agg={a.agg})")


if __name__ == "__main__":
    main()
