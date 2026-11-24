"""
bootstrap_study.py — 검사(study) 단위 paired bootstrap: AUROC 95% 신뢰구간 + E_u vs E_r 검정 + Holm 보정.

한 검사의 슬라이스들은 서로 상관되어 있으므로 슬라이스가 아니라 검사를 복원추출한다.
같은 복원추출 표본을 E_f · E_u · E_r 에 똑같이 적용하므로(paired) 차이 Δ = AUROC(E_r) − AUROC(E_u)의
분포가 상관을 반영한다. 모든 조건에 같은 seed를 쓰므로 조건 간에도 같은 검사 표본을 공유한다.

검정: z = Δ / SE_boot (SE_boot = bootstrap Δ의 표준편차), 양측 p = erfc(|z|/√2).
  - percentile p(p_perc)도 함께 저장하지만, B=2000이면 최솟값이 약 0.001이라 180개 Holm 보정에는 쓸 수 없다.
    원고의 보정 p는 p_z 기준으로 계산한다.
Holm 보정: 라벨 단위 비교(기본 5 인코더 × 6 view × 6 라벨 = 180개) 전체에 대해 적용.

입력: analysis/dump_preds.py 가 만든 {pred_dir}/{tag}_{split}.npz
출력 (--out_dir):
  bootstrap_perlabel.csv  인코더 × view × 라벨: AUROC f/u/r (+95% CI), Δ (+CI, SE), p_z, p_perc, p_holm
  bootstrap_summary.csv   인코더 × view (6라벨 평균): AUROC f/u/r (+CI), Δ (+CI), R% (+CI), p_z
                           + cos(E_f,E_u)/cos(E_f,E_r) 평균 (+CI)

실행 위치: 서버 또는 로컬 (CPU만, numpy만 필요). 조건당 B=2000에서 대략 1분.
Usage (서버):
  conda activate hayeon
  cd ~/hayeon/ich_cf_clean
  python analysis/bootstrap_study.py --pred_dir preds --B 2000 --out_dir .
  python analysis/bootstrap_study.py --pred_dir preds --B 200          # 빠른 동작 확인
결과 가져오기 (로컬 PowerShell):
  scp neumous-lab@100.73.97.101:/home/neumous-lab/hayeon/ich_cf_clean/bootstrap_*.csv "C:\\Users\\user\\Desktop\\MIGT\\01. sparse ct project\\result"
"""
import argparse
import csv
import glob
import math
import os
import re
import time

import numpy as np

# collect_test_results.py 와 같은 이름·순서 (기준 360 view tag만; --all 이면 나머지도)
NAME = {"medsiglip_after_r360": "MedSigLIP", "siglip_window_r360": "SigLIP", "dinov2_window_r360": "DINOv2",
        "resnet50_window_r360": "ResNet50", "densenet121_window_r360": "DenseNet121"}
ORDER = list(NAME)
FNAME = re.compile(r"^(?P<enc>.+)_nv(?P<v>\d+)_(?P<split>[a-z]+)\.npz$")


def weighted_auc(score, y, W):
    """가중 AUROC (가중치 = 복원추출 횟수 → 중복 표본과 동일). score (N,), y (N,) bool, W (b,N) → (b,)
    동점은 0.5로 처리 (sklearn roc_auc_score 와 같은 정의)."""
    order = np.argsort(score, kind="mergesort")
    s, yy = score[order], y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]           # 동점 그룹 시작 위치
    Ws = W[:, order]
    gp = np.add.reduceat(Ws * yy, starts, axis=1)                # 그룹별 양성 가중합
    gn = np.add.reduceat(Ws * ~yy, starts, axis=1)               # 그룹별 음성 가중합
    below = np.cumsum(gn, axis=1) - gn                           # 그룹보다 점수가 낮은 음성
    num = (gp * (below + 0.5 * gn)).sum(1)
    den = gp.sum(1) * gn.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def ci(x):
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def p_two_sided(point, boot):
    se = float(np.nanstd(boot, ddof=1))
    p_z = math.erfc(abs(point / se) / math.sqrt(2)) if se > 0 else 0.0
    b = boot[~np.isnan(boot)]
    p_perc = min(1.0, 2 * min((np.sum(b <= 0) + 1) / (len(b) + 1), (np.sum(b >= 0) + 1) / (len(b) + 1)))
    return se, p_z, p_perc


def holm(pvals):
    p = np.asarray(pvals, float)
    m, order = len(p), np.argsort(p)
    adj, run = np.empty(m), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[k]))
        adj[k] = run
    return adj


def run_one(path, B, seed, chunk):
    # dump_preds.py 초기 버전은 slice_id/study_id를 object 배열로 저장 → 직접 만든 파일이므로 pickle 허용
    d = np.load(path, allow_pickle=True)
    y = d["y"].astype(bool)
    labels = [str(x) for x in d["label_cols"]]
    probs = {"f": d["p_full"].astype(np.float64), "u": d["p_under"].astype(np.float64),
             "r": d["p_rest"].astype(np.float64)}
    cos = {"u": d["cos_u"].astype(np.float64), "r": d["cos_r"].astype(np.float64)}
    _, sidx = np.unique(d["study_id"], return_inverse=True)      # 슬라이스 → 검사 번호
    S, N, L = sidx.max() + 1, len(sidx), y.shape[1]
    n_slices = np.bincount(sidx, minlength=S)

    rng = np.random.default_rng(seed)                             # 모든 조건 같은 seed → 같은 검사 표본
    auc = {k: np.empty((B, L)) for k in probs}
    cosm = {k: np.empty(B) for k in cos}
    for b0 in range(0, B, chunk):
        b1 = min(B, b0 + chunk)
        counts = np.stack([np.bincount(rng.integers(0, S, S), minlength=S) for _ in range(b1 - b0)])
        W = counts[:, sidx].astype(np.float64)                    # (b, N) 슬라이스 가중치
        for k, p in probs.items():
            for c in range(L):
                auc[k][b0:b1, c] = weighted_auc(p[:, c], y[:, c], W)
        tot = (counts * n_slices).sum(1)
        for k, v in cos.items():
            cosm[k][b0:b1] = (W @ v) / tot

    ones = np.ones((1, N))
    point = {k: np.array([weighted_auc(p[:, c], y[:, c], ones)[0] for c in range(L)]) for k, p in probs.items()}
    return dict(labels=labels, auc=auc, point=point, cosm=cosm,
                cos_point={k: float(v.mean()) for k, v in cos.items()}, S=S, N=N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default="preds")
    ap.add_argument("--split", default="test")
    ap.add_argument("--B", type=int, default=2000, help="bootstrap 반복 수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=100, help="한 번에 계산할 반복 수 (메모리 ≈ chunk×N×8 byte×수 배)")
    ap.add_argument("--all", action="store_true", help="r360 이외 tag도 포함")
    ap.add_argument("--out_dir", default=".")
    a = ap.parse_args()

    files = {}
    for f in glob.glob(os.path.join(a.pred_dir, f"*_{a.split}.npz")):
        m = FNAME.match(os.path.basename(f))
        if m and (a.all or m["enc"] in NAME):
            files[(m["enc"], int(m["v"]))] = f
    if not files:
        raise SystemExit(f"{a.pred_dir}에 *_{a.split}.npz 가 없음 (먼저 analysis/dump_preds.py 실행)")
    rank = lambda e: ORDER.index(e) if e in ORDER else len(ORDER)
    keys = sorted(files, key=lambda k: (rank(k[0]), -k[1]))

    perl, summ = [], []
    for enc, v in keys:
        t0 = time.time()
        r = run_one(files[(enc, v)], a.B, a.seed, a.chunk)
        name = NAME.get(enc, enc)
        auc, pt = r["auc"], r["point"]
        for c, lab in enumerate(r["labels"]):
            dlt_b = auc["r"][:, c] - auc["u"][:, c]
            dlt = pt["r"][c] - pt["u"][c]
            se, p_z, p_perc = p_two_sided(dlt, dlt_b)
            row = dict(encoder=name, views=v, label=lab)
            for k in "fur":
                lo, hi = ci(auc[k][:, c])
                row.update({f"auc_{k}": round(pt[k][c], 4), f"auc_{k}_lo": round(lo, 4), f"auc_{k}_hi": round(hi, 4)})
            lo, hi = ci(dlt_b)
            row.update(delta=round(dlt, 4), delta_lo=round(lo, 4), delta_hi=round(hi, 4), delta_se=round(se, 5),
                       p_z=p_z, p_perc=p_perc)
            perl.append(row)

        mean = {k: auc[k].mean(1) for k in "fur"}
        mpt = {k: float(pt[k].mean()) for k in "fur"}
        dlt_b, dlt = mean["r"] - mean["u"], mpt["r"] - mpt["u"]
        with np.errstate(invalid="ignore", divide="ignore"):
            R_b = 100 * dlt_b / (mean["f"] - mean["u"])
        se, p_z, _ = p_two_sided(dlt, dlt_b)
        row = dict(encoder=name, views=v, n_studies=r["S"], n_slices=r["N"])
        for k in "fur":
            lo, hi = ci(mean[k])
            row.update({f"auc_{k}": round(mpt[k], 4), f"auc_{k}_lo": round(lo, 4), f"auc_{k}_hi": round(hi, 4)})
        lo, hi = ci(dlt_b)
        Rlo, Rhi = ci(R_b)
        row.update(delta=round(dlt, 4), delta_lo=round(lo, 4), delta_hi=round(hi, 4), p_z=p_z,
                   R_pct=round(100 * dlt / (mpt["f"] - mpt["u"]), 1), R_lo=round(Rlo, 1), R_hi=round(Rhi, 1))
        for k in "ur":
            lo, hi = ci(r["cosm"][k])
            row.update({f"cos_{k}": round(r["cos_point"][k], 4), f"cos_{k}_lo": round(lo, 4),
                        f"cos_{k}_hi": round(hi, 4)})
        summ.append(row)
        print(f"{name:<12}{v:>5}  studies={r['S']}  E_u={mpt['u']:.3f}  E_r={mpt['r']:.3f}  "
              f"Δ={dlt:+.3f} [{row['delta_lo']:+.3f}, {row['delta_hi']:+.3f}]  "
              f"R={row['R_pct']}% [{Rlo:.1f}, {Rhi:.1f}]  "
              f"({time.time() - t0:.0f}s)")

    adj = holm([row["p_z"] for row in perl])
    for row, p in zip(perl, adj):
        row["p_holm"] = p
    n_sig = int(np.sum(adj < 0.05))
    print(f"\n라벨 단위 비교 {len(perl)}개 중 Holm 보정 p < 0.05: {n_sig}개")
    for row in perl:
        if row["p_holm"] >= 0.05:
            print(f"  유의하지 않음: {row['encoder']} {row['views']}-view {row['label']}  "
                  f"Δ={row['delta']:+.4f} [{row['delta_lo']:+.4f}, {row['delta_hi']:+.4f}]  "
                  f"p_z={row['p_z']:.3g}  p_holm={row['p_holm']:.3g}")

    os.makedirs(a.out_dir, exist_ok=True)
    for fname, rows in (("bootstrap_perlabel.csv", perl), ("bootstrap_summary.csv", summ)):
        path = os.path.join(a.out_dir, fname)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"saved {path}  ({len(rows)} rows, B={a.B}, seed={a.seed})")


if __name__ == "__main__":
    main()
