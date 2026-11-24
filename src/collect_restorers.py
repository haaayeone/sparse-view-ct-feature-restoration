"""
collect_restorers.py — compare_restorers.py 결과 모으기: seed 평균 ± 표준편차, 그리고 (선택) 검사 단위 paired bootstrap.

입력: {in_dir}/results/*.json, {in_dir}/preds/*.npz  (compare_restorers.py 출력)
출력 (--out_dir, 기본 저장소 루트):
  results_restorers_runs.csv       실행 1건당 1행 (인코더 × view × 방법 × seed): AUROC, Δ, R, cos, 복원 MSE, 선택 에폭 …
  results_restorers_summary.csv    인코더 × view × 방법: seed 수, 평균 ± SD (AUROC_r, Δ, R, cos_r, val/test MSE)
  results_restorers_bootstrap.csv  (--B > 0) 인코더 × view × 방법: seed 평균 AUROC의 95% CI,
                                   기준 방법(--ref, 기본 diffusion)과의 차이 Δ = AUROC(ref) − AUROC(방법) (95% CI, z 검정 p, Holm)
  bootstrap은 analysis/bootstrap_study.py 와 같은 방식(검사 3,262건 복원추출, 같은 seed 42 → 같은 검사 표본).
  seed가 여러 개면 각 복원추출 표본에서 seed별 AUROC를 평균한 값으로 비교한다(검사 표본 변동만 CI에 반영,
  seed 간 변동은 summary의 SD로 따로 보고).

실행 위치: 서버 또는 로컬 (CPU, numpy만). bootstrap은 조건(인코더 × view)당 수 분.
Usage (서버):
  conda activate hayeon
  cd ~/hayeon/ich_cf_clean
  python collect_restorers.py
  python collect_restorers.py --B 2000
  python collect_restorers.py --B 2000 --ref diffusion_vl
결과 가져오기 (로컬 PowerShell, 한 줄씩):
  scp neumous-lab@100.73.97.101:/home/neumous-lab/hayeon/ich_cf_clean/results_restorers_*.csv "C:\\Users\\user\\Desktop\\MIGT\\01. sparse ct project\\result"
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis"))
from bootstrap_study import weighted_auc, ci, p_two_sided, holm  # noqa: E402

NAME = {"medsiglip_after_r360": "MedSigLIP", "siglip_window_r360": "SigLIP", "dinov2_window_r360": "DINOv2",
        "resnet50_window_r360": "ResNet50", "densenet121_window_r360": "DenseNet121"}
ENC_ORDER = list(NAME.values())
METHOD_ORDER = ["identity", "meanshift", "coral", "ridge", "mlp", "direct", "diffusion_vl", "diffusion"]
TAG = re.compile(r"^(?P<enc>.+)_nv(?P<v>\d+)(?:_frac(?P<frac>[\d.]+))?$")


def parse_tag(tag):
    m = TAG.match(tag)
    if not m:
        return tag, -1, 1.0
    return NAME.get(m["enc"], m["enc"]), int(m["v"]), float(m["frac"] or 1.0)


def sort_key(r):
    e = ENC_ORDER.index(r["encoder"]) if r["encoder"] in ENC_ORDER else 99
    mth = METHOD_ORDER.index(r["method"]) if r["method"] in METHOD_ORDER else 99
    return (e, -r["views"], -r["frac"], mth, r.get("seed", 0))


def write_csv(path, rows):
    if not rows:
        return
    cols = list(rows[0])
    for r in rows[1:]:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"saved {path}  ({len(rows)} rows)")


def msd(x):
    x = np.asarray(x, float)
    return float(np.mean(x)), (float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"))


def bootstrap_tag(in_dir, tag, methods, B, seed, chunk):
    """methods: {method: [seed,...]} → {method: (point, boot(B,))} (seed 평균 6라벨 평균 AUROC). E_f/E_u 포함."""
    d = np.load(os.path.join(in_dir, "preds", f"{tag}__ref_test.npz"))
    y = d["y"].astype(bool)
    _, sidx = np.unique(d["study_id"], return_inverse=True)
    S, N, L = sidx.max() + 1, len(sidx), y.shape[1]
    probs = {"E_f": [d["p_full"].astype(np.float64)], "E_u": [d["p_under"].astype(np.float64)]}
    for m, seeds in methods.items():
        probs[m] = [np.load(os.path.join(in_dir, "preds", f"{tag}__{m}__s{s}_test.npz"))["p_rest"].astype(np.float64)
                    for s in seeds]
    rng = np.random.default_rng(seed)
    boot = {m: np.zeros(B) for m in probs}
    for b0 in range(0, B, chunk):
        b1 = min(B, b0 + chunk)
        counts = np.stack([np.bincount(rng.integers(0, S, S), minlength=S) for _ in range(b1 - b0)])
        W = counts[:, sidx].astype(np.float64)
        for m, plist in probs.items():
            acc = np.zeros(b1 - b0)
            for p in plist:
                acc += np.nanmean(np.stack([weighted_auc(p[:, c], y[:, c], W) for c in range(L)], 1), 1)
            boot[m][b0:b1] = acc / len(plist)
    ones = np.ones((1, N))
    point = {m: float(np.mean([np.nanmean([weighted_auc(p[:, c], y[:, c], ones)[0] for c in range(L)])
                               for p in plist])) for m, plist in probs.items()}
    return {m: (point[m], boot[m]) for m in probs}, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="compare")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--B", type=int, default=0, help="bootstrap 반복 수 (0이면 생략, 원고와 같게 하려면 2000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--ref", default="diffusion", help="bootstrap 비교 기준 방법")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.in_dir, "results", "*.json")))
    if not files:
        raise SystemExit(f"{a.in_dir}/results/*.json 없음 (먼저 compare_restorers.py 실행)")
    runs = []
    for f in files:
        r = json.load(open(f, encoding="utf-8"))
        enc, v, frac = parse_tag(r["tag"])
        row = dict(encoder=enc, views=v, frac=frac, method=r["method"], seed=r["seed"], tag=r["tag"],
                   auroc_f=round(r["auroc_f"], 4), auroc_u=round(r["auroc_u"], 4), auroc_r=round(r["auroc_r"], 4),
                   delta=round(r["delta"], 4), R_pct=round(r["R_pct"], 1), cos_u=round(r["cos_u"], 4),
                   cos_r=round(r["cos_r"], 4), val_mse=r["val_restore_mse"], test_mse=r["test_restore_mse"],
                   best_epoch=r.get("best_epoch", ""), selection=r.get("selection", ""),
                   n_params=r.get("n_params", ""), hparam=r.get("lambda", r.get("reg", "")),
                   minutes=round(r.get("seconds", 0) / 60, 1))
        for lab, val in r["per_label_r"].items():
            row[f"r_{lab}"] = round(val, 4)
        runs.append(row)
    runs.sort(key=sort_key)
    write_csv(os.path.join(a.out_dir, "results_restorers_runs.csv"), runs)

    groups = defaultdict(list)
    for r in runs:
        groups[(r["encoder"], r["views"], r["frac"], r["method"], r["tag"])].append(r)
    summ = []
    for (enc, v, frac, m, tag), rs in groups.items():
        row = dict(encoder=enc, views=v, frac=frac, method=m, n_seeds=len(rs),
                   seeds=" ".join(str(r["seed"]) for r in rs), auroc_f=rs[0]["auroc_f"], auroc_u=rs[0]["auroc_u"])
        for k in ("auroc_r", "delta", "R_pct", "cos_r", "val_mse", "test_mse"):
            mu, sd = msd([r[k] for r in rs])
            nd = 7 if "mse" in k else (2 if k == "R_pct" else 4)
            row[k], row[f"{k}_sd"] = round(mu, nd), (round(sd, nd) if sd == sd else "")
        summ.append(row)
    summ.sort(key=sort_key)
    write_csv(os.path.join(a.out_dir, "results_restorers_summary.csv"), summ)

    print(f"\n{'encoder':<12}{'views':>6}  {'method':<13}{'n':>3}{'AUROC_r':>9}{'±SD':>8}{'R%':>7}"
          f"{'cos_r':>8}{'val MSE':>11}")
    last = None
    for r in summ:
        if (r["encoder"], r["views"], r["frac"]) != last:
            last = (r["encoder"], r["views"], r["frac"])
            fr = f"  (train_frac {r['frac']:g})" if r["frac"] < 1 else ""
            print(f"-- {r['encoder']} {r['views']}-view  E_f={r['auroc_f']:.3f}  E_u={r['auroc_u']:.3f}{fr}")
        sd = f"{r['auroc_r_sd']:.4f}" if r["auroc_r_sd"] != "" else "-"
        print(f"{'':<12}{'':>6}  {r['method']:<13}{r['n_seeds']:>3}{r['auroc_r']:>9.4f}{sd:>8}"
              f"{r['R_pct']:>7.1f}{r['cos_r']:>8.4f}{r['val_mse']:>11.6f}")

    if a.B <= 0:
        return
    by_tag = defaultdict(dict)
    for (enc, v, frac, m, tag), rs in groups.items():
        by_tag[tag][m] = [r["seed"] for r in rs]
    rows = []
    for tag, methods in by_tag.items():
        if not os.path.exists(os.path.join(a.in_dir, "preds", f"{tag}__ref_test.npz")):
            print(f"!! {tag}: ref preds 없음 → bootstrap 생략")
            continue
        t0 = time.time()
        res, S = bootstrap_tag(a.in_dir, tag, methods, a.B, a.seed, a.chunk)
        enc, v, frac = parse_tag(tag)
        ref = res.get(a.ref)
        for m, (pt, bt) in res.items():
            lo, hi = ci(bt)
            row = dict(encoder=enc, views=v, frac=frac, method=m, n_seeds=len(methods.get(m, [0])), n_studies=S,
                       auroc=round(pt, 4), auroc_lo=round(lo, 4), auroc_hi=round(hi, 4), ref=a.ref)
            if ref is not None and m not in (a.ref, "E_f", "E_u"):        # E_u는 identity와 같음
                dlt, db = ref[0] - pt, ref[1] - bt
                se, p_z, _ = p_two_sided(dlt, db)
                if se == 0:                                                 # 예측이 완전히 같음 (예: 두 선택 규칙이 같은 에폭)
                    p_z = 1.0 if dlt == 0 else 0.0
                dlo, dhi = ci(db)
                row.update(delta_ref_minus=round(dlt, 4), delta_lo=round(dlo, 4), delta_hi=round(dhi, 4),
                           delta_se=round(se, 5), p_z=p_z)
            rows.append(row)
        print(f"bootstrap {tag}: {len(res)} representations, B={a.B} ({time.time() - t0:.0f}s)")
    tested = [r for r in rows if "p_z" in r]
    for r, p in zip(tested, holm([r["p_z"] for r in tested])):
        r["p_holm"] = p
    order = {m: i for i, m in enumerate(["E_f", "E_u"] + METHOD_ORDER)}
    rows.sort(key=lambda r: (ENC_ORDER.index(r["encoder"]) if r["encoder"] in ENC_ORDER else 99, -r["views"],
                             -r["frac"], order.get(r["method"], 99)))
    write_csv(os.path.join(a.out_dir, "results_restorers_bootstrap.csv"), rows)
    print(f"\nΔ = AUROC({a.ref}) − AUROC(방법), 검사 단위 bootstrap 95% CI, Holm 보정 ({len(tested)}개 비교)")
    for r in tested:
        print(f"  {r['encoder']:<12}{r['views']:>4}  {r['method']:<13} Δ={r['delta_ref_minus']:+.4f} "
              f"[{r['delta_lo']:+.4f}, {r['delta_hi']:+.4f}]  p_holm={r['p_holm']:.3g}")


if __name__ == "__main__":
    main()
