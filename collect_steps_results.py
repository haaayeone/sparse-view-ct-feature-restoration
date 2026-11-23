"""
collect_steps_results.py — run_eval.py --steps S 로그를 모아 표 4용 CSV로 정리.

로그 파일명: logs/{tag}_{pooling}_test_steps{S}_{YYYYmmdd_HHMMSS}.log   (scripts/steps_sweep.sh 가 생성)
같은 (tag, S)가 여러 번 있으면 가장 최근 것만 사용.

출력: results_steps.csv  (encoder, views, steps, E_f, E_u, E_r, cos_r)  + 화면에 views × steps 표

실행 위치: 서버 (저장소 루트).
Usage (서버):
  cd ~/hayeon/ich_cf_clean
  python collect_steps_results.py
로컬로 가져오기 (로컬 PowerShell):
  scp neumous-lab@100.73.97.101:/home/neumous-lab/hayeon/ich_cf_clean/results_steps.csv "C:\\Users\\user\\Desktop\\MIGT\\01. sparse ct project\\논문작성\\figures\\paper_figs\\results_steps.csv"
"""
import argparse
import csv
import glob
import os
import re

# 기준 360 view 실험의 tag만 모은다 (<인코더>_r360_nv<V>). 예전 180-기준 로그는 --all 로만 포함.
NAME = {"medsiglip_after_r360": "MedSigLIP", "siglip_window_r360": "SigLIP", "dinov2_window_r360": "DINOv2",
        "resnet50_window_r360": "ResNet50", "densenet121_window_r360": "DenseNet121"}
FNAME = re.compile(r"^(?P<enc>.+)_nv(?P<v>\d+)_(?P<pool>[a-z]+)_test_steps(?P<s>\d+)_(?P<stamp>\d{8}_\d{6})\.log$")
NUM = r"([-+]?\d*\.\d+|[-+]?\d+)"
PAT = {"E_f": re.compile(r"AUROC E_full\s*=\s*" + NUM), "E_u": re.compile(r"AUROC E_under\s*=\s*" + NUM),
       "E_r": re.compile(r"AUROC E_restored\s*=\s*" + NUM),
       "cos_r": re.compile(r"cos\(E_full, E_restored\)\s*=\s*" + NUM)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--all", action="store_true", help="r360 이외의 tag(예전 180-기준 등)도 포함")
    ap.add_argument("--out", default="results_steps.csv")
    a = ap.parse_args()

    latest = {}
    for f in glob.glob(os.path.join(a.log_dir, "*_test_steps*_*.log")):
        m = FNAME.match(os.path.basename(f))
        if not m:
            continue
        if not a.all and m["enc"] not in NAME:
            continue
        key = (m["enc"], int(m["v"]), int(m["s"]))
        if key not in latest or m["stamp"] > latest[key][0]:
            latest[key] = (m["stamp"], f)
    if not latest:
        raise SystemExit(f"{a.log_dir}에 *_test_steps*_*.log 가 없음 — scripts/steps_sweep.sh 먼저 실행")

    rows = []
    for (enc, v, s) in sorted(latest, key=lambda k: (k[0], -k[1], k[2])):
        txt = open(latest[(enc, v, s)][1], encoding="utf-8", errors="replace").read()
        blk = txt[txt.rfind("Stage 3"):] if "Stage 3" in txt else txt
        vals = {}
        for k, p in PAT.items():
            m = p.search(blk)
            vals[k] = float(m.group(1)) if m else None
        rows.append(dict(encoder=NAME.get(enc, enc), views=v, steps=s, **vals))

    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"saved {a.out}  ({len(rows)} rows)\n")

    steps = sorted({r["steps"] for r in rows})
    print("Encoder".ljust(12) + "Views".rjust(6) + "".join(("E_r@" + str(s)).rjust(10) for s in steps))
    for (enc, v) in sorted({(r["encoder"], r["views"]) for r in rows}, key=lambda k: (k[0], -k[1])):
        cells = []
        for s in steps:
            r = next((x for x in rows if x["encoder"] == enc and x["views"] == v and x["steps"] == s), None)
            cells.append("-".rjust(10) if r is None or r["E_r"] is None else format(r["E_r"], ".3f").rjust(10))
        print(enc.ljust(12) + str(v).rjust(6) + "".join(cells))


if __name__ == "__main__":
    main()
