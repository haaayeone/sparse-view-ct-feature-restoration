"""
collect_test_results.py — run_eval.py --split test 로그에서 원고용 수치를 모아 CSV 두 개로 정리.

로그 파일명: logs/{tag}_{pooling}_test_{YYYYmmdd_HHMMSS}.log
  tag 예) medsiglip_after_r360_nv6, siglip_window_r360_nv180, dinov2_window_r360_nv12 …
같은 tag가 여러 번 있으면 가장 최근 것만 사용.

출력
  results_test_summary.csv   인코더 × 투영 수: E_f, E_u, E_r, Δ, R(%), cos(E_f,E_u), cos(E_f,E_r)   → 원고 표 2
  results_test_perlabel.csv  인코더 × 투영 수 × 라벨: AUROC(full/under/rest), Δ, DeLong p           → 원고 표 3

실행 위치: 서버 (저장소 루트).
Usage (서버):
  cd ~/hayeon/ich_cf_clean
  python collect_test_results.py
  python collect_test_results.py --log_dir logs --out_dir .

로컬로 가져오기 (로컬 PowerShell):
  scp neumous-lab@100.73.97.101:/home/neumous-lab/hayeon/ich_cf_clean/results_test_*.csv "C:\\Users\\user\\Desktop\\MIGT\\01. sparse ct project\\논문작성\\figures\\paper_figs"
"""
import argparse
import csv
import glob
import os
import re

# 기준 360 view 실험의 tag만 모은다 (<인코더>_r360_nv<V>). 예전 180-기준 로그는 --all 로만 포함.
NAME = {"medsiglip_after_r360": "MedSigLIP", "siglip_window_r360": "SigLIP", "dinov2_window_r360": "DINOv2",
        "resnet50_window_r360": "ResNet50", "densenet121_window_r360": "DenseNet121"}
ORDER = list(NAME)
FNAME = re.compile(r"^(?P<enc>.+)_nv(?P<v>\d+)_(?P<pool>[a-z]+)_test_(?P<stamp>\d{8}_\d{6})\.log$")
NUM = r"([-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?)"
PAT = {
    "E_f": re.compile(r"AUROC E_full\s*=\s*" + NUM),
    "E_u": re.compile(r"AUROC E_under\s*=\s*" + NUM),
    "E_r": re.compile(r"AUROC E_restored\s*=\s*" + NUM),
    "cos_u": re.compile(r"cos\(E_full, E_under\)\s*=\s*" + NUM),
    "cos_r": re.compile(r"cos\(E_full, E_restored\)\s*=\s*" + NUM),
}
LABEL = re.compile(r"^\s*(\w+)\s+full=" + NUM + r"\s+under=" + NUM + r"\s+rest=" + NUM)
DELONG = re.compile(r"^\s*(\w+)\s+p=" + NUM)


def parse(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    i = txt.rfind("Stage 3")                     # 마지막 평가 블록만
    blk = txt[i:] if i >= 0 else txt
    out = {k: (float(m.group(1)) if (m := p.search(blk)) else None) for k, p in PAT.items()}
    labels, pvals = {}, {}
    part = "label"
    for line in blk.splitlines():
        if "DeLong" in line:
            part = "delong"
        if part == "label" and (m := LABEL.search(line)):
            labels[m.group(1)] = tuple(float(m.group(k)) for k in (2, 3, 4))
        elif part == "delong" and (m := DELONG.search(line)):
            pvals[m.group(1)] = float(m.group(2))
    return out, labels, pvals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--all", action="store_true", help="r360 이외의 tag(예전 180-기준 등)도 포함")
    ap.add_argument("--out_dir", default=".")
    a = ap.parse_args()

    latest = {}
    for f in glob.glob(os.path.join(a.log_dir, "*_test_*.log")):
        m = FNAME.match(os.path.basename(f))
        if not m:
            continue
        if not a.all and m["enc"] not in NAME:
            continue
        key = (m["enc"], int(m["v"]))
        if key not in latest or m["stamp"] > latest[key][0]:
            latest[key] = (m["stamp"], f)
    if not latest:
        raise SystemExit(f"{a.log_dir}에 *_test_*.log 가 없음")

    rank = lambda e: ORDER.index(e) if e in ORDER else len(ORDER)
    keys = sorted(latest, key=lambda k: (rank(k[0]), -k[1]))
    summ, perl = [], []
    for enc, v in keys:
        s, labels, pvals = parse(latest[(enc, v)][1])
        ef, eu, er = s["E_f"], s["E_u"], s["E_r"]
        ok = None not in (ef, eu, er)
        summ.append(dict(encoder=NAME.get(enc, enc), views=v,
                         E_f=ef, E_u=eu, E_r=er,
                         delta=round(er - eu, 4) if ok else None,
                         R_pct=round(100 * (er - eu) / (ef - eu), 1) if ok and ef != eu else None,
                         cos_u=s["cos_u"], cos_r=s["cos_r"], log=os.path.basename(latest[(enc, v)][1])))
        for lab, (f_, u_, r_) in labels.items():
            perl.append(dict(encoder=NAME.get(enc, enc), views=v, label=lab, full=f_, under=u_, rest=r_,
                             delta=round(r_ - u_, 4), delong_p=pvals.get(lab)))

    os.makedirs(a.out_dir, exist_ok=True)
    for name, rows in (("results_test_summary.csv", summ), ("results_test_perlabel.csv", perl)):
        path = os.path.join(a.out_dir, name)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"saved {path}  ({len(rows)} rows)")

    # 서버 Python 3.10 호환: f-string 안에 같은 따옴표를 중첩하지 않음
    def fmt(x, spec=".3f", width=6):
        return "-".rjust(width) if x is None else format(x, spec).rjust(width)

    print("\n" + "Encoder".ljust(12) + "Views".rjust(6) + "E_f".rjust(7) + "E_u".rjust(7) + "E_r".rjust(7)
          + "Δ".rjust(8) + "R%".rjust(7) + "cos_u".rjust(7) + "cos_r".rjust(7))
    for r in summ:
        print(r["encoder"].ljust(12) + str(r["views"]).rjust(6) + " " + fmt(r["E_f"]) + " " + fmt(r["E_u"])
              + " " + fmt(r["E_r"]) + " " + fmt(r["delta"], "+.3f", 7) + " " + fmt(r["R_pct"], ".1f")
              + " " + fmt(r["cos_u"]) + " " + fmt(r["cos_r"]))


if __name__ == "__main__":
    main()
