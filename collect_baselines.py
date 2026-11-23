"""
collect_baselines.py — logs/baseline_*.log 의 SUMMARY 블록을 파싱해 CSV로 정리.

train_baselines.py 가 남긴 로그(파일명: baseline_{tag}_{pooling}_{split}_{ts}.log)에서
identity / ridge / mlp 행을 추출한다. 같은 (tag, baseline)은 최신 timestamp만 유지.

Usage:
    python collect_baselines.py --split test --out results_baselines.csv
"""

import argparse
import csv
import glob
import os
import re

FNAME_RE = re.compile(
    r"^baseline_(?P<tag>.+?)_(?P<pooling>[a-z]+)_(?P<split>test|val)_(?P<ts>\d{8}_\d{6})\.log$")
HEAD_RE = re.compile(
    r"SUMMARY\s+tag=(?P<tag>\S+)\s+split=(?P<split>\S+)\s+"
    r"\(E_full=(?P<full>[\d.]+),\s*E_under=(?P<under>[\d.]+)\)")
ROW_RE = re.compile(
    r"-\s+(?P<name>identity|ridge|mlp)\s+(?P<params>[\d,]+)\s+"
    r"(?P<auroc>[\d.]+)\s+(?P<delta>[+\-][\d.]+)\s+(?P<rec>[\d.\-]+|nan)\s+(?P<cos>[\d.\-]+)")
TAG_RE = re.compile(r"^(?P<enc>.+?)_(?:nv(?P<v1>\d+)|(?P<v2>\d+)view)$")


def parse_file(path):
    fname = os.path.basename(path)
    m = FNAME_RE.match(fname)
    if not m:
        return []
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    heads = list(HEAD_RE.finditer(text))
    if not heads:
        return []          # SUMMARY 없음(미완료)
    h = heads[-1]          # 마지막 SUMMARY 사용
    tail = text[h.start():]

    tm = TAG_RE.match(h.group("tag"))
    if not tm:
        return []
    encoder = tm.group("enc")
    views = int(tm.group("v1") or tm.group("v2"))

    rows = []
    for r in ROW_RE.finditer(tail):
        rec = r.group("rec")
        rows.append({
            "encoder": encoder,
            "views": views,
            "baseline": r.group("name"),
            "params": int(r.group("params").replace(",", "")),
            "full": float(h.group("full")),
            "under": float(h.group("under")),
            "restored": float(r.group("auroc")),
            "delta": float(r.group("delta")),
            "recovery_pct": float(rec) if rec not in ("nan", "-") else float("nan"),
            "cos_fr": float(r.group("cos")),
            "split": m.group("split"),
            "timestamp": m.group("ts"),
            "logfile": fname,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results_baselines.csv")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.log_dir, "baseline_*.log"))):
        rows += [r for r in parse_file(path) if r["split"] == args.split]

    if not rows:
        print(f"[!] split={args.split} baseline 결과 없음 ({args.log_dir}/baseline_*.log)")
        return

    # (encoder, views, baseline)별 최신 timestamp만
    best = {}
    for r in rows:
        k = (r["encoder"], r["views"], r["baseline"])
        if k not in best or r["timestamp"] > best[k]["timestamp"]:
            best[k] = r
    rows = sorted(best.values(), key=lambda r: (r["encoder"], r["views"],
                                                ["identity", "ridge", "mlp"].index(r["baseline"])))

    cols = ["encoder", "views", "baseline", "params", "full", "under", "restored",
            "delta", "recovery_pct", "cos_fr", "timestamp", "logfile"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"저장: {args.out}  ({len(rows)} rows)")
    for r in rows:
        print(f"  {r['encoder']:>20s} nv{r['views']:<3d} {r['baseline']:<9s} "
              f"E_r={r['restored']:.4f}  Δ={r['delta']:+.4f}  rec={r['recovery_pct']:.1f}%")


if __name__ == "__main__":
    main()
