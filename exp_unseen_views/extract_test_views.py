"""
exp_unseen_views/extract_test_views.py — 학습에 쓰지 않은 view 수(기본 30·70)의 test feature만 추출.

기존 추출(run_extract_encoder_mp.py)과 전처리·인코더·설정이 완전히 같고, 두 가지만 다르다.
  1) n_views 를 30·70 등으로 바꾼다 (sparse_view_indices가 [0°,180°)를 균등 분할하므로 임의의 view 수 가능).
  2) test 검사만 뽑는다 → master_df.csv에서 split == test 행만 남긴 master_df_test.csv 를 만들어 넣는다.
     test 112,935 슬라이스 → 조건 하나에 약 40–50분 (전체 추출 4–5시간의 15%).
     --n_studies N 이면 test 검사 N개만(seed 0으로 고정 추출) → master_df_test_sub{N}.csv. 출력 폴더는 같아서
     나중에 전체를 뽑으면 이미 뽑은 슬라이스는 건너뛴다(버리는 계산 없음).
기존 config는 건드리지 않고, 이 폴더의 configs/ 에 새 config를 쓴 뒤 같은 추출기를 부른다.
  출력: /data0/hayeon/features_v2/<인코더>/feature_nv30/test/*.npy  (train/ val/ 폴더는 비어 있음)
이미 저장된 .npy는 추출기가 건너뛰므로 끊겨도 같은 명령으로 이어서 실행하면 된다.

실행 위치: 서버 (GPU). 저장소 루트(~/hayeon/ich_cf_clean)에 이 폴더를 그대로 두고 실행. tmux 권장.
Usage (서버):
  tmux new -s unseen
  conda activate hayeon
  cd ~/hayeon/ich_cf_clean
  python exp_unseen_views/extract_test_views.py --dry_run          # config·csv만 만들고 실행할 명령 출력 (확인용)
  python exp_unseen_views/extract_test_views.py --n_studies 500    # 빠른 확인: test 500검사(약 1.7만 슬라이스)만, view당 약 6–7분
  python exp_unseen_views/extract_test_views.py                    # MedSigLIP, 30·70-view
  python exp_unseen_views/extract_test_views.py --views 30 70 --encoders medsiglip_after siglip
  (인코더 이름: medsiglip_after siglip dinov2 resnet50 densenet121)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                     # 저장소 루트 (ich_cf_clean/)
os.chdir(ROOT)


def base_extract_cfg(enc):
    """6-view 추출 config (경로·인코더 설정의 기준). scripts/extract_all.sh 와 같은 이름 규칙."""
    if enc == "medsiglip_after":
        return ROOT / "configs" / "medsiglip_after" / "extract_medsiglip_after_nv6.yaml"
    return ROOT / "configs" / enc / f"extract_{enc}_window_nv6.yaml"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", nargs="+", type=int, default=[30, 70])
    ap.add_argument("--encoders", nargs="+", default=["medsiglip_after"])
    ap.add_argument("--master_csv", default="/data0/hayeon/master_df.csv")
    ap.add_argument("--test_csv", default=None,
                    help="기본 /data0/hayeon/master_df_test.csv (--n_studies면 master_df_test_sub{N}.csv). "
                         "없으면 master_csv에서 split == test 행만 남겨 만든다")
    ap.add_argument("--n_studies", type=int, default=0,
                    help="빠른 확인용: test 검사 중 N개만 무작위(seed 0)로 추출 (0이면 전부 3,262개)")
    ap.add_argument("--dry_run", action="store_true", help="config·csv만 만들고 추출은 하지 않음")
    a = ap.parse_args()

    df = pd.read_csv(a.master_csv)
    te = df[df["split"] == "test"].reset_index(drop=True)
    if a.n_studies > 0:                                   # 검사 단위로 일부만 (슬라이스는 검사째로 전부)
        studies = sorted(te["study_id"].astype(str).unique())
        keep = set(np.random.default_rng(0).choice(studies, min(a.n_studies, len(studies)), replace=False))
        te = te[te["study_id"].astype(str).isin(keep)].reset_index(drop=True)
    test_csv = Path(a.test_csv or ("/data0/hayeon/master_df_test.csv" if a.n_studies <= 0
                                   else f"/data0/hayeon/master_df_test_sub{a.n_studies}.csv"))
    if test_csv.exists():
        old = pd.read_csv(test_csv, usecols=["slice_id"])
        if len(old) != len(te) or set(old["slice_id"]) != set(te["slice_id"]):
            raise SystemExit(f"!! {test_csv} 가 고른 test 검사와 다름 → 지우고 다시 실행")
        print(f"test csv 있음: {test_csv}  ({len(te):,} 슬라이스, {te['study_id'].nunique():,} 검사)")
    else:
        te.to_csv(test_csv, index=False)
        print(f"test csv 만듦: {test_csv}  ({len(te):,} 슬라이스, {te['study_id'].nunique():,} 검사)")

    (HERE / "configs").mkdir(exist_ok=True)
    jobs = []
    for enc in a.encoders:
        src = base_extract_cfg(enc)
        if not src.exists():
            raise SystemExit(f"!! 기준 config 없음: {src}")
        base = yaml.safe_load(open(src, encoding="utf-8"))
        for v in a.views:
            if not 1 <= v <= 180:
                raise SystemExit(f"!! view 수는 1–180: {v}")
            cfg = dict(base)
            cfg["master_csv"] = str(test_csv)
            cfg["n_views"] = v
            for k in ("output_dir", "preview_dir"):
                if k in cfg:
                    new = cfg[k].replace("_nv6", f"_nv{v}")
                    if new == cfg[k]:
                        raise SystemExit(f"!! {src}: {k}에 '_nv6'이 없어 경로를 못 바꿈: {cfg[k]}")
                    cfg[k] = new
            dst = HERE / "configs" / f"extract_{enc}_nv{v}_test.yaml"
            with open(dst, "w", encoding="utf-8") as f:
                f.write(f"# 자동 생성 (exp_unseen_views/extract_test_views.py): {src.name} 에서 n_views·master_csv·경로만 바꿈\n")
                yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
            jobs.append((enc, v, dst, Path(cfg["output_dir"])))
            print(f"config: {dst.relative_to(ROOT)}  →  {cfg['output_dir']}")

    for enc, v, cfg, out in jobs:
        cmd = [sys.executable, "extraction_3ch/run_extract_encoder_mp.py", str(cfg.relative_to(ROOT))]
        print(f"\n==================== extract {enc} nv{v} (test only)\n  $ {' '.join(cmd)}", flush=True)
        if a.dry_run:
            continue
        subprocess.run(cmd, check=True)
        n = len(list((out / "test").glob("*.npy")))
        print(f"  test .npy {n:,} / {len(te):,}" + ("" if n == len(te) else "  ← 부족 (실패한 슬라이스는 추출 로그 확인)"))
    print("\ndone" + ("  (dry run: 추출 안 함)" if a.dry_run else ""))


if __name__ == "__main__":
    main()
