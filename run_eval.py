"""
run_eval.py — 저장된 weight로 Stage 3 평가만 실행.

Usage (서버, 저장소 루트에서):
    cd ~/hayeon/ich_cf_clean
    python run_eval.py configs/medsiglip_after/train_medsiglip_after_nv6.yaml --split test
    # 샘플링 단계 수 바꿔 평가 (원고 표 4) — 로그 이름에 _steps{S}_ 가 붙어 기본 결과와 섞이지 않음
    python run_eval.py configs/medsiglip_after/train_medsiglip_after_nv6.yaml --split test --steps 5
    # 한 번에: bash scripts/steps_sweep.sh   →   python collect_steps_results.py
"""

import argparse
import os
import sys
import logging
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion import ColdDiffusionMLP
from probe import LinearProbe
from data import CTFeatureMapDataset
from utils import get_config
from train_diffusion import (
    load_to_ram, mean_cos_sim, compute_auroc,
    compute_auroc_per_label, eval_stage3, set_seed
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def run_eval(opt, pooling, split="val", steps_override=None):
    import pandas as pd

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(getattr(opt, "seed", 42)))
    tag        = getattr(opt, "tag", "ct")
    label_cols = getattr(opt, "label_cols",
                         ["epidural", "intraparenchymal", "intraventricular",
                          "subarachnoid", "subdural", "any"])
    if steps_override is not None:
        opt.diffusion_sample_steps = int(steps_override)

    # split별 로그 파일 → collect_test_results.py가 자동 수집 (split·timestamp 포함).
    # --steps 로 단계 수를 바꾼 평가는 _steps{S}_ 를 붙여 기본(1단계) test 결과와 섞이지 않게 한다.
    os.makedirs("logs", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    steps_tag = f"_steps{int(steps_override)}" if steps_override is not None else ""
    fh = logging.FileHandler(f"logs/{tag}_{pooling}_{split}{steps_tag}_{stamp}.log",
                             mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", "%m/%d/%Y %H:%M:%S"))
    logging.getLogger().addHandler(fh)

    try:
        pool_kwargs = {}
        if pooling == "gem":
            pool_kwargs["gem_p"] = getattr(opt, "gem_p", 3.0)
        elif pooling == "grid":
            pool_kwargs["grid_size"] = getattr(opt, "grid_size", 2)

        logger.info(f"pooling={pooling}  tag={tag}  split={split}  "
                    f"sample_steps={getattr(opt, 'diffusion_sample_steps', 1)}  {pool_kwargs}")

        df = pd.read_csv(opt.csv_path)
        val_ds = CTFeatureMapDataset(df, opt.full_feature_dir, opt.under_feature_dir,
                                     label_cols, split, pooling, **pool_kwargs)

        logger.info(f"Loading {split} data...")
        val_full, val_under, val_Y = load_to_ram(val_ds)
        emb_dim = val_full.size(1)
        logger.info(f"emb_dim={emb_dim}  N={val_full.size(0)}")

        T     = getattr(opt, "diffusion_T",      100)
        depth = getattr(opt, "diffusion_depth",  4)
        steps = getattr(opt, "diffusion_sample_steps", 1)

        diff_ckpt = f"weights/diffusion_{tag}_{pooling}_T{T}.best.pt"
        if not os.path.exists(diff_ckpt):
            logger.error(f"Diffusion ckpt 없음: {diff_ckpt}")
            return

        sd = torch.load(diff_ckpt, map_location="cpu", weights_only=True)
        hidden = sd["out.weight"].shape[1]
        logger.info(f"Diffusion loaded: {diff_ckpt}  (hidden={hidden})")

        probe_ckpt = f"weights/linprobe_{tag}_{pooling}.best.pt"
        if not os.path.exists(probe_ckpt):
            logger.error(f"Probe ckpt 없음: {probe_ckpt}")
            return

        probe = LinearProbe(emb_dim, len(label_cols)).to(device)
        probe.load_state_dict(torch.load(probe_ckpt, map_location=device, weights_only=True))
        probe.eval()
        logger.info(f"Probe loaded: {probe_ckpt}")

        opt.diffusion_hidden = hidden
        opt.pooling = pooling

        logger.info(f"\n{'='*60}\nEvaluating on [{split}] set\n{'='*60}")
        _, auroc_full, auroc_under, auroc_rest = eval_stage3(
            diff_ckpt, emb_dim, probe,
            val_full, val_under, val_Y, label_cols, opt, device
        )

        # collect_results가 파싱하는 DONE 블록 (콜론 형식)
        logger.info(
            f"\n{'='*60}\n"
            f"DONE  tag={tag}  pooling={pooling}  split={split}\n"
            f"  AUROC E_full     : {auroc_full:.4f}\n"
            f"  AUROC E_under    : {auroc_under:.4f}\n"
            f"  AUROC E_restored : {auroc_rest:.4f}\n"
            f"  Improvement      : {auroc_rest - auroc_under:+.4f}\n"
            f"{'='*60}"
        )
    finally:
        logging.getLogger().removeHandler(fh)
        fh.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="train yaml 경로")
    parser.add_argument("--pooling", default=None,
                        help="pooling 목록 (쉼표 구분, 미지정 시 config의 pooling 사용)")
    parser.add_argument("--split", default="val",
                        help="평가할 데이터 split (val | test, 기본: val)")
    parser.add_argument("--steps", type=int, default=None,
                        help="역방향 샘플링 단계 수 (미지정 시 config의 diffusion_sample_steps, 기본 1)")
    args = parser.parse_args()

    opt = get_config(args.config)
    poolings = args.pooling or getattr(opt, "pooling", "gap")
    for pooling in [p.strip() for p in poolings.split(",")]:
        run_eval(opt, pooling, split=args.split, steps_override=args.steps)


if __name__ == "__main__":
    main()
