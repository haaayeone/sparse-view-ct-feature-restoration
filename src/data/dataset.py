"""
dataset.py — Feature map dataset with on-the-fly pooling.

Feature maps (512, H', W') float16을 로드하고
지정된 pooling 방식으로 (C',) float32 embedding을 반환한다.
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pooling import get_pooling_fn


class CTFeatureMapDataset(Dataset):
    """
    CSV 기반 CT feature map 데이터셋.

    Parameters
    ----------
    csv_df        : master DataFrame (slice_id, split, label_cols...)
    full_dir      : features_full/{split}/{slice_id}.npy
    under_dir     : features_under_xN/{split}/{slice_id}.npy
    label_cols    : list of label column names
    split         : "train" | "val" | "test"
    pooling       : "gap" | "gem" | "grid"
    **pool_kwargs : gem_p, grid_size 등 pooling 파라미터
    """

    def __init__(self, csv_df, full_dir: str, under_dir: str,
                 label_cols: list, split: str,
                 pooling: str = "gap", **pool_kwargs):
        self.df        = csv_df[csv_df["split"] == split].reset_index(drop=True)
        self.full_dir  = Path(full_dir) / split
        self.under_dir = Path(under_dir) / split
        self.label_cols = label_cols
        self.pool_fn   = get_pooling_fn(pooling, **pool_kwargs)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sid = row["slice_id"]

        fmap_full  = np.load(self.full_dir  / f"{sid}.npy").astype(np.float32)
        fmap_under = np.load(self.under_dir / f"{sid}.npy").astype(np.float32)

        emb_full  = torch.tensor(self.pool_fn(fmap_full),  dtype=torch.float32)
        emb_under = torch.tensor(self.pool_fn(fmap_under), dtype=torch.float32)
        labels    = torch.tensor(
            row[self.label_cols].values.astype(np.float32), dtype=torch.float32
        )
        return emb_full, emb_under, labels
