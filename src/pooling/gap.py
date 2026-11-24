import numpy as np
import torch


def gap_pool(fmap) -> np.ndarray:
    """
    Global Average Pooling.
    fmap : (C, H, W) numpy array or tensor
    returns : (C,) float32 numpy array
    """
    if isinstance(fmap, torch.Tensor):
        return fmap.float().mean(dim=(-2, -1)).numpy()
    return fmap.astype(np.float32).mean(axis=(1, 2))
