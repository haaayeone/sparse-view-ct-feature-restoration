import numpy as np
import torch


def gem_pool(fmap, p: float = 3.0, eps: float = 1e-6) -> np.ndarray:
    """
    Generalized Mean Pooling.
    fmap : (C, H, W) numpy array or tensor
    p    : GeM exponent (p=1 → GAP, p→∞ → max pooling)
    returns : (C,) float32 numpy array
    """
    if isinstance(fmap, torch.Tensor):
        fmap = fmap.float().clamp(min=eps)
        return fmap.pow(p).mean(dim=(-2, -1)).pow(1.0 / p).numpy()

    fmap = fmap.astype(np.float32)
    fmap = np.clip(fmap, eps, None)
    return np.power(np.mean(np.power(fmap, p), axis=(1, 2)), 1.0 / p)
