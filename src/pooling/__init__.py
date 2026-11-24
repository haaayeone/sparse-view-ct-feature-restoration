import numpy as np
from .gap import gap_pool
from .gem import gem_pool
from .grid import grid_pool
from .none import none_pool


def get_pooling_fn(name: str, **kwargs):
    """
    name : "none" | "gap" | "gem" | "grid"
    kwargs:
      gem  → gem_p (float, default 3.0)
      grid → grid_size (int, default 2)

    Returns a callable: (C, H, W) ndarray → (C',) ndarray
    "none"은 입력이 이미 (C,) pooled embedding(MedSigLIP internal pooling)일 때 사용.
    """
    if name == "none":
        return none_pool
    elif name == "gap":
        return gap_pool
    elif name == "gem":
        p = kwargs.get("gem_p", 3.0)
        return lambda fmap: gem_pool(fmap, p=p)
    elif name == "grid":
        grid_size = kwargs.get("grid_size", 2)
        return lambda fmap: grid_pool(fmap, grid_size=grid_size)
    else:
        raise ValueError(f"Unknown pooling '{name}'. Choose: none | gap | gem | grid")
