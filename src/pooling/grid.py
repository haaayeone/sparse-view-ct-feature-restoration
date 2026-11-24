import numpy as np


def grid_pool(fmap, grid_size: int = 2, pool_fn: str = "avg") -> np.ndarray:
    """
    Grid Pooling: spatial map을 grid_size×grid_size 셀로 분할,
    각 셀을 pooling 후 concatenate.

    fmap      : (C, H, W) numpy array
    grid_size : 분할 수 (2 → 2×2=4 cells)
    pool_fn   : "avg" | "max"

    returns : (C * grid_size^2,) float32 numpy array
    """
    C, H, W = fmap.shape
    fmap = fmap.astype(np.float32)

    h_step = H // grid_size
    w_step = W // grid_size
    fmap_crop = fmap[:, : h_step * grid_size, : w_step * grid_size]

    cells = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell = fmap_crop[
                :,
                i * h_step : (i + 1) * h_step,
                j * w_step : (j + 1) * w_step,
            ]
            if pool_fn == "avg":
                cells.append(cell.mean(axis=(1, 2)))
            else:
                cells.append(cell.max(axis=(1, 2)))

    return np.concatenate(cells, axis=0)
