import numpy as np
import torch


def none_pool(emb) -> np.ndarray:
    """
    Identity pooling — 입력이 이미 (C,) pooled embedding 벡터인 경우 사용.

    MedSigLIP처럼 모델 internal pooling으로 이미 벡터(hidden_dim,)를 저장한 feature는
    외부 pooling이 필요 없으므로 그대로 통과시킨다.

    emb     : (C,) numpy array or tensor  (방어적으로 (C, H, W)면 flatten)
    returns : (C',) float32 numpy array
    """
    if isinstance(emb, torch.Tensor):
        return emb.float().reshape(-1).numpy()
    return emb.astype(np.float32).reshape(-1)
