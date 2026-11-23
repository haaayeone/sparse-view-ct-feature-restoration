"""
backbone_cnn.py — torchvision ImageNet CNN encoder wrapper (간단 baseline 비교군).

ResNet-50 / DenseNet-121 / EfficientNet-B0 등 고전적 ImageNet CNN을 encoder로 사용.
각 모델의 **내부 global average pooling 출력**(classifier head 직전)을 임베딩으로 쓴다.
→ 외부 pooling 없음, 출력은 (feat_dim,) float16 pooled 벡터로 다른 encoder와 동일 포맷.

전처리는 torchvision weights 가 제공하는 표준 transform(weights.transforms())을 사용
(ImageNet resize/crop/normalize). CT window/3슬라이스 RGB PIL을 그대로 넣으면 된다.
"""

import gc
import time
import numpy as np
import torch
import torch.nn as nn


# model_name -> (constructor_name, weights_enum_name, weights_member, head_attr, feat_dim)
_REGISTRY = {
    "resnet50":        ("resnet50",        "ResNet50_Weights",       "IMAGENET1K_V2", "fc",         2048),
    "densenet121":     ("densenet121",     "DenseNet121_Weights",    "IMAGENET1K_V1", "classifier", 1024),
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights","IMAGENET1K_V1", "classifier", 1280),
}


class TorchvisionCNNExtractor:
    def __init__(self, device: torch.device, model_name: str = "resnet50"):
        from torchvision import models

        if model_name not in _REGISTRY:
            raise ValueError(
                f"지원하지 않는 CNN: {model_name}  (지원: {list(_REGISTRY)})"
            )
        ctor_name, weights_cls, weights_member, head_attr, feat_dim = _REGISTRY[model_name]
        self.device = device
        self.model_name = model_name
        self.feat_dim = feat_dim

        print(f"  CNN baseline 로드 중: {model_name} (ImageNet)")
        weights = getattr(getattr(models, weights_cls), weights_member)
        model = getattr(models, ctor_name)(weights=weights)
        # classifier head 제거 → forward 가 GAP 직후 pooled 벡터를 반환
        setattr(model, head_attr, nn.Identity())
        self.model = model.to(device).eval()
        # 표준 ImageNet 전처리 (resize/centercrop/normalize)
        self.preprocess = weights.transforms()
        print(f"  로드 완료 (feat_dim={feat_dim})")

    def extract_batch(self, pil_images: list, max_retries: int = 3) -> list:
        """list of PIL Images → list of (feat_dim,) float16 arrays."""
        for attempt in range(max_retries):
            try:
                batch = torch.stack(
                    [self.preprocess(img) for img in pil_images]
                ).to(self.device)

                with torch.no_grad():
                    feat = self.model(batch)          # (B, feat_dim)
                result = feat.cpu().to(torch.float16).numpy()

                del batch, feat
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return [result[b] for b in range(result.shape[0])]

            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                gc.collect()
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise
