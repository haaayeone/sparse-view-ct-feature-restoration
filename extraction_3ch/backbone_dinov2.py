"""
backbone_dinov2.py — DINOv2 (일반 도메인, self-supervised) vision encoder wrapper.

비교군(control) encoder. 일반 도메인 + SSL 패러다임의 강baseline.
VLM(MedSigLIP/SigLIP)이 아닌 self-supervised 계열이라, "의료 여부"와 "VLM 여부"를
분리해서 볼 수 있게 해준다.

DINOv2 Model 의 pooler_output(= LayerNorm 후 CLS token)을 이미지 임베딩으로 사용.
출력: (hidden_dim,) float16 pooled embedding. 외부 pooling 없음.
"""

import gc
import time
import numpy as np
import torch


class Dinov2Extractor:
    def __init__(self, device: torch.device,
                 model_id: str = "facebook/dinov2-large"):
        from transformers import AutoModel, AutoImageProcessor

        self.device = device
        print(f"  DINOv2 로드 중: {model_id}")
        self.model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
        ).to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        print("  DINOv2 로드 완료")

    def extract_batch(self, pil_images: list, max_retries: int = 3) -> list:
        """
        list of PIL Images → list of (hidden_dim,) float16 arrays.
        DINOv2 pooler_output(CLS token) 사용.
        """
        for attempt in range(max_retries):
            try:
                pixel_values = self.processor(
                    images=pil_images, return_tensors="pt",
                )["pixel_values"].to(self.device, dtype=torch.bfloat16)

                with torch.no_grad():
                    out = self.model(pixel_values=pixel_values)

                # Dinov2Model: pooler_output = LayerNorm(CLS token), (B, hidden_dim)
                embeds = out.pooler_output
                result = embeds.cpu().to(torch.float16).numpy()

                del pixel_values, out, embeds
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
