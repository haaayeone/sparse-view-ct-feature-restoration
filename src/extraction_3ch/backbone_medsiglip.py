"""
backbone_medsiglip.py — MedSigLIP vision encoder wrapper for CT feature extraction.

3장의 인접 슬라이스를 RGB처럼 쌓아 (H, W, 3) PIL Image로 변환 후
MedSigLIP(google/medsiglip-448)에 입력하여 **internal attention pooling**으로
이미지 임베딩 벡터를 추출한다.

출력: (hidden_dim,) float16  — feature map이 아니라 pooled embedding 벡터.
      MedSigLIP 내부 attention pooling head(pooler_output)를 그대로 사용하므로
      이후 GAP / GeM / Grid 등 외부 pooling을 적용하지 않는다.
"""

import gc
import time
import numpy as np
import torch
from PIL import Image


class MedSigLIPExtractor:
    def __init__(self, device: torch.device,
                 model_id: str = "google/medsiglip-448"):
        from transformers import AutoModel, AutoImageProcessor

        self.device = device
        print(f"  MedSigLIP 로드 중: {model_id}")
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
        )
        self.model.eval()
        # 이미지 feature만 추출하므로 텍스트 tokenizer는 로드하지 않는다.
        # (AutoProcessor는 SiglipTokenizer→sentencepiece 의존성을 끌어옴)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        print("  MedSigLIP 로드 완료")

    @staticmethod
    def to_pil(prev: np.ndarray, curr: np.ndarray,
               next_: np.ndarray) -> Image.Image:
        """
        3장의 (H, W) float32 [0,1] 슬라이스 → PIL RGB Image

        prev  : 이전 슬라이스
        curr  : 현재 슬라이스
        next_ : 다음 슬라이스
        """
        stack = np.stack([prev, curr, next_], axis=-1)   # (H, W, 3)
        stack = (stack * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(stack, mode="RGB")

    def extract_feature_map(self, pil_image: Image.Image,
                            max_retries: int = 3) -> np.ndarray:
        """
        PIL Image → MedSigLIP → pooled embedding (hidden_dim,) float16
        (이름은 backbone_medgemma와 호환을 위해 유지하지만, 반환은 벡터다.)
        """
        return self.extract_batch([pil_image], max_retries)[0]

    def extract_batch(self, pil_images: list,
                      max_retries: int = 3) -> list:
        """
        list of PIL Images → list of (hidden_dim,) float16 arrays

        MedSigLIP의 internal attention pooling(pooler_output)을 사용한다.
        get_image_features()가 곧 vision pooling head의 출력 (B, hidden_dim).
        """
        for attempt in range(max_retries):
            try:
                pixel_values = self.processor(
                    images=pil_images,
                    return_tensors="pt",
                )["pixel_values"].to(self.device, dtype=torch.bfloat16)

                with torch.no_grad():
                    # vision tower의 internal attention pooling head(pooler_output) 사용.
                    # get_image_features는 버전에 따라 텐서가 아닌 output 객체를 반환할 수
                    # 있어(=.cpu() 없음) 쓰지 않는다.
                    out = self.model.vision_model(pixel_values=pixel_values)

                # 방어적: 텐서면 그대로, output 객체면 pooler_output 추출 (B, hidden_dim)
                embeds = out if isinstance(out, torch.Tensor) else out.pooler_output

                result = embeds.cpu().to(torch.float16).numpy()  # (B, hidden_dim)

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
