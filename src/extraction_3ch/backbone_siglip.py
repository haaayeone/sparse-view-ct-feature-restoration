"""
backbone_siglip.py — Vanilla SigLIP (일반 도메인) vision encoder wrapper.

비교군(control) encoder. MedSigLIP과 **아키텍처가 동일**하고 사전학습 데이터만
일반(자연영상)이라, "의료 사전학습의 효과"를 isolate 하는 대조군으로 쓴다.

MedSigLIPExtractor 를 그대로 재사용한다 — 둘 다 SiglipModel 이라
vision_model(...).pooler_output(internal attention pooling) 로직이 동일하다.
출력: (hidden_dim,) float16 pooled embedding. 외부 pooling 없음.
"""

from extraction_3ch.backbone_medsiglip import MedSigLIPExtractor


class SiglipExtractor(MedSigLIPExtractor):
    def __init__(self, device, model_id: str = "google/siglip-so400m-patch14-384"):
        # 부모(__init__)가 'MedSigLIP 로드 중'을 출력하지만 로직은 동일(SiglipModel).
        print(f"  [SigLIP control] model_id={model_id}")
        super().__init__(device, model_id=model_id)
