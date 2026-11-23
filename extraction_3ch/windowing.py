"""
windowing.py — HU windowing 기반 3채널 입력 생성 (신규 실험 모듈).

기존 인접-슬라이스 3채널 방식 대신, **현재 슬라이스 한 장**에
서로 다른 CT window (brain / subdural / bone 등)를 적용해 RGB 3채널을 만든다.

기존 파일은 수정하지 않는다. extraction.sparse_ct 는 read-only 로 import 만 하며,
windowing 모드에 필요한 "max 정규화를 하지 않은" FBP 재구성은 여기서 자체 구현한다.
(sparse_ct.full_reconstruct / undersample_nviews 는 마지막에 recon/=max 를 하므로
 절대 HU 스케일이 사라져 windowing 에 쓸 수 없기 때문.)

전처리 매핑은 sparse_ct.preprocess_dicom 과 동일:
    HU → clip(-1024, 2048) → proxy = (HU + 1024) / 3072  ∈ [0, 1]
따라서 역변환:  HU ≈ proxy * 3072 - 1024
"""

import numpy as np
from PIL import Image
from skimage.transform import radon, iradon
from skimage.transform import resize as sk_resize

# read-only import (수정하지 않음)
from extraction.sparse_ct import ANGLES_FULL, N_ANGLES_FULL, sparse_view_indices

# preprocess_dicom 의 HU clip 범위와 일치해야 함
HU_MIN = -1024.0
HU_MAX = 2048.0
HU_RANGE = HU_MAX - HU_MIN  # 3072.0

# ICH 표준 3-window  (level, width) in HU
# brain     : 출혈/실질 대비
# subdural   : 얇은 경막하 출혈 강조
# bone       : 골절/석회화 (streak 에 민감하므로 필요시 soft-tissue 로 교체)
DEFAULT_WINDOWS = [
    (40.0, 80.0),     # brain
    (80.0, 200.0),    # subdural
    (600.0, 2800.0),  # bone
]


# ── FBP 재구성 (max 정규화 없음, HU-proxy 스케일 유지) ──────────────────────────

def reconstruct_hu_proxy(image_2d: np.ndarray,
                         radon_size: int = 256,
                         n_views: int = None,
                         acceleration: int = None) -> np.ndarray:
    """
    preprocess_dicom 의 [0,1] HU-proxy 이미지를 받아 FBP 재구성.
    **per-image max 정규화를 하지 않아** HU 스케일이 보존된다.

    n_views / acceleration 둘 다 None  → 180 view (0°–179°, 1° 간격) 재구성 (= n_views=180)
    n_views ≤ 180                      → 1° 격자에서 해당 view 수로 sparse 재구성
                                         (sparse_view_indices: [0°,180°)를 균등 분할, 예 6 view → 0,30,…,150°)
    n_views > 180 (예: 360)            → [0°,180°) 균등 분할 (360 → 0.5° 간격). 기준(reference) 영상용.
                                         1° 격자의 모든 각도가 0.5° 격자에 포함되고 Radon 투영은 각도별로
                                         독립이므로, sparse 조건(≤180)은 기준을 360으로 바꿔도 그대로다.
    acceleration 지정                  → N_ANGLES_FULL // acceleration view

    Returns: (radon_size, radon_size) float32  (대략 [0,1] proxy 스케일)
    """
    small = sk_resize(image_2d, (radon_size, radon_size),
                      anti_aliasing=True).astype(np.float32)

    if n_views is None and acceleration is not None:
        n_views = N_ANGLES_FULL // acceleration

    if n_views is not None and n_views > N_ANGLES_FULL:
        # dense reference (360 view 등): 같은 FBP 절차, 각도만 더 촘촘하게
        th = np.linspace(0.0, 180.0, n_views, endpoint=False)
        recon = iradon(radon(small, theta=th, circle=True), theta=th,
                       circle=True, filter_name="ramp")
        return np.clip(recon, 0.0, None).astype(np.float32)

    sino_full = radon(small, theta=ANGLES_FULL, circle=True)

    if n_views is None:
        # full reconstruction
        recon = iradon(sino_full, theta=ANGLES_FULL,
                       circle=True, filter_name="ramp")
    else:
        us_idx = sparse_view_indices(n_views)
        angles_us = ANGLES_FULL[us_idx]
        sino_us = sino_full[:, us_idx]
        recon = iradon(sino_us, theta=angles_us,
                       circle=True, filter_name="ramp")

    recon = np.clip(recon, 0.0, None)   # 음수만 제거, max 정규화는 하지 않음
    return recon.astype(np.float32)


# ── Windowing ──────────────────────────────────────────────────────────────────

def proxy_to_hu(proxy: np.ndarray) -> np.ndarray:
    """preprocess_dicom 의 [0,1] proxy → 근사 HU."""
    return proxy.astype(np.float32) * HU_RANGE + HU_MIN


def apply_window(hu: np.ndarray, level: float, width: float) -> np.ndarray:
    """HU 이미지에 (level, width) window 적용 → [0,1] float32."""
    lo = level - width / 2.0
    hi = level + width / 2.0
    out = (hu - lo) / max(hi - lo, 1e-6)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def make_window_stack(recon_proxy: np.ndarray,
                      windows: list = None) -> np.ndarray:
    """
    HU-proxy 재구성 이미지 → 여러 window 적용 → (H, W, 3) [0,1] float32.

    windows 가 3개 미만이면 마지막 채널을 복제해 3채널을 맞춘다.
    """
    if windows is None:
        windows = DEFAULT_WINDOWS
    hu = proxy_to_hu(recon_proxy)
    chans = [apply_window(hu, lvl, wid) for (lvl, wid) in windows]
    while len(chans) < 3:
        chans.append(chans[-1])
    return np.stack(chans[:3], axis=-1)  # (H, W, 3)


def window_to_pil(recon_proxy: np.ndarray,
                  windows: list = None) -> Image.Image:
    """HU-proxy 재구성 이미지 → 3-window RGB PIL Image."""
    stack = make_window_stack(recon_proxy, windows)
    stack = (stack * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(stack, mode="RGB")


def parse_windows(cfg_windows) -> list:
    """
    YAML config 의 windows 값을 [(level, width), ...] 로 변환.
    None 이면 DEFAULT_WINDOWS 반환.

    허용 형식:
        windows:
          - [40, 80]
          - [80, 200]
          - [600, 2800]
    """
    if not cfg_windows:
        return DEFAULT_WINDOWS
    return [(float(lvl), float(wid)) for (lvl, wid) in cfg_windows]
