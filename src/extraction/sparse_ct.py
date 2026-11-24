"""
sparse_ct.py — DICOM preprocessing and angular undersampling via Radon transform.

Full CT   : 180 angles → FBP 재구성
Sparse CT : n_views angles → FBP 재구성 → streak noise 발생

전처리 정규화 정책 (고정 window, per-image max 대체):
    FBP(ramp)는 절대 HU를 보존하지 못하므로(DC 소거 + 스케일 불일치),
    HU→[0,1] window를 **재구성 前에** 적용한 뒤 radon/iradon 한다.
    → full/sparse가 동일 스케일을 갖고(per-image max confound 제거),
      brain window 대비도 보존된다.  (reconstruct_windowed / reconstruct_3window)
"""

import numpy as np
from skimage.transform import radon, iradon
from skimage.transform import resize as sk_resize

N_ANGLES_FULL = 180
ANGLES_FULL = np.linspace(0, 180, N_ANGLES_FULL, endpoint=False)


def sparse_view_indices(n_views: int) -> np.ndarray:
    """180각(1° 간격) 격자에서 [0°, 180°)를 n_views 등분한 view 인덱스.

    평행빔에서 180°는 0°와 같은 방향이므로 끝점 180°는 포함하지 않는다.
      n_views=6  → 0, 30, 60, 90, 120, 150
      n_views=23 → 0, 8, 16, 23, 31, … (1° 격자라 7°/8° 간격이 섞이는 것은 불가피)
    (2026-09-30 수정 전: linspace(0, 179, n_views) → 0°와 179°가 함께 들어가 사실상 1 view 중복)
    """
    if not 1 <= n_views <= N_ANGLES_FULL:
        raise ValueError(f"n_views는 1–{N_ANGLES_FULL}: {n_views}")
    idx = np.round(np.linspace(0, N_ANGLES_FULL, n_views, endpoint=False)).astype(int) % N_ANGLES_FULL
    assert len(np.unique(idx)) == n_views, "view 인덱스 중복"
    return idx

# RSNA 표준 3-window (center, width). HU를 [0,1]로 고정 매핑하는 데 사용.
WINDOWS = {
    "brain":    (40.0, 80.0),
    "subdural": (80.0, 200.0),
    "soft":     (40.0, 380.0),
}


# ── HU 변환 (12-bit 보정 포함) ──────────────────────────────────────────────────

def correct_dcm(dcm) -> None:
    """RSNA 12-bit overflow 보정 (in-place). appian42 등 표준 방식."""
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x >= px_mode] = x[x >= px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000.0


def dicom_to_hu(dcm) -> np.ndarray:
    """DICOM → HU (float32). 12-bit overflow 대상이면 correct_dcm 먼저 적용."""
    try:
        bits = int(getattr(dcm, "BitsStored", 0))
        prep = int(getattr(dcm, "PixelRepresentation", 0))
        inter = float(getattr(dcm, "RescaleIntercept", 0))
        if bits == 12 and prep == 0 and inter > -100:
            correct_dcm(dcm)
    except Exception:
        pass
    px = dcm.pixel_array.astype(np.float32)
    slope = float(getattr(dcm, "RescaleSlope", 1))
    intercept = float(getattr(dcm, "RescaleIntercept", 0))
    return px * slope + intercept


def apply_window(hu: np.ndarray, center: float, width: float) -> np.ndarray:
    """HU → [0,1] 고정 window 정규화 (per-image max 아님)."""
    lo = center - width / 2.0
    return np.clip((hu - lo) / width, 0.0, 1.0).astype(np.float32)


# ── 고정 window 재구성 (권장 경로) ──────────────────────────────────────────────

def reconstruct_windowed(hu_2d: np.ndarray,
                         center: float,
                         width: float,
                         n_views: int = None,
                         radon_size: int = 256) -> np.ndarray:
    """
    HU → 고정 window[0,1] → radon → (sparse면 n_views 서브샘플) → iradon(FBP,ramp) → clip[0,1].

    window를 **재구성 前**에 적용하는 것이 핵심.
    FBP가 절대 HU를 보존하지 못하므로 재구성 후 절대 HU window를 적용하면 포화된다.
    full(n_views=None)과 sparse가 동일 window로 정규화돼 스케일이 일관된다.

    Returns: (radon_size, radon_size) float32 in [0,1]
    """
    w = apply_window(hu_2d, center, width)
    small = sk_resize(w, (radon_size, radon_size),
                      anti_aliasing=True, preserve_range=True).astype(np.float32)
    sino = radon(small, theta=ANGLES_FULL, circle=True)
    if n_views is None:
        recon = iradon(sino, theta=ANGLES_FULL, circle=True, filter_name="ramp")
    else:
        idx = sparse_view_indices(n_views)
        recon = iradon(sino[:, idx], theta=ANGLES_FULL[idx],
                       circle=True, filter_name="ramp")
    return np.clip(recon, 0.0, 1.0).astype(np.float32)


def reconstruct_3window(hu_2d: np.ndarray,
                        n_views: int = None,
                        radon_size: int = 256) -> np.ndarray:
    """
    brain/subdural/soft 3-window을 각각 재구성해 RGB로 스택.

    Returns: (radon_size, radon_size, 3) float32 in [0,1]
             채널 순서 = (brain, subdural, soft)
    """
    chans = [reconstruct_windowed(hu_2d, *WINDOWS[k], n_views=n_views,
                                  radon_size=radon_size)
             for k in ("brain", "subdural", "soft")]
    return np.stack(chans, axis=-1)


def reconstruct_hu(hu_2d: np.ndarray,
                   n_views: int = None,
                   radon_size: int = 256) -> np.ndarray:
    """
    [DEPRECATED — 절대 HU window와 함께 쓰지 말 것]
    HU 이미지를 그대로 radon→iradon 한 결과를 반환한다. FBP는 절대 HU를 보존하지
    못하므로 이 출력에 apply_window(절대 HU 기준)를 적용하면 포화된다.
    고정 window 재구성이 필요하면 reconstruct_windowed / reconstruct_3window 를 쓸 것.
    """
    small = sk_resize(hu_2d, (radon_size, radon_size),
                      anti_aliasing=True, preserve_range=True).astype(np.float32)
    sino_full = radon(small, theta=ANGLES_FULL, circle=True)
    if n_views is None:
        recon = iradon(sino_full, theta=ANGLES_FULL, circle=True, filter_name="ramp")
    else:
        us_idx = sparse_view_indices(n_views)
        recon = iradon(sino_full[:, us_idx], theta=ANGLES_FULL[us_idx],
                       circle=True, filter_name="ramp")
    return recon.astype(np.float32)


# ── 레거시 [0,1] 경로 (per-image max, 이전 실험 재현용) ─────────────────────────

def preprocess_dicom(dcm) -> np.ndarray:
    """DICOM → HU → [0, 1]  shape (H, W) float32  (레거시 full-HU 압축)"""
    pixel = dcm.pixel_array.astype(np.float32)
    slope = float(getattr(dcm, "RescaleSlope", 1))
    intercept = float(getattr(dcm, "RescaleIntercept", 0))
    hu = pixel * slope + intercept
    hu = np.clip(hu, -1024, 2048)
    return (hu + 1024) / 3072.0


def full_reconstruct(image_2d: np.ndarray, radon_size: int = 256) -> np.ndarray:
    """[레거시] Full CT: 180 angles FBP 재구성 + per-image max 정규화."""
    small = sk_resize(image_2d, (radon_size, radon_size),
                      anti_aliasing=True).astype(np.float32)
    sino = radon(small, theta=ANGLES_FULL, circle=True)
    recon = iradon(sino, theta=ANGLES_FULL, circle=True, filter_name="ramp")
    recon = np.clip(recon, 0, None)
    r_max = recon.max()
    if r_max > 0:
        recon /= r_max
    return recon.astype(np.float32)


def angular_undersample(image_2d: np.ndarray,
                        acceleration: int,
                        radon_size: int = 256) -> np.ndarray:
    """[레거시] Sparse CT: acceleration 배 각도 감소 후 FBP."""
    n_views = N_ANGLES_FULL // acceleration
    return undersample_nviews(image_2d, n_views, radon_size)


def undersample_nviews(image_2d: np.ndarray,
                       n_views: int,
                       radon_size: int = 256) -> np.ndarray:
    """[레거시] Sparse CT: n_views개 각도 균등 샘플링 후 FBP + per-image max."""
    us_idx = sparse_view_indices(n_views)
    angles_us = ANGLES_FULL[us_idx]

    small = sk_resize(image_2d, (radon_size, radon_size),
                      anti_aliasing=True).astype(np.float32)
    sino_full = radon(small, theta=ANGLES_FULL, circle=True)
    sino_us = sino_full[:, us_idx]
    recon = iradon(sino_us, theta=angles_us, circle=True, filter_name="ramp")
    recon = np.clip(recon, 0, None)
    r_max = recon.max()
    if r_max > 0:
        recon /= r_max
    return recon.astype(np.float32)
