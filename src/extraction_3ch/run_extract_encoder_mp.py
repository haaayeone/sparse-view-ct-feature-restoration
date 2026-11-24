"""
run_extract_encoder_mp.py — run_extract_encoder.py의 ProcessPool 병렬판.

전처리 결과는 원본과 동일(window: reconstruct_hu_proxy+window_to_pil / adjacent: recon_adjacent).
바뀐 것: ThreadPool → ProcessPoolExecutor(48코어), 진행중 작업 상한(OOM 방지), atomic save(잘린 npy 방지).
  - 워커(프로세스): DICOM→recon→PIL (CPU only, 모델 없음)
  - 메인: GPU 배치 추론 + 저장

Usage:
    python extraction_3ch/run_extract_encoder_mp.py configs/siglip/extract_siglip_window_nv90.yaml
"""

import gc
import os
import sys
import numpy as np
import pandas as pd
import pydicom
import torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parent.parent))
try:
    pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE
except Exception:
    pass

from utils import get_config
from extraction.sparse_ct import (
    preprocess_dicom, full_reconstruct, angular_undersample, undersample_nviews,
)
from extraction_3ch.windowing import reconstruct_hu_proxy, window_to_pil, parse_windows


def build_extractor(encoder, device, model_id):
    encoder = encoder.lower()
    if encoder == "medsiglip":
        from extraction_3ch.backbone_medsiglip import MedSigLIPExtractor
        return MedSigLIPExtractor(device, model_id=model_id or "google/medsiglip-448")
    if encoder == "siglip":
        from extraction_3ch.backbone_siglip import SiglipExtractor
        return SiglipExtractor(device, model_id=model_id or "google/siglip-so400m-patch14-384")
    if encoder == "dinov2":
        from extraction_3ch.backbone_dinov2 import Dinov2Extractor
        return Dinov2Extractor(device, model_id=model_id or "facebook/dinov2-large")
    if encoder in ("resnet50", "densenet121", "efficientnet_b0"):
        from extraction_3ch.backbone_cnn import TorchvisionCNNExtractor
        return TorchvisionCNNExtractor(device, model_name=encoder)
    raise ValueError(f"알 수 없는 encoder: {encoder}")


def build_slice_order(df):
    df = df.copy()
    df["slice_order"] = pd.to_numeric(df["slice_order"], errors="coerce").fillna(0)
    prev_ids, next_ids = [], []
    for _, group in df.groupby("series_uid", sort=False):
        ids = group.sort_values("slice_order")["slice_id"].tolist()
        for i, sid in enumerate(ids):
            prev_ids.append((sid, ids[i - 1] if i > 0 else sid))
            next_ids.append((sid, ids[i + 1] if i < len(ids) - 1 else sid))
    df["prev_slice_id"] = df["slice_id"].map(dict(prev_ids))
    df["next_slice_id"] = df["slice_id"].map(dict(next_ids))
    return df


def _recon_adj(img, acc, rs, nv):
    if nv is not None:
        return undersample_nviews(img, nv, rs)
    elif acc:
        return angular_undersample(img, acc, rs)
    return full_reconstruct(img, rs)


# ── 워커(프로세스) 전처리: 원본과 동일한 결과 ─────────────────────────────────────
def _preprocess(task):
    """task=(sid, split, fpc, fpp, fpn, mode, nv, acc, rs, windows)
    반환: (sid, split, uint8 (H,W,3) or None, err or None)"""
    sid, sp, fpc, fpp, fpn, mode, nv, acc, rs, windows = task
    try:
        if mode == "window":
            img = preprocess_dicom(pydicom.dcmread(fpc))
            recon = reconstruct_hu_proxy(img, radon_size=rs, n_views=nv, acceleration=acc)
            pil = window_to_pil(recon, windows)
        else:  # adjacent
            curr = _recon_adj(preprocess_dicom(pydicom.dcmread(fpc)), acc, rs, nv)
            prev = curr if fpp == fpc else _recon_adj(preprocess_dicom(pydicom.dcmread(fpp)), acc, rs, nv)
            nxt  = curr if fpn == fpc else _recon_adj(preprocess_dicom(pydicom.dcmread(fpn)), acc, rs, nv)
            stack = (np.stack([prev, curr, nxt], axis=-1) * 255).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(stack, mode="RGB")
        return (sid, sp, np.asarray(pil, dtype=np.uint8), None)
    except Exception as e:  # noqa: BLE001
        return (sid, sp, None, repr(e)[:150])


def run(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA를 못 잡음 — nvidia-smi/드라이버 확인 후 재실행 (CPU 폴백 방지)")

    encoder     = getattr(opt, "encoder", "siglip")
    mode        = getattr(opt, "input_mode", "window")
    if mode != "window":
        # adjacent 경로(_recon_adj)는 레거시 sparse_ct 함수(영상별 최댓값 정규화)를 써서
        # 최종 파이프라인과 전처리가 다르다. 최종 실험에서는 쓰지 않으므로 막아 둔다.
        raise ValueError(f"input_mode={mode!r}: 최종 파이프라인은 'window'만 지원 "
                         "(adjacent는 레거시 per-image max 경로라 비활성화)")
    model_id    = getattr(opt, "model_id", None)
    output_dir  = Path(opt.output_dir)
    acc         = getattr(opt, "acceleration", None)
    nv          = getattr(opt, "n_views", None)
    rs          = getattr(opt, "radon_size", 256)
    n_preview   = getattr(opt, "n_preview", 5)
    preview_dir = Path(getattr(opt, "preview_dir", output_dir.parent / f"preview_{encoder}"))
    batch_size  = getattr(opt, "batch_size", 64)
    num_workers = getattr(opt, "num_workers", max(1, (os.cpu_count() or 8) - 4))
    windows     = parse_windows(getattr(opt, "windows", None))

    view_tag = (f"nv{nv}" if nv is not None else (f"x{acc}" if acc else "full"))
    tag = f"{encoder}_{mode}_{view_tag}"
    for sp in ("train", "val", "test"):
        (output_dir / sp).mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"CT Feature Extraction [MP]  [{tag}]")
    print(f"  Encoder  : {encoder}   mode: {mode}   workers: {num_workers}   batch: {batch_size}")
    print(f"  Output   : {output_dir}   Device: {device}")
    print("=" * 60 + "\n")

    extractor = build_extractor(encoder, device, model_id)
    print()

    df = pd.read_csv(opt.master_csv)
    if mode == "adjacent":
        print("슬라이스 순서 정렬 중..."); df = build_slice_order(df); print("  완료\n")
    fp_map = dict(zip(df["slice_id"], df["filepath"]))

    done = {f.stem for f in output_dir.glob("*/*.npy")}
    todo = df[~df["slice_id"].isin(done)].reset_index(drop=True)
    print(f"전체: {len(df):,}  완료: {len(done):,}  대상: {len(todo):,}\n")

    tasks = []
    for _, r in todo.iterrows():
        fpc = r["filepath"]
        fpp = fp_map.get(r.get("prev_slice_id"), fpc) if mode == "adjacent" else fpc
        fpn = fp_map.get(r.get("next_slice_id"), fpc) if mode == "adjacent" else fpc
        tasks.append((r["slice_id"], r["split"], fpc, fpp, fpn, mode, nv, acc, rs, windows))

    failed, preview_count = [], 0
    batch_meta, batch_imgs = [], []
    pbar = tqdm(total=len(tasks), desc=f"[{tag}]", ncols=80)

    def flush():
        if not batch_imgs:
            return
        try:
            embeds = extractor.extract_batch(batch_imgs)
            for (sid, sp), emb in zip(batch_meta, embeds):
                tmp = output_dir / sp / f".{sid}.tmp.npy"
                np.save(str(tmp), emb)
                os.replace(tmp, output_dir / sp / f"{sid}.npy")   # atomic
        except Exception as e:  # noqa: BLE001
            for sid, _ in batch_meta:
                failed.append((sid, str(e)))
            torch.cuda.empty_cache(); gc.collect()
        pbar.update(len(batch_meta))
        batch_meta.clear(); batch_imgs.clear()

    maxq = max(num_workers * 3, 32)      # 진행중 작업 상한 (OOM 방지)
    it = iter(tasks)
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        inflight = set()
        for _ in range(maxq):
            t = next(it, None)
            if t is None:
                break
            inflight.add(pool.submit(_preprocess, t))
        while inflight:
            done_set, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            inflight = set(inflight)
            for fut in done_set:
                nt = next(it, None)
                if nt is not None:
                    inflight.add(pool.submit(_preprocess, nt))
                sid, sp, arr, err = fut.result()
                if arr is None:
                    failed.append((sid, err)); pbar.update(1); continue
                img = Image.fromarray(arr, "RGB")
                if preview_count < n_preview:
                    img.save(preview_dir / f"{sid}.png"); preview_count += 1
                batch_meta.append((sid, sp)); batch_imgs.append(img)
                if len(batch_imgs) >= batch_size:
                    flush()
        flush()

    pbar.close()
    done_total = len(list(output_dir.glob("*/*.npy")))
    print(f"\n완료: {done_total:,} / {len(df):,}   실패: {len(failed):,}")
    if failed:
        logp = output_dir.parent / f"failed_{tag}.txt"
        with open(logp, "w") as f:
            for sid, err in failed:
                f.write(f"{sid}\t{err}\n")
        print(f"실패 목록: {logp}")


if __name__ == "__main__":
    opt = get_config(sys.argv[1])
    run(opt)
