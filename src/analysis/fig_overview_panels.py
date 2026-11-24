"""
fig_overview_panels.py — Overview 그림 (a) 패널용 중간 영상 추출 (슬라이스 하나).

최종 파이프라인 함수를 그대로 사용한다 (원고 §3.4–3.5와 같은 영상):
  DICOM → preprocess_dicom ([−1024,2048] → [0,1]) → 256×256 → Radon
  → 기준(reference) 360 view ([0°,180°) 0.5° 간격) / sparse n_v view (1° 격자, sparse_view_indices)
  → reconstruct_hu_proxy (FBP) → window_to_pil (brain / subdural / bone → RGB)
  sparse 각도는 모두 360-view 격자에 포함되므로, sparse 사이노그램 = 기준 사이노그램에서 n_v 열만 남긴 것.

저장 (--out_dir):
  overview_panels_{slice_id}_nv{n_v}.npz   로컬 조립용 원자료
     input_brain     256×256  입력 영상 (재구성 전) 뇌 윈도우 [0,1]
     sino_ref        256×360  기준(360 view) 사이노그램
     sino_sparse     256×360  기준 사이노그램에서 n_v 열만 남긴 것 (나머지 0)
     view_deg        (n_v,)   남긴 각도(°);  ref_views = 360
     rec_ref_brain   256×256  360-view FBP, 뇌 윈도우
     rec_sparse_brain 256×256 n_v-view FBP, 뇌 윈도우
     rgb_ref / rgb_sparse  256×256×3 uint8  인코더 입력 RGB
  overview_preview_{slice_id}_nv{n_v}.png  확인용 미리보기 (최종 그림 아님)

실행 위치: 서버 (CPU만 사용, 수 초). 추출 작업과 동시에 돌려도 됨.
Usage (서버):
  cd ~/hayeon/ich_cf_clean
  python analysis/fig_overview_panels.py --master_csv /data0/hayeon/master_df.csv --views 45 \
      --out_dir /home/neumous-lab/hayeon/ich_cf_clean/plots/paper_figs/overview
  # Fig. 2 와 같은 슬라이스를 쓰려면:  --slice_id ID_xxxxxxxxx
  # 여러 view 수 비교:  for v in 23 45 90; do python analysis/fig_overview_panels.py --master_csv /data0/hayeon/master_df.csv --slice_id ID_d69a63518 --views $v --out_dir plots/paper_figs/overview; done
결과 가져오기 (로컬 PowerShell):
  scp -r neumous-lab@100.73.97.101:/home/neumous-lab/hayeon/ich_cf_clean/plots/paper_figs/overview "C:\\Users\\user\\Desktop\\MIGT\\01. sparse ct project\\논문작성\\figures\\paper_figs"
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

sys.path.append(str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore", message="Radon transform: image must be zero outside")
try:
    pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE
except Exception:
    pass

from skimage.transform import radon
from skimage.transform import resize as sk_resize

from extraction.sparse_ct import ANGLES_FULL, preprocess_dicom, sparse_view_indices
from extraction_3ch.windowing import (DEFAULT_WINDOWS, apply_window, proxy_to_hu,
                                      reconstruct_hu_proxy, window_to_pil)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_csv", required=True)
    ap.add_argument("--slice_id", default=None, help="미지정 시 --example_label 양성 test 슬라이스 중 무작위")
    ap.add_argument("--example_label", default="intraparenchymal")
    ap.add_argument("--split", default="test")
    ap.add_argument("--views", type=int, default=45)
    ap.add_argument("--radon_size", type=int, default=256)
    ap.add_argument("--ref_views", type=int, default=360, help="기준 view 수 (학습 config의 feature_ref360과 동일)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="plots/paper_figs/overview")
    a = ap.parse_args()

    df = pd.read_csv(a.master_csv)
    if a.slice_id:
        row = df[df["slice_id"] == a.slice_id].iloc[0]
    else:
        pool = df[(df["split"] == a.split) & (df[a.example_label] == 1)]
        row = pool.iloc[int(np.random.default_rng(a.seed).integers(len(pool)))]
    sid, rs, nv, nr = row["slice_id"], a.radon_size, a.views, a.ref_views
    print(f"slice: {sid}  views: {nv}  reference: {nr}")

    img01 = preprocess_dicom(pydicom.dcmread(row["filepath"]))            # 512×512 proxy [0,1]
    small = sk_resize(img01, (rs, rs), anti_aliasing=True).astype(np.float32)
    th_ref = np.linspace(0.0, 180.0, nr, endpoint=False)
    sino_ref = radon(small, theta=th_ref, circle=True).astype(np.float32)
    deg = ANGLES_FULL[sparse_view_indices(nv)]                          # sparse 각도 (1° 격자)
    col = np.round(deg * nr / 180.0).astype(int)                        # 기준 격자에서의 열 번호
    assert np.allclose(th_ref[col], deg), "sparse 각도가 기준 격자에 없음"
    sino_sparse = np.zeros_like(sino_ref); sino_sparse[:, col] = sino_ref[:, col]

    rec_full = reconstruct_hu_proxy(img01, radon_size=rs, n_views=nr)  # 기준 (n_views>180 경로)
    rec_sparse = reconstruct_hu_proxy(img01, radon_size=rs, n_views=nv)
    brain = DEFAULT_WINDOWS[0]
    bw = lambda proxy: apply_window(proxy_to_hu(proxy), *brain)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"overview_panels_{sid}_nv{nv}.npz",
        slice_id=sid, views=nv, ref_views=nr, view_deg=deg,
        input_brain=bw(small), sino_ref=sino_ref, sino_sparse=sino_sparse,
        rec_ref_brain=bw(rec_full), rec_sparse_brain=bw(rec_sparse),
        rgb_ref=np.asarray(window_to_pil(rec_full, DEFAULT_WINDOWS)),
        rgb_sparse=np.asarray(window_to_pil(rec_sparse, DEFAULT_WINDOWS)),
        windows=np.array(DEFAULT_WINDOWS),
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vmax = np.percentile(sino_ref, 99.5)
    panels = [("input (brain window)", bw(small), "gray", 1), (f"sinogram, {nr} views (ref)", sino_ref, "gray", vmax),
              (f"sinogram, {nv} views", sino_sparse, "gray", vmax), (f"FBP {nr} (brain)", bw(rec_full), "gray", 1),
              (f"FBP {nv} (brain)", bw(rec_sparse), "gray", 1), (f"RGB {nr}", np.asarray(window_to_pil(rec_full)), None, None),
              (f"RGB {nv}", np.asarray(window_to_pil(rec_sparse)), None, None)]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.2 * len(panels), 2.6))
    for ax, (t, im, cmap, vm) in zip(axes, panels):
        ax.imshow(im, cmap=cmap, vmin=0 if vm else None, vmax=vm, aspect="auto" if "sinogram" in t else "equal",
                  interpolation="nearest")
        ax.set_title(t, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{sid} — preview only (final figure is assembled locally)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"overview_preview_{sid}_nv{nv}.png", dpi=150, facecolor="white")
    print(f"saved → {out}/overview_panels_{sid}_nv{nv}.npz, overview_preview_{sid}_nv{nv}.png   (views kept, deg: {deg.astype(int).tolist()})")


if __name__ == "__main__":
    main()
