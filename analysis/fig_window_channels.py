"""
fig_window_channels.py — 인코더 입력 RGB를 윈도우 채널별로 나눠 보기 (슬라이스 하나, view 수별).

추출기와 같은 함수를 그대로 쓴다:
  DICOM → preprocess_dicom → reconstruct_hu_proxy(n_views) → make_window_stack(DEFAULT_WINDOWS)
  R = brain    (L40 / W80)
  G = subdural (L80 / W200)
  B = bone     (L600 / W2800)
행 = view 수, 열 = R | G | B (각각 회색조) | RGB 합성 (인코더가 받는 영상).

저장 (--out_dir):
  window_channels_{slice_id}.png / .pdf   격자 그림
  window_channels_{slice_id}.npz          채널 원자료 (views, stack[V, H, W, 3] ∈ [0,1])

실행 위치: 서버 (CPU, 1분 이내). 추출과 동시에 돌려도 됨.
Usage (서버):
  cd ~/hayeon/ich_cf_clean
  python analysis/fig_window_channels.py --master_csv /data0/hayeon/master_df.csv --slice_id ID_d69a63518 \
      --views 360,180,90,45,23,12,6 --out_dir plots/paper_figs/window_channels
  # 몇 개만:  --views 360,45,6
결과 가져오기 (로컬 PowerShell):
  scp -r neumous-lab@100.73.97.101:/home/neumous-lab/hayeon/ich_cf_clean/plots/paper_figs/window_channels "C:\\Users\\user\\Desktop\\MIGT\\01. sparse ct project\\논문작성\\figures\\paper_figs"
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

from extraction.sparse_ct import preprocess_dicom
from extraction_3ch.windowing import DEFAULT_WINDOWS, make_window_stack, reconstruct_hu_proxy

NAMES = ["brain", "subdural", "bone"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_csv", required=True)
    ap.add_argument("--slice_id", default="ID_d69a63518")
    ap.add_argument("--views", default="360,180,90,45,23,12,6", help="행 순서대로")
    ap.add_argument("--radon_size", type=int, default=256)
    ap.add_argument("--out_dir", default="plots/paper_figs/window_channels")
    a = ap.parse_args()

    df = pd.read_csv(a.master_csv)
    row = df[df["slice_id"] == a.slice_id].iloc[0]
    views = [int(v) for v in a.views.split(",")]
    img01 = preprocess_dicom(pydicom.dcmread(row["filepath"]))

    stacks = []
    for nv in views:
        rec = reconstruct_hu_proxy(img01, radon_size=a.radon_size, n_views=nv)
        stacks.append(make_window_stack(rec, DEFAULT_WINDOWS))          # (H, W, 3) ∈ [0,1]
        print(f"  {nv} views done")
    stacks = np.stack(stacks)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"window_channels_{a.slice_id}.npz", views=np.array(views), stack=stacks,
                        windows=np.array(DEFAULT_WINDOWS))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heads = [f"{c}: {n} (L{lv:g}/W{wd:g})" for c, n, (lv, wd) in zip("RGB", NAMES, DEFAULT_WINDOWS)] + ["RGB (encoder input)"]
    fig, axes = plt.subplots(len(views), 4, figsize=(4 * 2.3, len(views) * 2.3), squeeze=False)
    for i, nv in enumerate(views):
        for j in range(4):
            ax = axes[i, j]
            img = stacks[i] if j == 3 else stacks[i, :, :, j]
            ax.imshow(img, cmap=None if j == 3 else "gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(heads[j], fontsize=8.5)
        axes[i, 0].set_ylabel(f"{nv} views" + (" (ref)" if nv == 360 else ""), fontsize=9)
    fig.suptitle(a.slice_id, fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"window_channels_{a.slice_id}.{ext}", dpi=200, facecolor="white")
    print(f"saved → {out}/window_channels_{a.slice_id}.(png|pdf|npz)")


if __name__ == "__main__":
    main()
