"""
fig_views_psnr.py — FBP reconstruction vs. number of views (Fig. 2 candidate)
                     + PSNR/SSIM vs. views (supplementary evidence for the 360-view reference).

Pipeline is identical to the feature-extraction path used in the paper:
  DICOM -> preprocess_dicom (fixed HU clip [-1024, 2048] -> [0, 1])
        -> resize to radon_size (anti-aliased)
        -> parallel-beam Radon -> ramp-filtered FBP -> clip negatives (no per-image max)
  n_v <= 180 : evenly spaced indices on the 1-degree grid (sparse conditions, same as the paper)
  n_v  > 180 : [0, 180) split evenly (360 = reference, 0.5-degree step; 720 = denser, for comparison only)

Ground truth = the resized input image before the Radon transform.
Metrics are computed inside the inscribed FOV circle, on the brain-windowed image by default
(--metric_domain full uses the whole [0, 1] proxy range instead).
Slices with almost no tissue inside the display window (fraction of FOV pixels strictly inside the window
< --min_tissue, e.g. the vertex) are excluded from the mean: there both images are flat after windowing and
PSNR is infinite.

Grid (Fig. 2): panels in --grid_views order (default 360 → 6), labelled (a), (b), … below each panel, no titles.
The first panel (reference) sits alone in the left column; the rest fill a 2 × 3 block (layout of the paper figure).
--show_input adds the input image as an extra first panel; --titles restores per-panel view-count titles.

Outputs (in --out_dir):
  fig2_views_grid.png / .pdf        image grid: reference + each view count, with mean & example metrics
  figS_psnr_ssim_vs_views.png / .pdf  mean ± SD of PSNR and SSIM across sampled slices
  psnr_ssim_vs_views.csv            per-view summary table

Usage (server):
  cd ~/hayeon/ich_cf_clean
  python analysis/fig_views_psnr.py --master_csv /data0/hayeon/master_df.csv \
      --split test --n 2000 --views 6,12,23,45,90,180,360,720 --example_id ID_d69a63518 \
      --workers 16 --out_dir plots/paper_figs
  결과 가져오기 (로컬 PowerShell):
  scp neumous-lab@100.73.97.101:/home/neumous-lab/hayeon/ich_cf_clean/plots/paper_figs/fig2_views_grid.* "C:\\Users\\user\\Desktop\\MIGT\\01. sparse ct project\\논문작성\\figures\\paper_figs" 
  # 특정 슬라이스를 예시로 쓰고 출혈 부위를 확대하려면:
  #   --example_id ID_xxxxxxxxx --zoom 120,60,64      (x, y, size on the 256 grid)
"""

import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim
from skimage.transform import iradon, radon
from skimage.transform import resize as sk_resize

sys.path.append(str(Path(__file__).resolve().parent.parent))
try:
    pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE
except Exception:
    pass

from extraction.sparse_ct import ANGLES_FULL, N_ANGLES_FULL, preprocess_dicom, sparse_view_indices

PAPER_VIEWS = {6, 12, 23, 45, 90, 180, 360}


# ── reconstruction (same procedure as the extraction runners) ────────────────
def hu_to_proxy(hu):
    return (hu + 1024.0) / 3072.0


def thetas_for(nv):
    if nv <= N_ANGLES_FULL:                          # 추출 파이프라인과 같은 view 선택 규칙
        return ANGLES_FULL[sparse_view_indices(nv)]
    return np.linspace(0.0, 180.0, nv, endpoint=False)


def fbp(small, nv):
    th = thetas_for(nv)
    sino = radon(small, theta=th, circle=True)
    rec = iradon(sino, theta=th, circle=True, filter_name="ramp")
    return np.clip(rec, 0.0, None).astype(np.float32)


_MASK = {}


def circle_mask(n):
    if n not in _MASK:
        yy, xx = np.mgrid[:n, :n]
        c = (n - 1) / 2.0
        _MASK[n] = ((yy - c) ** 2 + (xx - c) ** 2) <= (n / 2.0) ** 2
    return _MASK[n]


def window(x, level, width):
    lo, hi = hu_to_proxy(level - width / 2.0), hu_to_proxy(level + width / 2.0)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def to_metric_space(x, domain, level, width):
    return window(x, level, width) if domain == "window" else np.clip(x, 0.0, 1.0)


def load_small(fp, rs):
    img = preprocess_dicom(pydicom.dcmread(fp))
    return sk_resize(img, (rs, rs), anti_aliasing=True).astype(np.float32)


def metrics(gt, rec):
    return (float(_psnr(gt, rec, data_range=1.0)), float(_ssim(gt, rec, data_range=1.0)))


# ── worker ───────────────────────────────────────────────────────────────────
def _eval_one(task):
    fp, views, rs, domain, level, width, min_tissue = task
    try:
        small = load_small(fp, rs)
    except Exception:  # noqa: BLE001  (e.g., the known corrupted ID_6431af929)
        return None
    m = circle_mask(rs)
    w = window(small, level, width)
    if ((w > 0) & (w < 1) & m).sum() / m.sum() < min_tissue:
        return "empty"                       # 윈도우 안 조직이 거의 없음 → 평균에서 제외
    gt = to_metric_space(np.where(m, small, 0.0), domain, level, width)
    out = []
    for nv in views:
        rec = to_metric_space(np.where(m, fbp(small, nv), 0.0), domain, level, width)
        out.append(metrics(gt, rec))
    return out


# ── plotting ─────────────────────────────────────────────────────────────────
def plot_grid(panels, n_used, args, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    total = len(panels)
    ncols = args.ncols or 4
    if args.titles:                                         # 예전 배치: 순서대로 채움
        nrows = math.ceil(total / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.4, nrows * 2.55))
        axes = np.atleast_1d(axes).ravel()
        used = list(axes[:total])
    else:                                                   # 논문 배치: 첫 패널은 왼쪽 열 단독
        nrows = math.ceil((total - 1) / (ncols - 1))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.4, nrows * 2.6))
        axes = np.atleast_2d(axes)
        used = [axes[0, 0]] + [axes[r, c] for r in range(nrows) for c in range(1, ncols)][:total - 1]
    zoom = [int(v) for v in args.zoom.split(",")] if args.zoom else None

    for k, (ax, (title, img, stats)) in enumerate(zip(used, panels)):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if args.titles:
            ax.set_title(title, fontsize=8, pad=3)
        else:
            ax.text(0.5, -0.04, f"({chr(ord('a') + k)})", transform=ax.transAxes, ha="center", va="top",
                    fontsize=10, fontweight="bold")
        if stats is not None:
            (mp, ms), (ep, es) = stats
            kw = dict(transform=ax.transAxes, va="top", family="monospace",
                      fontsize=5.5, color="white")
            ax.text(0.03, 0.97, f"mean (n={n_used})\nPSNR {mp:.2f}\nSSIM {ms:.4f}", ha="left", **kw)
            ax.text(0.97, 0.97, f"this slice\nPSNR {ep:.2f}\nSSIM {es:.4f}", ha="right", **kw)
        if zoom:
            x, y, s = zoom
            ax.add_patch(Rectangle((x, y), s, s, fill=False, ec="#eda100", lw=0.8))
            ins = ax.inset_axes([0.62, 0.02, 0.36, 0.36])
            ins.imshow(img[y:y + s, x:x + s], cmap="gray", vmin=0, vmax=1,
                       interpolation="nearest")
            ins.set_xticks([]); ins.set_yticks([])
            for sp in ins.spines.values():
                sp.set_edgecolor("#eda100"); sp.set_linewidth(0.8)
    for ax in np.ravel(axes):
        if not any(ax is u for u in used):
            ax.axis("off")

    if args.titles:
        dom = (f"brain window (L{args.level:g}/W{args.width:g})" if args.metric_domain == "window"
               else "full [0, 1] range")
        fig_h = nrows * 2.55
        fig.suptitle(f"FBP reconstruction vs. number of views  (display: L{args.level:g}/W{args.width:g}; "
                     f"metrics on {dom}, inside FOV)", fontsize=8.5, y=1 - 0.08 / fig_h, va="top")
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.32 / fig_h), h_pad=1.4, w_pad=0.3)
    else:
        fig.tight_layout(h_pad=2.2, w_pad=0.6)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig2_views_grid.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


def plot_curves(summary, args, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, gray, ink2 = "#2a78d6", "#898781", "#52514e"
    v = summary["views"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, key, lab in zip(axes, ("psnr", "ssim"), ("PSNR (dB)", "SSIM")):
        mu, sd = summary[f"{key}_mean"].to_numpy(), summary[f"{key}_std"].to_numpy()
        ax.fill_between(v, mu - sd, mu + sd, color=blue, alpha=0.12, lw=0)
        ax.plot(v, mu, color=blue, lw=1.6, zorder=3)
        paper = np.isin(v, list(PAPER_VIEWS))
        ax.scatter(v[paper], mu[paper], s=26, color=blue, edgecolor="white", lw=0.8,
                   zorder=4, label="views used in the study")
        ax.scatter(v[~paper], mu[~paper], s=26, facecolor="white", edgecolor=blue, lw=1.2,
                   zorder=4, label="denser than reference")
        ax.axvline(args.reference, color=gray, lw=1.0, ls=(0, (4, 3)))
        ax.text(args.reference * 1.06, ax.get_ylim()[1], f"reference ({args.reference})", fontsize=6.5, color=ink2,
                va="top", ha="left")
        ax.set_xscale("log"); ax.set_xticks(v); ax.set_xticklabels(v, fontsize=7)
        ax.minorticks_off()
        ax.set_xlabel("Projection views", fontsize=8, color=ink2)
        ax.set_ylabel(lab, fontsize=8, color=ink2)
        ax.tick_params(labelsize=7, colors=ink2)
        ax.grid(axis="y", color="#e1e0d9", lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"figS_psnr_ssim_vs_views.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_csv", required=True)
    ap.add_argument("--split", default="test", help="train | val | test | all")
    ap.add_argument("--n", type=int, default=2000, help="slices averaged for the mean metrics")
    ap.add_argument("--views", default="6,12,23,45,90,180,360,720")
    ap.add_argument("--reference", type=int, default=360, help="marked as (reference) in titles")
    ap.add_argument("--radon_size", type=int, default=256)
    ap.add_argument("--level", type=float, default=40.0, help="display/metric window level (HU)")
    ap.add_argument("--width", type=float, default=80.0, help="display/metric window width (HU)")
    ap.add_argument("--metric_domain", choices=["window", "full"], default="window")
    ap.add_argument("--example_id", default=None, help="slice_id shown in the grid")
    ap.add_argument("--example_label", default="intraparenchymal",
                    help="if no example_id: pick a random slice positive for this label")
    ap.add_argument("--zoom", default=None, help="x,y,size on the radon grid, e.g. 120,60,64")
    ap.add_argument("--ncols", type=int, default=0)
    ap.add_argument("--grid_views", default="360,180,90,45,23,12,6",
                    help="views shown in the grid, in panel order (a), (b), … (subset of --views)")
    ap.add_argument("--show_input", action="store_true", help="add the input image as the first panel")
    ap.add_argument("--titles", action="store_true", help="old style: view-count titles + suptitle, no (a)(b) labels")
    ap.add_argument("--min_tissue", type=float, default=0.05,
                    help="exclude slices whose FOV has less than this fraction of pixels inside the window")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 4))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="plots/paper_figs")
    args = ap.parse_args()

    views = sorted({int(v) for v in args.views.split(",")})
    rs = args.radon_size
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.master_csv)
    pool_df = df if args.split == "all" else df[df["split"] == args.split]
    sample = pool_df.sample(n=min(args.n, len(pool_df)), random_state=args.seed)
    print(f"views={views}  split={args.split}  sampled={len(sample)}  workers={args.workers}")

    # 1) mean metrics over sampled slices
    tasks = [(fp, views, rs, args.metric_domain, args.level, args.width, args.min_tissue)
             for fp in sample["filepath"]]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        raw = list(ex.map(_eval_one, tasks, chunksize=4))
    res = [r for r in raw if isinstance(r, list)]
    arr = np.asarray(res)                                  # (N, V, 2)
    n_used = arr.shape[0]
    n_empty = sum(1 for r in raw if r == "empty")
    print(f"evaluated {n_used} slices ({n_empty} excluded: < {args.min_tissue:.0%} tissue in window; "
          f"{len(tasks) - n_used - n_empty} failed to decode)")
    if not np.isfinite(arr).all():
        print("  warning: non-finite metrics remain; they are excluded from the means")
        arr = np.where(np.isfinite(arr), arr, np.nan)
    summary = pd.DataFrame({
        "views": views,
        "psnr_mean": np.nanmean(arr[:, :, 0], 0), "psnr_std": np.nanstd(arr[:, :, 0], 0),
        "ssim_mean": np.nanmean(arr[:, :, 1], 0), "ssim_std": np.nanstd(arr[:, :, 1], 0),
        "n": n_used,
    })
    summary.to_csv(out_dir / "psnr_ssim_vs_views.csv", index=False)
    print(summary.round(4).to_string(index=False))

    # 2) example slice
    if args.example_id:
        ex_row = df[df["slice_id"] == args.example_id].iloc[0]
    else:
        pos = pool_df[pool_df[args.example_label] == 1]
        ex_row = pos.iloc[int(rng.integers(len(pos)))]
    print(f"example slice: {ex_row['slice_id']}")
    small = load_small(ex_row["filepath"], rs)
    m = circle_mask(rs)
    gt_raw = np.where(m, small, 0.0)
    gt_m = to_metric_space(gt_raw, args.metric_domain, args.level, args.width)

    # sanity: our FBP == the extraction pipeline's reconstruct_hu_proxy (sparse ≤180 and reference 360)
    try:
        from extraction_3ch.windowing import reconstruct_hu_proxy
        img01 = preprocess_dicom(pydicom.dcmread(ex_row["filepath"]))
        for nv in [v for v in views if v <= N_ANGLES_FULL or v == args.reference]:
            ref = reconstruct_hu_proxy(img01, radon_size=rs, n_views=nv)
            print(f"  check n_v={nv}: max|ours - pipeline| = {np.abs(ref - fbp(small, nv)).max():.2e}")
    except Exception as e:  # noqa: BLE001
        print(f"  (pipeline equivalence check skipped: {e!r})")

    grid_views = [int(v) for v in args.grid_views.split(",")]
    missing = [v for v in grid_views if v not in views]
    assert not missing, f"--grid_views {missing} not in --views"
    panels = [(f"Input image ({rs}×{rs})", window(gt_raw, args.level, args.width), None)] if args.show_input else []
    for nv in grid_views:
        j = views.index(nv)
        rec = np.where(m, fbp(small, nv), 0.0)
        ex_metrics = metrics(gt_m, to_metric_space(rec, args.metric_domain, args.level, args.width))
        avg = (summary.loc[j, "psnr_mean"], summary.loc[j, "ssim_mean"])
        title = f"{nv} views" + ("  (reference)" if nv == args.reference else "")
        panels.append((title, window(rec, args.level, args.width), (avg, ex_metrics)))

    for k, nv in enumerate(grid_views):
        print(f"  ({chr(ord('a') + k + int(args.show_input))}) {nv} views")
    plot_grid(panels, n_used, args, out_dir)
    plot_curves(summary, args, out_dir)
    print(f"saved → {out_dir}/fig2_views_grid.(png|pdf), figS_psnr_ssim_vs_views.(png|pdf), "
          f"psnr_ssim_vs_views.csv")


if __name__ == "__main__":
    main()
