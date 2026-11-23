"""
fig_view_artifact_factors.py — 같은 슬라이스·같은 view 수에서 "줄무늬가 얼마나 세게 보이는가"를 바꾸는 요인 분리.

다른 사람이 만든 view-수 그림(45 view에서도 출혈이 보임)과 우리 Fig. 2(45 view가 흑백으로 포화)의 차이가
어디서 오는지 한 요인씩 바꿔 본다. 재구성 절차는 논문 파이프라인과 같다 (평행 빔 Radon → ramp FBP → 음수 clip).

  A  우리 설정        256×256, HU clip [−1024, 2048], ramp, 표시 W80 (L40)
  B  표시 창만 넓힘    A와 같은 재구성, 표시 W120 (L40)      ← 상대 그림의 [0.49, 0.55]가 [−1024,1024] 정규화라면 약 L40/W120
  C  뼈 clip          HU clip [−1024, 1024] 후 투영 (두개골 대비 감소, 물리적으로는 부정확)
  D  해상도           512×512 원본 (resize 없음)
  E  필터             Hann 창 FBP (고주파 억제)
  F  B+C+D 조합       상대 그림의 추정 설정
  G  윈도우 먼저(W80)  HU → 뇌 윈도우 [0,1] → Radon/FBP → clip[0,1]  (예전 lab 규칙 'window BEFORE reconstruction')
                      두개골·피부(공기) 경계 대비가 투영 전에 사라져 줄무늬가 크게 약해진다. 물리적으로는 부정확.
  H  윈도우 먼저(W120)
  I  SART 2회 반복     FBP 대신 반복 재구성 (우리 감쇠, W80) — 상대가 FBP가 아닌 방법을 썼을 가능성 확인
각 패널에 표시: 뇌 실질 화소(기준 영상 HU −20~120) 중 표시 창 밖으로 포화된 비율(sat), 해당 설정 자체 기준 대비 SSIM.

슬라이스 자동 선택 (--auto_pick K):
  test split에서 --pick_label 양성 슬라이스 --pick_pool 장을 무작위로 골라, 출혈로 보이는 덩어리
  (HU 55–90, 뼈에서 8 px 이상 떨어진 연결 영역)의 최대 면적으로 점수를 매긴다. 검사(study)당 1장.
  → pick_candidates.csv (전체 점수), pick_contact_sheet.png (상위 12장, 뇌 윈도우),
    상위 K장 각각 artifact_factors_{slice_id}_nv{V}.png
  점수는 분할 마스크가 아닌 HU 기준 휴리스틱이므로 contact sheet를 눈으로 보고 고를 것.

실행 위치: 서버 (CPU). 슬라이스 하나 1분 이내, 자동 선택은 pool 400장 기준 수 분. 추출과 동시에 돌려도 됨.
Usage (서버):
  cd ~/hayeon/ich_cf_clean
  # 슬라이스 지정 (쉼표로 여러 개 가능)
  python analysis/fig_view_artifact_factors.py --master_csv /data0/hayeon/master_df.csv --slice_id ID_de5494f39 \
      --views 45,90 --out_dir plots/paper_figs/artifact_factors
  # 출혈이 큰 슬라이스 자동 선택 (상위 3장)
  python analysis/fig_view_artifact_factors.py --master_csv /data0/hayeon/master_df.csv --auto_pick 3 \
      --views 45,90 --workers 16 --out_dir plots/paper_figs/artifact_factors_auto
  # 고른 슬라이스로 Fig. 2 다시 그리기:
  #   python analysis/fig_views_psnr.py --master_csv /data0/hayeon/master_df.csv --example_id <slice_id> ...
결과 가져오기 (로컬 PowerShell; 자동 선택은 폴더 이름만 artifact_factors_auto):
  scp -r neumous-lab@100.73.97.101:/home/neumous-lab/hayeon/ich_cf_clean/plots/paper_figs/artifact_factors "C:\\Users\\user\\Desktop\\MIGT\\01. sparse ct project\\논문작성\\figures\\paper_figs"
"""
import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from skimage.metrics import structural_similarity as ssim
from skimage.transform import iradon, iradon_sart, radon
from skimage.transform import resize as sk_resize

sys.path.append(str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore", message="Radon transform: image must be zero outside")
try:
    pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE
except Exception:
    pass

from extraction.sparse_ct import ANGLES_FULL, sparse_view_indices

AIR = -1024.0
CONFIGS = [  # key, title, size, hu_max, filter, display width
    ("A", "A  ours (256, clip 2048, ramp, W80)", 256, 2048.0, "ramp", 80.0),
    ("B", "B  display W120", 256, 2048.0, "ramp", 120.0),
    ("C", "C  bone clipped at 1024 HU", 256, 1024.0, "ramp", 80.0),
    ("D", "D  512 native", 512, 2048.0, "ramp", 80.0),
    ("E", "E  Hann filter", 256, 2048.0, "hann", 80.0),
    ("F", "F  512 + clip 1024 + W120", 512, 1024.0, "ramp", 120.0),
]


def load_hu(fp):
    d = pydicom.dcmread(fp)
    return d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1)) + float(getattr(d, "RescaleIntercept", 0))


def prep(hu, size, hu_max):
    """HU clip → 감쇠(공기=0) → 크기 조정. 논문 파이프라인의 proxy와 선형 관계."""
    att = np.clip(hu, AIR, hu_max) - AIR
    if att.shape[0] != size:
        att = sk_resize(att, (size, size), anti_aliasing=True, preserve_range=True)
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2
    fov = (yy - c) ** 2 + (xx - c) ** 2 <= (size / 2) ** 2
    return np.where(fov, att, 0.0).astype(np.float32), fov


def fbp_hu(att, nv, filt):
    th = ANGLES_FULL[sparse_view_indices(nv)]
    rec = iradon(radon(att, theta=th, circle=True), theta=th, circle=True, filter_name=filt)
    return np.clip(rec, 0.0, None) + AIR                     # 음수(공기 미만)만 clip → HU


def window_before(hu, size, level, width, nv):
    """윈도우를 먼저 적용([0,1])한 영상을 투영·FBP → clip[0,1]. 반환: (기준 [0,1], 재구성 clip 전)."""
    w = np.clip((hu - (level - width / 2)) / width, 0.0, 1.0)
    w = sk_resize(w, (size, size), anti_aliasing=True, preserve_range=True).astype(np.float32)
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2
    w = np.where((yy - c) ** 2 + (xx - c) ** 2 <= (size / 2) ** 2, w, 0.0)
    th = ANGLES_FULL[sparse_view_indices(nv)]
    return w, iradon(radon(w, theta=th, circle=True), theta=th, circle=True, filter_name="ramp")


def sart_hu(att, nv, n_iter=2):
    th = ANGLES_FULL[sparse_view_indices(nv)]
    sino = radon(att, theta=th, circle=True).astype(np.float64)
    rec = None
    for _ in range(n_iter):
        rec = iradon_sart(sino, theta=th, image=rec)
    return np.clip(rec, 0.0, None) + AIR


def win(hu, level, width):
    return np.clip((hu - (level - width / 2)) / width, 0.0, 1.0)


# ── 출혈이 큰 슬라이스 자동 선택 ─────────────────────────────────────────────
def hemorrhage_score(fp):
    """출혈로 보이는 연결 영역의 최대 면적 (512 격자 기준 화소 수). 실패 시 -1."""
    from scipy import ndimage as ndi
    try:
        hu = load_hu(fp)
    except Exception:  # noqa: BLE001
        return -1.0
    sm = ndi.uniform_filter(hu, 5)
    near_bone = ndi.binary_dilation(hu > 300, iterations=8)
    cand = ndi.binary_opening((sm > 55) & (sm < 90) & ~near_bone, iterations=2)
    lab, n = ndi.label(cand)
    if n == 0:
        return 0.0
    return float(np.max(ndi.sum(cand, lab, range(1, n + 1)))) * (512 * 512) / hu.size


def pick_slices(df, a, out, plt):
    pool = df[(df["split"] == "test") & (df[a.pick_label] == 1)]
    pool = pool.sample(n=min(a.pick_pool, len(pool)), random_state=a.seed)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        scores = list(ex.map(hemorrhage_score, pool["filepath"], chunksize=4))
    pool = pool.assign(score=scores).sort_values("score", ascending=False)
    if "study_id" in pool.columns:
        pool = pool.drop_duplicates("study_id")
    cols = [c for c in ("slice_id", "study_id", "score", "filepath") if c in pool.columns]
    pool[cols].to_csv(out / "pick_candidates.csv", index=False)

    top = pool.head(12)
    fig, axes = plt.subplots(2, 6, figsize=(16, 6))
    for ax, (_, r) in zip(axes.ravel(), top.iterrows()):
        g, _ = prep(load_hu(r["filepath"]), 256, 2048.0)
        ax.imshow(win(g + AIR, a.level, 80), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{r['slice_id']}\nscore {r['score']:.0f} px", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"top candidates ({a.pick_label}+, test, {len(scores)} sampled) — brain window W80", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "pick_contact_sheet.png", dpi=130, facecolor="white")
    plt.close(fig)
    print(f"saved {out}/pick_contact_sheet.png, pick_candidates.csv")
    print(top[["slice_id", "score"]].to_string(index=False))
    return list(top["slice_id"].head(a.auto_pick))


# ── 요인별 비교 그림 ─────────────────────────────────────────────────────────
def render(hu, sid, nv, a, out, plt, rows):
    fig, axes = plt.subplots(3, 4, figsize=(13, 10.2))
    axes = axes.ravel()
    gt256, _ = prep(hu, 256, 2048.0)
    axes[0].imshow(win(gt256 + AIR, a.level, 80), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("input 256 (no FBP), W80", fontsize=9)
    gt512, _ = prep(hu, 512, 2048.0)
    axes[4].imshow(win(gt512 + AIR, a.level, 80), cmap="gray", vmin=0, vmax=1)
    axes[4].set_title("input 512 (no FBP), W80", fontsize=9)

    def add(key, k, img, title, sat, s, sd, **meta):
        rows.append(dict(slice_id=sid, views=nv, config=key, sat_brain=sat, ssim_window=s,
                         err_sd_brain_HU=sd, **meta))
        axes[k].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[k].set_title(f"{title}\nsat {100 * sat:.0f}%   SSIM {s:.3f}   err SD {sd:.0f} HU", fontsize=8.5)

    for (key, title, size, hmax, filt, width), k in zip(CONFIGS, [1, 2, 3, 5, 6, 7]):
        att, fov = prep(hu, size, hmax)
        gt = att + AIR
        rec = fbp_hu(att, nv, filt)
        brain = fov & (gt > -20) & (gt < 120)
        lo, hi = a.level - width / 2, a.level + width / 2
        add(key, k, win(rec, a.level, width), title,
            float(((rec[brain] < lo) | (rec[brain] > hi)).mean()),
            float(ssim(win(gt, a.level, width), win(rec, a.level, width), data_range=1.0)),
            float((rec - gt)[brain].std()), size=size, hu_max=hmax, filter=filt, width=width)

    # ── 3행: 윈도우 먼저(G, H), SART(I) ──
    gt_hu = gt256 + AIR
    _, fov = prep(hu, 256, 2048.0)
    brain = fov & (gt_hu > -20) & (gt_hu < 120)
    for key, title, width, k in (("G", "G  window BEFORE FBP (W80)", 80.0, 9),
                                 ("H", "H  window BEFORE FBP (W120)", 120.0, 10)):
        gt_w, rec_w = window_before(hu, 256, a.level, width, nv)
        rec_c = np.clip(rec_w, 0.0, 1.0)
        add(key, k, rec_c, title + " *",
            float(((rec_w[brain] < 0) | (rec_w[brain] > 1)).mean()),
            float(ssim(gt_w, rec_c, data_range=1.0)),
            float(((rec_w - gt_w) * width)[brain].std()), size=256, hu_max=np.nan, filter="ramp", width=width)
    att, _ = prep(hu, 256, 2048.0)
    rec = sart_hu(att, nv)
    lo, hi = a.level - 40, a.level + 40
    add("I", 11, win(rec, a.level, 80), "I  SART ×2 (window after, W80)",
        float(((rec[brain] < lo) | (rec[brain] > hi)).mean()),
        float(ssim(win(gt_hu, a.level, 80), win(rec, a.level, 80), data_range=1.0)),
        float((rec - gt_hu)[brain].std()), size=256, hu_max=2048.0, filter="sart2", width=80.0)

    axes[8].axis("off")
    axes[8].text(0.0, 0.5, "sat = brain-tissue pixels outside\nthe display window\n\n"
                 "* G, H: error measured in the\nwindowed image (× width → HU);\n"
                 "values outside the window are\nlost before projection",
                 fontsize=8.5, va="center", ha="left", transform=axes[8].transAxes)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{sid}  —  {nv} views: which factor makes the brain window saturate?", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"artifact_factors_{sid}_nv{nv}.png", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"saved {out}/artifact_factors_{sid}_nv{nv}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_csv", required=True)
    ap.add_argument("--slice_id", default="ID_de5494f39", help="쉼표로 여러 개 가능. --auto_pick 이면 무시")
    ap.add_argument("--auto_pick", type=int, default=0, help="출혈 면적 상위 K장을 자동 선택 (0 = 끔)")
    ap.add_argument("--pick_label", default="intraparenchymal")
    ap.add_argument("--pick_pool", type=int, default=400, help="점수를 매길 양성 슬라이스 수 (무작위)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--views", default="45")
    ap.add_argument("--level", type=float, default=40.0)
    ap.add_argument("--out_dir", default="plots/paper_figs/artifact_factors")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(a.master_csv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    sids = pick_slices(df, a, out, plt) if a.auto_pick > 0 else [x.strip() for x in a.slice_id.split(",")]

    rows = []
    fp = dict(zip(df["slice_id"], df["filepath"]))
    for sid in sids:
        hu = load_hu(fp[sid])
        for nv in [int(v) for v in a.views.split(",")]:
            render(hu, sid, nv, a, out, plt, rows)

    tab = pd.DataFrame(rows)
    tab.to_csv(out / "artifact_factors.csv", index=False)
    print(tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
