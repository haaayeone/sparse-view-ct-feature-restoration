"""
fig_tsne_cosine.py — 원고 §4.3 / Fig. 5: 임베딩 정렬.
  (a) t-SNE: E_f(기준) · E_u(희소) · E_r(복원)을 함께 2차원으로 (무작위 부분집합)
  (b) 슬라이스별 코사인 유사도 cos(E_f, E_u) vs cos(E_f, E_r) 분포 (test 전체)
  + 개별 대응 확인: 부분집합 안에서 E_r(또는 E_u)의 가장 가까운 E_f가 자기 자신인 비율(top-1)
    → "군집이 겹친다고 개별 임베딩이 대응하는 것은 아니다"라는 §4.3의 단서를 수치로 보완.

학습된 확산 모델(weights/diffusion_{tag}_none_T100.best.pt)로 E_r을 1단계 복원해 계산한다.

저장 (--out_dir):
  tsne_cosine_{tag}.npz   coords(3n×2), group(0=E_f,1=E_u,2=E_r), any_label, cos_u, cos_r (test 전체), top1
  tsne_cosine_{tag}.csv   요약 (mean/median/p5 코사인, top-1 비율)
  tsne_cosine_{tag}.png/.pdf  초안 그림 (최종 조립은 로컬)

실행 위치: 서버 (GPU, train_all.sh 로 해당 config 학습이 끝난 뒤). t-SNE n=1000이면 수 분.
Usage (서버):
  cd ~/hayeon/ich_cf_clean
  python analysis/fig_tsne_cosine.py --train_config configs/medsiglip_after/train_medsiglip_after_nv6.yaml \
      --n_tsne 1000 --out_dir /home/neumous-lab/hayeon/ich_cf_clean/plots/paper_figs/tsne
결과 가져오기 (로컬 PowerShell):
  scp -r neumous-lab@100.73.97.101:/home/neumous-lab/hayeon/ich_cf_clean/plots/paper_figs/tsne "C:\\Users\\user\\Desktop\\MIGT\\01. sparse ct project\\논문작성\\figures\\paper_figs"
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data import CTFeatureMapDataset
from diffusion import ColdDiffusionMLP
from train_diffusion import load_to_ram, set_seed
from utils import get_config

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"


def unit(x):
    return x / (x.norm(dim=1, keepdim=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_tsne", type=int, default=1000, help="t-SNE·top-1에 쓸 슬라이스 수")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="plots/paper_figs/tsne")
    a = ap.parse_args()

    set_seed(a.seed)
    cfg = get_config(a.train_config)
    tag, pooling = cfg.tag, getattr(cfg, "pooling", "none")
    labels = list(getattr(cfg, "label_cols", ["epidural", "intraparenchymal", "intraventricular",
                                               "subarachnoid", "subdural", "any"]))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(cfg.csv_path)
    ds = CTFeatureMapDataset(df, cfg.full_feature_dir, cfg.under_feature_dir, labels, a.split, pooling)
    ef, eu, Y = load_to_ram(ds)
    print(f"{tag}  {a.split}  N={ef.size(0)}  D={ef.size(1)}")

    T = int(getattr(cfg, "diffusion_T", 100)); depth = int(getattr(cfg, "diffusion_depth", 4))
    ckpt = f"weights/diffusion_{tag}_{pooling}_T{T}.best.pt"
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    model = ColdDiffusionMLP(emb_dim=ef.size(1), hidden=sd["out.weight"].shape[1], depth=depth, T=T).to(dev)
    model.load_state_dict(sd); model.eval()
    with torch.no_grad():
        er = torch.cat([model.sample(eu[i:i + 1024].to(dev), steps=1).cpu() for i in range(0, eu.size(0), 1024)])

    cos_u = (unit(ef) * unit(eu)).sum(1).numpy()
    cos_r = (unit(ef) * unit(er)).sum(1).numpy()

    rng = np.random.default_rng(a.seed)
    n = min(a.n_tsne, ef.size(0))
    idx = np.sort(rng.choice(ef.size(0), size=n, replace=False))
    F, U, R = unit(ef[idx]), unit(eu[idx]), unit(er[idx])
    top1_u = float(((U @ F.T).argmax(1) == torch.arange(n)).float().mean())
    top1_r = float(((R @ F.T).argmax(1) == torch.arange(n)).float().mean())

    from sklearn.manifold import TSNE
    X = torch.cat([ef[idx], eu[idx], er[idx]]).numpy()
    coords = TSNE(n_components=2, perplexity=a.perplexity, init="pca", random_state=a.seed).fit_transform(X)
    group = np.repeat([0, 1, 2], n)
    any_label = Y[idx, labels.index("any")].numpy() if "any" in labels else np.zeros(n)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"tsne_cosine_{tag}.npz", coords=coords, group=group, idx=idx,
                        any_label=np.tile(any_label, 3), cos_u=cos_u, cos_r=cos_r,
                        top1_u=top1_u, top1_r=top1_r)
    summ = {"tag": tag, "N": int(ef.size(0)), "n_tsne": n,
            "cos_u_mean": float(cos_u.mean()), "cos_u_median": float(np.median(cos_u)),
            "cos_u_p5": float(np.percentile(cos_u, 5)),
            "cos_r_mean": float(cos_r.mean()), "cos_r_median": float(np.median(cos_r)),
            "cos_r_p5": float(np.percentile(cos_r, 5)),
            "top1_u": top1_u, "top1_r": top1_r}
    with open(out / f"tsne_cosine_{tag}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ)); w.writeheader(); w.writerow(summ)
    for k, v in summ.items():
        print(f"  {k:14s} {v:.4f}" if isinstance(v, float) else f"  {k:14s} {v}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for g, (lab, col) in enumerate((("$E_\\mathrm{f}$ (reference)", GRAY), ("$E_\\mathrm{u}$ (sparse)", ORANGE),
                                    ("$E_\\mathrm{r}$ (restored)", BLUE))):
        c = coords[group == g]
        axes[0].scatter(c[:, 0], c[:, 1], s=4, alpha=0.5, color=col, lw=0, label=lab)
    axes[0].set_xticks([]); axes[0].set_yticks([]); axes[0].legend(frameon=False, fontsize=7, markerscale=3)
    axes[0].set_title("(a) t-SNE", fontsize=9, loc="left")
    bins = np.linspace(min(cos_u.min(), cos_r.min()), 1, 60)
    axes[1].hist(cos_u, bins=bins, color=ORANGE, alpha=0.75, label="cos($E_\\mathrm{f}$, $E_\\mathrm{u}$)")
    axes[1].hist(cos_r, bins=bins, color=BLUE, alpha=0.75, label="cos($E_\\mathrm{f}$, $E_\\mathrm{r}$)")
    axes[1].set_xlabel("Per-slice cosine similarity", fontsize=8); axes[1].set_yticks([])
    axes[1].legend(frameon=False, fontsize=7); axes[1].set_title("(b) per-slice alignment", fontsize=9, loc="left")
    for s in ("top", "right", "left"):
        axes[1].spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"tsne_cosine_{tag}.{ext}", dpi=300, facecolor="white")
    print(f"saved → {out}/tsne_cosine_{tag}.(npz|csv|png|pdf)")


if __name__ == "__main__":
    main()
