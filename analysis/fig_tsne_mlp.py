"""
fig_tsne_mlp.py — residual MLP로 복원한 임베딩의 t-SNE·코사인 정렬 (발표 자료 t-SNE 슬라이드, 원고 그림 4).

analysis/fig_tsne_cosine.py(원고 v0.1, diffusion 가중치)를 MLP 기준으로 바꾼 것.
view마다 test 전체에서 E_f(360-view 기준)·E_u(희소)·E_r = g 복원(1회 순전파)을 만들고
  - 코사인 cos(E_f, E_u), cos(E_f, E_r): test 전체의 평균·중앙값·5번째 백분위
  - top-1 대응률: 무작위 부분집합(n) 안에서 E_r(또는 E_u)의 최근접 E_f가 같은 슬라이스인 비율
  - t-SNE: 같은 부분집합의 [E_f; E_u; E_r]을 함께 2차원으로
을 계산한다. 가중치는 compare_restorers.py --save_weights 가 저장한 {tag}__mlp__s{seed}.pt 를
--weights_dirs 순서대로 찾는다. 없는 view는 건너뛰고, 가중치를 만드는 명령을 출력한다.

저장 (--out_dir, 기본 plots/tsne_mlp):
  tsne_mlp_{tag}.npz           coords(3n×2), group(0=E_f,1=E_u,2=E_r), idx, any_label, cos_u, cos_r(test 전체)
  tsne_mlp_summary_{enc}.csv   view별 요약 (코사인 평균·중앙값·p5, top-1)
  tsne_mlp_{enc}.png/.pdf      view별 t-SNE 패널을 가로로 이은 그림 (발표 자료용)

실행 위치: 서버 (GPU 있으면 복원이 빠름, t-SNE는 CPU). n=2000이면 view당 수 분.
Usage (서버):
  conda activate hayeon
  cd ~/hayeon/ich_cf_clean
  python analysis/fig_tsne_mlp.py --encoder medsiglip_after --views 45 12 6
가중치가 없다고 나오면 (같은 seed·레시피로 MLP만 다시 학습, 결과는 compare_tsne/에만 씀):
  python compare_restorers.py configs/medsiglip_after/train_medsiglip_after_nv6.yaml --seeds 42 --methods mlp --save_weights --out_dir compare_tsne
결과 가져오기 (로컬 PowerShell):
  scp -r neumous-lab@100.73.97.101:/home/neumous-lab/hayeon/ich_cf_clean/plots/tsne_mlp "C:\\Users\\user\\Desktop\\MIGT\\01. sparse ct project\\result\\"
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
from train_baselines import ResidualMLP
from train_diffusion import load_to_ram, set_seed
from utils import get_config

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"
LABELS_DEFAULT = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "any"]


def cfg_path(enc, v):          # scripts/compare_all.sh 와 같은 규칙
    if enc == "medsiglip_after":
        return f"configs/medsiglip_after/train_medsiglip_after_nv{v}.yaml"
    return f"configs/{enc}/train_{enc}_window_nv{v}.yaml"


def unit(x):
    return x / (x.norm(dim=1, keepdim=True) + 1e-8)


@torch.no_grad()
def restore(model, eu, dev, chunk=8192):
    model.eval()
    return torch.cat([model(eu[i:i + chunk].to(dev).float()).float().cpu() for i in range(0, eu.size(0), chunk)])


def one_view(enc, v, a, dev):
    cpath = cfg_path(enc, v)
    if not Path(cpath).exists():
        print(f"!! config 없음: {cpath}")
        return None
    cfg = get_config(cpath)
    tag, pooling = cfg.tag, getattr(cfg, "pooling", "none")
    wfile = next((Path(d) / f"{tag}__mlp__s{a.model_seed}.pt" for d in a.weights_dirs
                  if (Path(d) / f"{tag}__mlp__s{a.model_seed}.pt").exists()), None)
    if wfile is None:
        print(f"!! {v}-view MLP 가중치 없음 ({tag}__mlp__s{a.model_seed}.pt in {a.weights_dirs}) → 건너뜀. 만들려면:\n"
              f"   python compare_restorers.py {cpath} --seeds {a.model_seed} --methods mlp --save_weights "
              f"--out_dir compare_tsne")
        return None
    labels = list(getattr(cfg, "label_cols", LABELS_DEFAULT))
    df = pd.read_csv(cfg.csv_path)
    ds = CTFeatureMapDataset(df, cfg.full_feature_dir, cfg.under_feature_dir, labels, a.split, pooling)
    ef, eu, Y = load_to_ram(ds)
    ef, eu = ef.float(), eu.float()
    model = ResidualMLP(ef.size(1), int(getattr(cfg, "diffusion_hidden", 1024)),
                        int(getattr(cfg, "diffusion_depth", 4))).to(dev)
    model.load_state_dict(torch.load(wfile, map_location=dev, weights_only=True))
    er = restore(model, eu, dev)
    print(f"{tag}  {a.split}  N={ef.size(0)}  D={ef.size(1)}  weights={wfile}")

    cos_u = (unit(ef) * unit(eu)).sum(1).numpy()
    cos_r = (unit(ef) * unit(er)).sum(1).numpy()
    rng = np.random.default_rng(a.seed)
    n = min(a.n_tsne, ef.size(0))
    idx = np.sort(rng.choice(ef.size(0), size=n, replace=False))
    F_, U_, R_ = unit(ef[idx]), unit(eu[idx]), unit(er[idx])
    ar = torch.arange(n)
    top1_u = float(((U_ @ F_.T).argmax(1) == ar).float().mean())
    top1_r = float(((R_ @ F_.T).argmax(1) == ar).float().mean())

    from sklearn.manifold import TSNE
    X = torch.cat([ef[idx], eu[idx], er[idx]]).numpy()
    coords = TSNE(n_components=2, perplexity=a.perplexity, init="pca", random_state=a.seed).fit_transform(X)
    group = np.repeat([0, 1, 2], n)
    any_label = Y[idx, labels.index("any")].numpy() if "any" in labels else np.zeros(n)
    out = Path(a.out_dir)
    np.savez_compressed(out / f"tsne_mlp_{tag}.npz", coords=coords, group=group, idx=idx,
                        any_label=np.tile(any_label, 3), cos_u=cos_u, cos_r=cos_r, top1_u=top1_u, top1_r=top1_r)
    row = {"encoder": enc, "views": v, "tag": tag, "N": int(ef.size(0)), "n_tsne": n,
           "cos_u_mean": float(cos_u.mean()), "cos_u_median": float(np.median(cos_u)),
           "cos_u_p5": float(np.percentile(cos_u, 5)),
           "cos_r_mean": float(cos_r.mean()), "cos_r_median": float(np.median(cos_r)),
           "cos_r_p5": float(np.percentile(cos_r, 5)), "top1_u": top1_u, "top1_r": top1_r}
    for k, val in row.items():
        print(f"  {k:13s} {val:.4f}" if isinstance(val, float) else f"  {k:13s} {val}")
    return row, coords, group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="medsiglip_after")
    ap.add_argument("--views", type=int, nargs="+", default=[45, 12, 6])
    ap.add_argument("--model_seed", type=int, default=42, help="복원기 학습 seed (가중치 파일 이름)")
    ap.add_argument("--weights_dirs", nargs="+", default=["compare/weights", "compare_w6/weights", "compare_tsne/weights"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_tsne", type=int, default=2000)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42, help="부분집합·t-SNE seed")
    ap.add_argument("--out_dir", default="plots/tsne_mlp")
    a = ap.parse_args()

    set_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for v in a.views:
        r = one_view(a.encoder, v, a, dev)
        if r is not None:
            results.append((v, *r))
    if not results:
        raise SystemExit("그릴 view가 없음 (가중치 확인)")

    with open(out / f"tsne_mlp_summary_{a.encoder}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0][1]))
        w.writeheader()
        for _, row, _, _ in results:
            w.writerow(row)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = len(results)
    fig, axes = plt.subplots(1, k, figsize=(3.6 * k, 3.7), squeeze=False)
    names = (("Full-view $E_\\mathrm{f}$", GRAY), ("Sparse-view $E_\\mathrm{u}$", ORANGE),
             ("Restored $E_\\mathrm{r}$", BLUE))
    for ax, (v, row, coords, group) in zip(axes[0], results):
        for g, (lab, col) in enumerate(names):
            c = coords[group == g]
            ax.scatter(c[:, 0], c[:, 1], s=3, alpha=0.55, color=col, lw=0, label=lab, rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#cccccc")
        ax.set_title(f"{v} views   cos {row['cos_u_mean']:.3f} → {row['cos_r_mean']:.3f}", fontsize=10)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9, markerscale=4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"tsne_mlp_{a.encoder}.{ext}", dpi=300, facecolor="white")
    print(f"saved → {out}/tsne_mlp_{a.encoder}.(png|pdf), tsne_mlp_summary_{a.encoder}.csv")


if __name__ == "__main__":
    main()
