"""
train_baselines.py — diffusion 정당화용 deterministic regression baselines.

동일한 (E_full, E_under) 쌍·동일한 고정 linear probe로 세 baseline을 평가해
cold-diffusion 복원과 직접 비교한다:

  identity : E_r = E_u                       (복원 없음 — 하한선)
  ridge    : E_r = W E_u + b                 (닫힌해 선형회귀, λ는 val로 선택)
  mlp      : E_r = E_u - g(E_u)              (residual MLP — diffusion과 같은
                                              SwiGLU 4블록×1024, time-emb만 없음)

config는 train_diffusion.py와 동일한 yaml을 그대로 사용한다
(csv_path, full_feature_dir, under_feature_dir, tag, pooling).
probe는 학습하지 않고 기존 weights/linprobe_{tag}_{pooling}.best.pt 를 로드한다.

Usage (서버, 저장소 루트에서; 먼저 train_diffusion.py로 같은 config의 프로브가 있어야 함):
    cd ~/hayeon/ich_cf_clean
    python train_baselines.py configs/medsiglip_after/train_medsiglip_after_nv6.yaml --baseline all --split test
    python train_baselines.py configs/medsiglip_after/train_medsiglip_after_nv6.yaml --baseline ridge --split test
    python collect_baselines.py        # logs/baseline_*.log → results_baselines.csv
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import CTFeatureMapDataset
from probe import LinearProbe
from utils import get_config
from train_diffusion import (
    load_to_ram, mean_cos_sim, get_probs, delong_per_label, set_seed,
)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


# ── Residual MLP (diffusion 본체와 동일 구조, time-embedding 제외) ─────────────

class SwiGLUBlock(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden * 2)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.res = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        gate, val = self.fc1(x).chunk(2, dim=-1)
        return self.norm(self.fc2(F.silu(gate) * val) + self.res(x))


class ResidualMLP(nn.Module):
    """E_r = E_u - g(E_u).  g = SwiGLU×depth(=diffusion과 동일 폭/깊이)."""

    def __init__(self, emb_dim, hidden=1024, depth=4):
        super().__init__()
        layers = [SwiGLUBlock(emb_dim, hidden, hidden)]
        for _ in range(depth - 1):
            layers.append(SwiGLUBlock(hidden, hidden, hidden))
        self.blocks = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, emb_dim)

    def forward(self, e_u):
        return e_u - self.out(self.blocks(e_u))


# ── Baselines ──────────────────────────────────────────────────────────────────

def fit_ridge(train_u, train_f, val_u, val_f, device,
              lambdas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """닫힌해 ridge: E_r = W E_u + b.  λ는 val 복원 MSE 최소로 선택."""
    Xu = train_u.to(device).float()
    Yf = train_f.to(device).float()
    mu_u, mu_f = Xu.mean(0, keepdim=True), Yf.mean(0, keepdim=True)
    Xc, Yc = Xu - mu_u, Yf - mu_f
    D = Xu.size(1)
    XtX = Xc.T @ Xc                       # (D, D)
    XtY = Xc.T @ Yc
    eye = torch.eye(D, device=device)

    best = None
    vu = val_u.to(device).float()
    vf = val_f.to(device).float()
    for lam in lambdas:
        W = torch.linalg.solve(XtX + lam * len(Xu) * eye, XtY)   # (D, D)
        pred = (vu - mu_u) @ W + mu_f
        mse = F.mse_loss(pred, vf).item()
        logger.info(f"  ridge λ={lam:g}: val restoration MSE={mse:.6f}")
        if best is None or mse < best[0]:
            best = (mse, lam, W.clone())
    mse, lam, W = best
    logger.info(f"  → selected λ={lam:g} (val MSE={mse:.6f})")

    def predict(e_u):
        return (e_u.to(device).float() - mu_u) @ W + mu_f
    n_params = W.numel() + mu_u.numel() + mu_f.numel()
    return predict, n_params


def fit_mlp(train_u, train_f, val_u, val_f, emb_dim, device,
            hidden=1024, depth=4, epochs=200, bs=256, lr=1e-4, wd=1e-4):
    """residual MLP를 diffusion과 동일 레시피(AdamW/에폭/배치)로 학습.
    checkpoint는 (leaky denoising loss가 아니라) val '복원' MSE 최소로 선택."""
    model = ResidualMLP(emb_dim, hidden, depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    N = train_u.size(0)
    vu = val_u.to(device).float()
    vf = val_f.to(device).float()

    best_mse, best_sd = float("inf"), None
    for ep in range(1, epochs + 1):
        model.train()
        perm, losses = torch.randperm(N), []
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            eu = train_u[idx].to(device).float()
            ef = train_f[idx].to(device).float()
            loss = F.mse_loss(model(eu), ef)
            optim.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            vmse = F.mse_loss(model(vu), vf).item()
        if vmse < best_mse:
            best_mse = vmse
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 20 == 0 or ep == 1:
            logger.info(f"  mlp ep {ep:3d}: train={np.mean(losses):.6f}  "
                        f"val_restore={vmse:.6f}  best={best_mse:.6f}")
    model.load_state_dict(best_sd)

    @torch.no_grad()
    def predict(e_u, chunk=4096):
        outs = []
        for i in range(0, e_u.size(0), chunk):
            outs.append(model(e_u[i:i + chunk].to(device).float()).cpu())
        return torch.cat(outs).to(device)
    return predict, n_params


# ── 평가 ────────────────────────────────────────────────────────────────────────

def mean_auroc(probs, Y):
    from sklearn.metrics import roc_auc_score
    aucs = []
    for c in range(Y.size(1)):
        y = Y[:, c].numpy()
        if 0 < y.sum() < len(y):
            aucs.append(roc_auc_score(y, probs[:, c]))
    return float(np.mean(aucs)) if aucs else 0.0


def run(opt, baselines, split):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_cols = getattr(opt, "label_cols",
                         ["epidural", "intraparenchymal", "intraventricular",
                          "subarachnoid", "subdural", "any"])
    pooling = getattr(opt, "pooling", "none")
    tag = getattr(opt, "tag", "ct")
    set_seed(int(getattr(opt, "seed", 42)))

    os.makedirs("logs", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(f"logs/baseline_{tag}_{pooling}_{split}_{stamp}.log",
                             mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s", "%m/%d/%Y %H:%M:%S"))
    logging.getLogger().addHandler(fh)

    import pandas as pd
    df = pd.read_csv(opt.csv_path)
    pool_kwargs = {}
    if pooling == "gem":
        pool_kwargs["gem_p"] = getattr(opt, "gem_p", 3.0)
    elif pooling == "grid":
        pool_kwargs["grid_size"] = getattr(opt, "grid_size", 2)

    logger.info(f"tag={tag} pooling={pooling} eval split={split} baselines={baselines}")
    tr = CTFeatureMapDataset(df, opt.full_feature_dir, opt.under_feature_dir,
                             label_cols, "train", pooling, **pool_kwargs)
    ev = CTFeatureMapDataset(df, opt.full_feature_dir, opt.under_feature_dir,
                             label_cols, split, pooling, **pool_kwargs)
    va = CTFeatureMapDataset(df, opt.full_feature_dir, opt.under_feature_dir,
                             label_cols, "val", pooling, **pool_kwargs)

    logger.info("Loading train..."); tr_f, tr_u, _ = load_to_ram(tr)
    logger.info("Loading val...");   va_f, va_u, _ = load_to_ram(va)
    logger.info(f"Loading {split}..."); ev_f, ev_u, ev_Y = load_to_ram(ev)
    emb_dim = tr_f.size(1)
    logger.info(f"emb_dim={emb_dim}  train_N={tr_f.size(0)}  {split}_N={ev_f.size(0)}")

    # 고정 probe 로드 (diffusion 실험과 동일한 것)
    probe_ckpt = f"weights/linprobe_{tag}_{pooling}.best.pt"
    probe = LinearProbe(emb_dim, len(label_cols)).to(device)
    probe.load_state_dict(torch.load(probe_ckpt, map_location=device, weights_only=True))
    probe.eval()
    logger.info(f"probe loaded: {probe_ckpt}")

    probs_f = get_probs(probe, ev_f, device)
    probs_u = get_probs(probe, ev_u, device)
    a_full, a_under = mean_auroc(probs_f, ev_Y), mean_auroc(probs_u, ev_Y)
    logger.info(f"AUROC E_full={a_full:.4f}  E_under={a_under:.4f}  "
                f"cos(f,u)={mean_cos_sim(ev_f, ev_u):.4f}")

    results = []
    for name in baselines:
        logger.info(f"\n--- baseline: {name} ---")
        if name == "identity":
            ev_r = ev_u.clone(); n_params = 0
        elif name == "ridge":
            predict, n_params = fit_ridge(tr_u, tr_f, va_u, va_f, device)
            ev_r = predict(ev_u).cpu()
        elif name == "mlp":
            predict, n_params = fit_mlp(tr_u, tr_f, va_u, va_f, emb_dim, device,
                                        hidden=getattr(opt, "diffusion_hidden", 1024),
                                        depth=getattr(opt, "diffusion_depth", 4),
                                        epochs=getattr(opt, "diffusion_epochs", 200),
                                        bs=getattr(opt, "diffusion_batch", 256),
                                        lr=getattr(opt, "diffusion_lr", 1e-4),
                                        wd=getattr(opt, "diffusion_wd", 1e-4))
            ev_r = predict(ev_u).cpu()
        else:
            raise ValueError(name)

        probs_r = get_probs(probe, ev_r, device)
        a_rest = mean_auroc(probs_r, ev_Y)
        cos_r = mean_cos_sim(ev_f, ev_r)
        gap = a_full - a_under
        rec = (a_rest - a_under) / gap * 100 if gap > 0 else float("nan")
        dl = delong_per_label(probs_u, probs_r, ev_Y, label_cols)
        n_sig = sum(1 for v in dl.values() if v[2] == v[2] and v[2] < 0.05)
        logger.info(
            f"[{name}] params={n_params:,}  AUROC E_r={a_rest:.4f}  "
            f"Δ={a_rest - a_under:+.4f}  recovery={rec:.1f}%  "
            f"cos(f,r)={cos_r:.4f}  DeLong sig {n_sig}/{len(label_cols)}")
        results.append((name, n_params, a_rest, a_rest - a_under, rec, cos_r))

    logger.info("\n" + "=" * 68)
    logger.info(f"SUMMARY  tag={tag} split={split}  "
                f"(E_full={a_full:.4f}, E_under={a_under:.4f})")
    logger.info(f"{'baseline':<10}{'params':>12}{'AUROC_r':>10}{'Δ':>9}"
                f"{'rec%':>8}{'cos(f,r)':>10}")
    for name, np_, ar, d, rec, cr in results:
        logger.info(f"{name:<10}{np_:>12,}{ar:>10.4f}{d:>+9.4f}{rec:>8.1f}{cr:>10.4f}")
    logger.info("=" * 68)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--baseline", default="all",
                    help="identity | ridge | mlp | all (쉼표 조합 가능)")
    ap.add_argument("--split", default="test", help="평가 split (기본 test)")
    args = ap.parse_args()
    names = (["identity", "ridge", "mlp"] if args.baseline == "all"
             else [b.strip() for b in args.baseline.split(",")])
    run(get_config(args.config), names, args.split)
