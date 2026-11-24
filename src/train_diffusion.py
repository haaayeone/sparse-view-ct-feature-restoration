"""
train_diffusion.py — 3-stage CT diffusion pipeline.

Stage 1 : ColdDiffusionMLP  (E_under → E_full)
Stage 2 : Linear probe on E_full
Stage 3 : AUROC evaluation + t-SNE

Usage (서버, 저장소 루트에서):
    cd ~/hayeon/ich_cf_clean
    python train_diffusion.py configs/medsiglip_after/train_medsiglip_after_nv6.yaml
    # 5개 인코더 × 5개 투영 수 전부: bash scripts/train_all.sh

재현성: config의 seed(기본 42)로 python/numpy/torch 난수를 고정한다. 같은 인코더의
config들은 같은 E_full·같은 seed로 프로브를 학습하므로 투영 수와 무관하게 같은 프로브가 된다.

Pooling 변경 실험:
    train.yaml의 pooling 값을 gap | gem | grid 로 바꾸기만 하면 됨.
    feature map 재추출 없이 pooling만 바꿔서 빠르게 실험 가능.
"""

import logging
import os
import sys
from datetime import datetime
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion import ColdDiffusionMLP, EMA
from probe import LinearProbe
from data import CTFeatureMapDataset
from utils import get_config

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """python/numpy/torch(CPU·CUDA) 난수 고정 — 학습·평가 재현성."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_file_logging(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    ))
    logging.getLogger().addHandler(fh)
    logger.info(f"Logging to file: {log_path}")


def load_to_ram(dataset, num_workers: int = 1):
    loader = DataLoader(dataset, batch_size=512, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    all_full, all_under, all_Y = [], [], []
    for ef, eu, y in tqdm(loader, desc="  Loading to RAM", leave=False):
        all_full.append(ef); all_under.append(eu); all_Y.append(y)
    return torch.cat(all_full), torch.cat(all_under), torch.cat(all_Y)


def mean_cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_n = F.normalize(a.float(), dim=-1)
    b_n = F.normalize(b.float(), dim=-1)
    return (a_n * b_n).sum(-1).mean().item()


def compute_auroc(model: nn.Module, embeddings: torch.Tensor,
                  labels_Y: torch.Tensor, device: torch.device) -> float:
    from sklearn.metrics import roc_auc_score
    model.eval()
    N, bs, all_logits = embeddings.size(0), 512, []
    with torch.no_grad():
        for i in range(0, N, bs):
            e = embeddings[i : i + bs].to(device)
            logits = model(e)
            all_logits.append(torch.stack(logits, dim=1).cpu())
    logits_all = torch.cat(all_logits, dim=0).squeeze(-1)
    probs = torch.sigmoid(logits_all).numpy()
    aurocs = []
    for c in range(labels_Y.size(1)):
        y = labels_Y[:, c].numpy()
        if 0 < y.sum() < len(y):
            try:
                aurocs.append(roc_auc_score(y, probs[:, c]))
            except Exception:
                pass
    return float(np.mean(aurocs)) if aurocs else 0.0


def compute_auroc_per_label(model, embeddings, labels_Y, label_cols, device):
    from sklearn.metrics import roc_auc_score
    model.eval()
    N, bs, all_logits = embeddings.size(0), 512, []
    with torch.no_grad():
        for i in range(0, N, bs):
            e = embeddings[i : i + bs].to(device)
            logits = model(e)
            all_logits.append(torch.stack(logits, dim=1).cpu())
    logits_all = torch.cat(all_logits, dim=0).squeeze(-1)
    probs = torch.sigmoid(logits_all).numpy()
    result = {}
    for c, name in enumerate(label_cols):
        y = labels_Y[:, c].numpy()
        if 0 < y.sum() < len(y):
            try:
                result[name] = roc_auc_score(y, probs[:, c])
            except Exception:
                result[name] = float("nan")
        else:
            result[name] = float("nan")
    return result


# ── DeLong Test ────────────────────────────────────────────────────────────────

def get_probs(model: nn.Module, embeddings: torch.Tensor,
              device: torch.device) -> np.ndarray:
    """probe로 예측 확률 반환. shape: (N, num_labels)"""
    model.eval()
    N, bs, all_logits = embeddings.size(0), 512, []
    with torch.no_grad():
        for i in range(0, N, bs):
            e = embeddings[i : i + bs].to(device)
            logits = model(e)
            all_logits.append(torch.stack(logits, dim=1).cpu())
    logits_all = torch.cat(all_logits, dim=0).squeeze(-1)
    return torch.sigmoid(logits_all).numpy()


def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def _fastdelong(preds_sorted, label_1_count):
    m = label_1_count
    n = preds_sorted.shape[1] - m
    k = preds_sorted.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r] = _compute_midrank(preds_sorted[r, :m])
        ty[r] = _compute_midrank(preds_sorted[r, m:])
        tz[r] = _compute_midrank(preds_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(y_true: np.ndarray,
                preds_a: np.ndarray,
                preds_b: np.ndarray):
    """
    두 예측(preds_a, preds_b)의 AUROC 차이가 유의미한지 DeLong test.
    Returns: (auc_a, auc_b, p_value)
    """
    order = (-y_true).argsort()
    label_1_count = int(y_true.sum())
    if label_1_count == 0 or label_1_count == len(y_true):
        return float("nan"), float("nan"), float("nan")
    preds_sorted = np.vstack([preds_a, preds_b])[:, order]
    aucs, cov = _fastdelong(preds_sorted, label_1_count)
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, cov), l.T))
    p = float(2 * scipy.stats.norm.sf(float(z)))
    return float(aucs[0]), float(aucs[1]), p


def delong_per_label(probs_a: np.ndarray, probs_b: np.ndarray,
                     labels_Y: torch.Tensor, label_cols: list):
    """
    각 label에 대해 DeLong test 수행.
    Returns: dict {label: (auc_a, auc_b, p_value)}
    """
    results = {}
    for c, name in enumerate(label_cols):
        y = labels_Y[:, c].numpy()
        if 0 < y.sum() < len(y):
            results[name] = delong_test(y, probs_a[:, c], probs_b[:, c])
        else:
            results[name] = (float("nan"), float("nan"), float("nan"))
    return results


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_training_curves(train_losses, val_losses, cos_under, cos_restored,
                         save_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ep = range(1, len(train_losses) + 1)
        axes[0].plot(ep, train_losses, label="Train MSE")
        axes[0].plot(ep, val_losses,   label="Val MSE")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
        axes[0].set_title("Diffusion Training Loss"); axes[0].legend()

        axes[1].plot(ep, cos_under,    label="cos(E_full, E_under)")
        axes[1].plot(ep, cos_restored, label="cos(E_full, E_restored)")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Cosine Similarity")
        axes[1].set_title("Embedding Alignment"); axes[1].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Training curves saved → {save_path}")
    except Exception as e:
        logger.warning(f"Could not save training curves: {e}")


def plot_tsne(val_full, val_under, val_restored, val_Y, label_cols, tag, save_dir, pooling="gap"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        os.makedirs(save_dir, exist_ok=True)
        n   = min(800, val_full.size(0))
        idx = torch.randperm(val_full.size(0))[:n]
        stacked = np.concatenate([
            val_full[idx].float().numpy(),
            val_under[idx].float().numpy(),
            val_restored[idx].float().numpy(),
        ], axis=0)

        logger.info(f"t-SNE 실행 중... ({stacked.shape[0]} points, D={stacked.shape[1]})")
        coords = TSNE(n_components=2, perplexity=30, max_iter=500,
                      random_state=42).fit_transform(stacked)
        cf, cu, cr = coords[:n], coords[n : 2*n], coords[2*n:]

        for c, name in enumerate(label_cols):
            lbl = val_Y[idx, c].numpy()
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, coord, title in zip(axes, [cf, cu, cr],
                                         ["E_full", "E_under", "E_restored"]):
                sc = ax.scatter(coord[:, 0], coord[:, 1],
                                c=lbl, cmap="bwr", alpha=0.5, s=10)
                ax.set_title(title); ax.axis("off")
            plt.colorbar(sc, ax=axes[-1])
            plt.suptitle(f"t-SNE  label={name}")
            plt.tight_layout()
            fpath = os.path.join(save_dir, f"tsne_{pooling}_{name}.png")
            plt.savefig(fpath, dpi=150)
            plt.close()
            logger.info(f"  t-SNE saved → {fpath}")
    except Exception as e:
        logger.warning(f"t-SNE failed: {e}")


# ── Stage 1 ────────────────────────────────────────────────────────────────────

def train_stage1(opt, train_full, train_under, val_full, val_under,
                 emb_dim: int, device: torch.device):
    T       = getattr(opt, "diffusion_T",           100)
    hidden  = getattr(opt, "diffusion_hidden",      1024)
    depth   = getattr(opt, "diffusion_depth",       4)
    epochs  = getattr(opt, "diffusion_epochs",      200)
    lr      = getattr(opt, "diffusion_lr",          1e-4)
    wd      = getattr(opt, "diffusion_wd",          1e-4)
    bs      = getattr(opt, "diffusion_batch",       256)
    ema_d   = getattr(opt, "diffusion_ema_decay",   0.999)
    tag     = getattr(opt, "tag",                   "ct")
    pooling = getattr(opt, "pooling",               "gap")

    core   = ColdDiffusionMLP(emb_dim=emb_dim, hidden=hidden, depth=depth, T=T).to(device)
    ema    = EMA(core, decay=ema_d)
    optim  = torch.optim.AdamW(core.parameters(), lr=lr, weight_decay=wd)
    scaler = GradScaler("cuda") if torch.cuda.is_available() else None
    model  = nn.DataParallel(core) if torch.cuda.device_count() > 1 else core

    os.makedirs("weights", exist_ok=True)
    ckpt     = f"weights/diffusion_{tag}_{pooling}_T{T}.best.pt"
    N        = train_full.size(0)
    best_val = float("inf")

    hist_train, hist_val, hist_cos_under, hist_cos_rest = [], [], [], []

    logger.info(
        f"\n{'='*60}\n"
        f"Stage 1 — Cold Diffusion\n"
        f"  tag={tag}  pooling={pooling}  emb_dim={emb_dim}  T={T}\n"
        f"  hidden={hidden}  depth={depth}  epochs={epochs}  lr={lr}  EMA={ema_d}\n"
        f"  train_N={N}  val_N={val_full.size(0)}\n"
        f"  Baseline cos(E_full, E_under) = {mean_cos_sim(train_full, train_under):.4f}\n"
        f"{'='*60}"
    )

    for ep in range(1, epochs + 1):
        core.train()
        perm, losses = torch.randperm(N), []
        for i in range(0, N, bs):
            idx        = perm[i : i + bs]
            ef, eu     = train_full[idx].to(device), train_under[idx].to(device)
            t          = torch.randint(0, T, (ef.size(0),), device=device)
            x_t        = core.degrade(ef, eu, t)
            delta_true = eu - ef

            optim.zero_grad()
            if scaler:
                with autocast("cuda"):
                    loss = F.mse_loss(model(x_t, t, eu), delta_true)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(core.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                loss = F.mse_loss(model(x_t, t, eu), delta_true)
                loss.backward()
                nn.utils.clip_grad_norm_(core.parameters(), 1.0)
                optim.step()

            ema.update()
            losses.append(loss.item())

        core.eval()
        with torch.no_grad():
            ef_v, eu_v = val_full.to(device), val_under.to(device)
            t_v        = torch.randint(0, T, (ef_v.size(0),), device=device)
            x_v        = core.degrade(ef_v, eu_v, t_v)
            val_loss   = F.mse_loss(model(x_v, t_v, eu_v), eu_v - ef_v).item()

            steps        = getattr(opt, "diffusion_sample_steps", 1)
            er           = core.sample(eu_v, steps=steps)
            cos_u        = mean_cos_sim(ef_v.cpu(), eu_v.cpu())
            cos_r        = mean_cos_sim(ef_v.cpu(), er.cpu())
            mse_restored = F.mse_loss(er.cpu(), ef_v.cpu()).item()

        train_loss = sum(losses) / len(losses)
        hist_train.append(train_loss)
        hist_val.append(val_loss)
        hist_cos_under.append(cos_u)
        hist_cos_rest.append(cos_r)

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            orig = {k: v.clone() for k, v in core.state_dict().items()}
            ema.apply_shadow()
            torch.save(core.state_dict(), ckpt)
            ema.restore(orig)

        logger.info(
            f"Ep {ep:3d}/{epochs}  "
            f"train={train_loss:.6f}  val={val_loss:.6f}  "
            f"mse(full,rest)={mse_restored:.6f}  "
            f"cos(f,u)={cos_u:.4f}  cos(f,r)={cos_r:.4f}"
            + ("  ← BEST" if is_best else "")
        )

    logger.info(
        f"\nStage 1 done.  best_val={best_val:.6f}  ckpt={ckpt}\n"
        f"  Final cos(E_full, E_under)    = {hist_cos_under[-1]:.4f}\n"
        f"  Final cos(E_full, E_restored) = {hist_cos_rest[-1]:.4f}"
    )
    return ckpt, hist_train, hist_val, hist_cos_under, hist_cos_rest


# ── Stage 2 ────────────────────────────────────────────────────────────────────

def train_stage2(opt, train_full, train_Y, val_full, val_Y,
                 emb_dim: int, device: torch.device):
    num_labels = train_Y.size(1)
    pooling    = getattr(opt, "pooling", "gap")
    probe      = LinearProbe(emb_dim, num_labels).to(device)
    optim      = torch.optim.AdamW(
        probe.parameters(),
        lr=getattr(opt, "lr", 1e-4),
        weight_decay=getattr(opt, "weight_decay", 1e-4),
    )
    epochs   = getattr(opt, "num_epochs",      200)
    bs       = getattr(opt, "diffusion_batch", 256)
    patience = getattr(opt, "patience",        20)
    tag      = getattr(opt, "tag",             "ct")
    ckpt     = f"weights/linprobe_{tag}_{pooling}.best.pt"
    os.makedirs("weights", exist_ok=True)
    N, best_auroc, wait = train_full.size(0), 0.0, 0

    logger.info(
        f"\n{'='*60}\n"
        f"Stage 2 — Linear Probe\n"
        f"  pooling={pooling}  num_labels={num_labels}  epochs={epochs}\n"
        f"  ckpt={ckpt}  (pooling별 공유, acceleration 무관)\n"
        f"{'='*60}"
    )

    for ep in range(1, epochs + 1):
        probe.train()
        perm, ep_loss = torch.randperm(N), []
        for i in range(0, N, bs):
            idx    = perm[i : i + bs]
            e, y   = train_full[idx].to(device), train_Y[idx].to(device)
            logits = probe(e)
            loss   = sum(
                F.binary_cross_entropy_with_logits(logits[c], y[:, c : c + 1])
                for c in range(num_labels)
            ) / num_labels
            optim.zero_grad()
            loss.backward()
            optim.step()
            ep_loss.append(loss.item())

        if ep % 20 == 0 or ep in (1, epochs):
            auroc = compute_auroc(probe, val_full, val_Y, device)
            if auroc > best_auroc:
                best_auroc = auroc
                wait = 0
                torch.save(probe.state_dict(), ckpt)
            else:
                wait += 1
            logger.info(
                f"  Probe ep {ep:3d}  "
                f"loss={sum(ep_loss)/len(ep_loss):.5f}  "
                f"val_AUROC={auroc:.4f}  best={best_auroc:.4f}"
            )
            if wait >= patience // 20 and ep > 40:
                logger.info(f"  Early stopping at ep {ep}")
                break

    logger.info(f"Stage 2 done.  best_AUROC={best_auroc:.4f}  ckpt={ckpt}")
    probe.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return probe, best_auroc


# ── Stage 3 ────────────────────────────────────────────────────────────────────

def eval_stage3(diff_ckpt: str, emb_dim: int, probe,
                val_full, val_under, val_Y, label_cols, opt, device):
    T      = getattr(opt, "diffusion_T",            100)
    hidden = getattr(opt, "diffusion_hidden",       1024)
    depth  = getattr(opt, "diffusion_depth",        4)
    steps  = getattr(opt, "diffusion_sample_steps", 1)

    diff = ColdDiffusionMLP(emb_dim=emb_dim, hidden=hidden, depth=depth, T=T).to(device)
    diff.load_state_dict(torch.load(diff_ckpt, map_location=device, weights_only=True))
    diff.eval()

    restored = []
    with torch.no_grad():
        for i in range(0, val_under.size(0), 512):
            restored.append(
                diff.sample(val_under[i : i + 512].to(device), steps=steps).cpu()
            )
    val_restored = torch.cat(restored)

    probs_full  = get_probs(probe, val_full,     device)
    probs_under = get_probs(probe, val_under,    device)
    probs_rest  = get_probs(probe, val_restored, device)

    from sklearn.metrics import roc_auc_score
    def _mean_auroc(probs):
        aucs = []
        for c in range(val_Y.size(1)):
            y = val_Y[:, c].numpy()
            if 0 < y.sum() < len(y):
                try:
                    aucs.append(roc_auc_score(y, probs[:, c]))
                except Exception:
                    pass
        return float(np.mean(aucs)) if aucs else 0.0

    auroc_full  = _mean_auroc(probs_full)
    auroc_under = _mean_auroc(probs_under)
    auroc_rest  = _mean_auroc(probs_rest)

    per_label_full  = {label_cols[c]: roc_auc_score(val_Y[:, c].numpy(), probs_full[:, c])
                       for c in range(len(label_cols))
                       if 0 < val_Y[:, c].sum() < len(val_Y)}
    per_label_under = {label_cols[c]: roc_auc_score(val_Y[:, c].numpy(), probs_under[:, c])
                       for c in range(len(label_cols))
                       if 0 < val_Y[:, c].sum() < len(val_Y)}
    per_label_rest  = {label_cols[c]: roc_auc_score(val_Y[:, c].numpy(), probs_rest[:, c])
                       for c in range(len(label_cols))
                       if 0 < val_Y[:, c].sum() < len(val_Y)}

    # DeLong test: E_under vs E_restored (per label)
    delong_results = delong_per_label(probs_under, probs_rest, val_Y, label_cols)

    w = max(len(k) for k in label_cols) + 2
    per_label_str = "\n".join(
        f"    {k:<{w}} full={per_label_full.get(k, float('nan')):.4f}  "
        f"under={per_label_under.get(k, float('nan')):.4f}  "
        f"rest={per_label_rest.get(k, float('nan')):.4f}  "
        f"Δ={per_label_rest.get(k, float('nan'))-per_label_under.get(k, float('nan')):+.4f}"
        for k in label_cols
    )

    delong_str = "\n".join(
        f"    {k:<{w}} p={delong_results[k][2]:.4f}"
        + (" *" if delong_results[k][2] < 0.05 else "")
        for k in label_cols
    )

    logger.info(
        f"\n{'='*60}\n"
        f"Stage 3 — Evaluation\n"
        f"  AUROC E_full     = {auroc_full:.4f}   ← ceiling\n"
        f"  AUROC E_under    = {auroc_under:.4f}   ← degraded\n"
        f"  AUROC E_restored = {auroc_rest:.4f}   ← after diffusion\n"
        f"  Δ(under→restored) = {auroc_rest - auroc_under:+.4f}\n"
        f"  cos(E_full, E_under)    = {mean_cos_sim(val_full, val_under):.4f}\n"
        f"  cos(E_full, E_restored) = {mean_cos_sim(val_full, val_restored):.4f}\n"
        f"\n  Per-label AUROC:\n{per_label_str}\n"
        f"\n  DeLong test (E_under vs E_restored, * = p<0.05):\n{delong_str}\n"
        f"{'='*60}"
    )
    return val_restored, auroc_full, auroc_under, auroc_rest


# ── Main ───────────────────────────────────────────────────────────────────────

def run(opt):
    import pandas as pd

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_cols = getattr(opt, "label_cols",
                         ["epidural", "intraparenchymal", "intraventricular",
                          "subarachnoid", "subdural", "any"])
    pooling    = getattr(opt, "pooling", "gap")
    tag        = getattr(opt, "tag",     "ct")
    seed       = int(getattr(opt, "seed", 42))
    set_seed(seed)

    os.makedirs("logs", exist_ok=True)
    # tag/pooling이 같아도 실행마다 로그가 안 겹치도록 timestamp 부여.
    # sweep에서 호출되면 RUN_STAMP env로 동일 stamp를 공유한다.
    run_stamp = os.environ.get("RUN_STAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_file_logging(f"logs/{tag}_{pooling}_{run_stamp}.log")

    pool_kwargs = {}
    if pooling == "gem":
        pool_kwargs["gem_p"] = getattr(opt, "gem_p", 3.0)
    elif pooling == "grid":
        pool_kwargs["grid_size"] = getattr(opt, "grid_size", 2)

    logger.info(
        f"\nDevice   : {device}  GPUs={torch.cuda.device_count()}\n"
        f"Pooling  : {pooling}  {pool_kwargs}\n"
        f"Tag      : {tag}"
    )

    df = pd.read_csv(opt.csv_path)

    train_ds = CTFeatureMapDataset(df, opt.full_feature_dir, opt.under_feature_dir,
                                   label_cols, "train", pooling, **pool_kwargs)
    val_ds   = CTFeatureMapDataset(df, opt.full_feature_dir, opt.under_feature_dir,
                                   label_cols, "val",   pooling, **pool_kwargs)

    logger.info("Loading train data to RAM...")
    train_full, train_under, train_Y = load_to_ram(train_ds)
    logger.info("Loading val data to RAM...")
    val_full,   val_under,   val_Y   = load_to_ram(val_ds)

    emb_dim = train_full.size(1)
    logger.info(f"emb_dim={emb_dim}  train_N={train_full.size(0)}  val_N={val_full.size(0)}")

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    diff_ckpt, h_train, h_val, h_cos_u, h_cos_r = train_stage1(
        opt, train_full, train_under, val_full, val_under, emb_dim, device
    )

    os.makedirs("plots", exist_ok=True)
    plot_training_curves(
        h_train, h_val, h_cos_u, h_cos_r,
        save_path=f"plots/training_curves_{tag}_{pooling}.png",
    )

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    shared_probe_ckpt = f"weights/linprobe_{tag}_{pooling}.best.pt"
    if os.path.exists(shared_probe_ckpt):
        logger.info(
            f"\n{'='*60}\n"
            f"Stage 2 skip — 기존 probe 로드 (pooling={pooling})\n"
            f"  ckpt={shared_probe_ckpt}\n"
            f"{'='*60}"
        )
        num_labels = train_Y.size(1)
        probe = LinearProbe(emb_dim, num_labels).to(device)
        probe.load_state_dict(
            torch.load(shared_probe_ckpt, map_location=device, weights_only=True)
        )
    else:
        probe, _ = train_stage2(
            opt, train_full, train_Y, val_full, val_Y, emb_dim, device
        )

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    val_restored, *aurocs = eval_stage3(
        diff_ckpt, emb_dim, probe,
        val_full, val_under, val_Y, label_cols, opt, device,
    )

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    logger.info("\nt-SNE 생성 중...")
    plot_tsne(val_full, val_under, val_restored, val_Y, label_cols, tag,
              os.path.join("plots", f"tsne_{tag}"), pooling=pooling)

    logger.info(
        f"\n{'='*60}\n"
        f"DONE  tag={tag}  pooling={pooling}\n"
        f"  AUROC E_full     : {aurocs[0]:.4f}\n"
        f"  AUROC E_under    : {aurocs[1]:.4f}\n"
        f"  AUROC E_restored : {aurocs[2]:.4f}\n"
        f"  Improvement      : {aurocs[2]-aurocs[1]:+.4f}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    opt = get_config(sys.argv[1])
    run(opt)
