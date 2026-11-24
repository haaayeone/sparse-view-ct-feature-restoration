"""
compare_restorers.py — 복원기 공정 비교 + 학습 seed 반복 (피드백 1.1·1.2·2.3).

같은 (E_f, E_u) 쌍, 같은 고정 선형 프로브(weights/linprobe_{tag}_none.best.pt, 원고와 동일),
같은 학습 레시피, 같은 선택 기준(val 복원 MSE)으로 아래 방법들을 학습·평가한다.
기존 원고용 가중치(weights/diffusion_*.best.pt)는 읽지도 덮어쓰지도 않는다. 결과는 --out_dir(기본 compare/)에만 쓴다.

  닫힌해 (seed 무관, 1회)
    identity     E_r = E_u                                    복원 없음 (하한)
    meanshift    E_r = E_u + (μ_f − μ_u)                      평균 이동
    coral        E_r = (E_u − μ_u) C_u^-1/2 C_f^1/2 + μ_f      CORAL류 아핀 정렬 (쌍 정보 미사용, ε는 val로 선택)
    ridge        E_r = W E_u + b                              ridge 회귀 (쌍 사용, λ는 val로 선택)
  학습 (seed별) — 모두 AdamW 1e-4/1e-4, 배치 256, 200 에폭, grad clip 1.0, AMP, EMA 0.999 (원고 레시피)
    diffusion    원고의 cold diffusion (t ~ U{0..T−1}). 한 번의 학습에서 체크포인트 두 개를 고른다:
                   diffusion     val 복원 MSE(1단계, t = T−1) 최소 에폭의 EMA 가중치  ← 공정 비교용
                   diffusion_vl  val 손실(무작위 t, 원고 규칙) 최소 에폭의 EMA 가중치   ← 원고 재현·seed 반복용
    direct       같은 ColdDiffusionMLP(파라미터 수 동일)를 t = T−1, 입력 x = E_u 에서만 학습 = "같은 MLP의 직접 회귀"
    mlp          time embedding 없는 residual MLP (train_baselines.py와 같은 구조, 입력 E_u만)
  복원은 모두 1단계: E_r = E_u − D(E_u, E_u, T−1)  (mlp는 E_r = E_u − g(E_u)).

출력 (--out_dir):
  results/{tag}__{method}__s{seed}.json   지표(평균·라벨별 AUROC, R, cos, val/test 복원 MSE, 선택 에폭, 학습 곡선)
                                          닫힌해는 s0. 파일이 있으면 그 (tag, method, seed)는 건너뜀 → 끊겨도 같은 명령으로 이어서 실행
  preds/{tag}__ref_test.npz               slice_id, study_id, label_cols, y, p_full, p_under, cos_u
  preds/{tag}__{method}__s{seed}_test.npz p_rest, cos_r   (collect_restorers.py 의 study 단위 bootstrap용)
  weights/{tag}__{method}__s{seed}.pt     --save_weights 일 때만 (fp16, 모델당 ≈ 40 MB)
  logs/compare_{tag}_{stamp}.log

실행 위치: 서버 (GPU, 저장소 루트). 먼저 train_diffusion.py로 같은 config의 프로브가 만들어져 있어야 함(이미 있음).
학습 1회(200 에폭)는 GPU에 따라 대략 10–30분, 데이터 로딩은 config당 1회.
Usage (서버):
  conda activate hayeon
  cd ~/hayeon/ich_cf_clean
  # 1단계: MedSigLIP 6-view, seed 42, 전체 방법 (결과 확인용)
  python compare_restorers.py configs/medsiglip_after/train_medsiglip_after_nv6.yaml --seeds 42
  # seed 3개
  python compare_restorers.py configs/medsiglip_after/train_medsiglip_after_nv6.yaml --seeds 42 43 44
  # 일부 방법만
  python compare_restorers.py configs/medsiglip_after/train_medsiglip_after_nv6.yaml --seeds 42 --methods identity ridge direct
  # 5 인코더 × 6·23·90 view 전부: bash scripts/compare_all.sh  → python collect_restorers.py
  # 동작 확인(2 에폭, 결과는 compare_smoke/에): --epochs 2 --out_dir compare_smoke
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from data import CTFeatureMapDataset
from diffusion import ColdDiffusionMLP, EMA
from probe import LinearProbe
from train_baselines import ResidualMLP
from train_diffusion import load_to_ram, get_probs, set_seed
from utils import get_config

logger = logging.getLogger("compare")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _sh = logging.StreamHandler()
    _sh.setFormatter(logging.Formatter("%(asctime)s - %(message)s", "%m/%d/%Y %H:%M:%S"))
    logger.addHandler(_sh)

CLOSED = ["identity", "meanshift", "coral", "ridge"]
LEARNED = ["diffusion", "direct", "mlp"]          # diffusion → diffusion + diffusion_vl
ALL_METHODS = CLOSED + LEARNED
LABELS_DEFAULT = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "any"]


# ── 공통 ──────────────────────────────────────────────────────────────────────

def auroc_per_label(probs, Y, label_cols):
    from sklearn.metrics import roc_auc_score
    out = {}
    for c, name in enumerate(label_cols):
        y = Y[:, c]
        out[name] = float(roc_auc_score(y, probs[:, c])) if 0 < y.sum() < len(y) else float("nan")
    return out


def cos_per_slice(a, b):
    return (F.normalize(a.float(), dim=-1) * F.normalize(b.float(), dim=-1)).sum(-1)


def mse(a, b, chunk=65536):
    tot, n = 0.0, 0
    for i in range(0, a.size(0), chunk):
        d = a[i:i + chunk].float() - b[i:i + chunk].to(a.device).float()
        tot += float((d * d).sum())
        n += d.numel()
    return tot / n


def mean_cov(X, device, chunk=65536, cov=True):
    """float64 누적 평균(·공분산). X: (N, D) CPU 또는 GPU 텐서."""
    N, D = X.shape
    mu = torch.zeros(D, dtype=torch.float64, device=device)
    for i in range(0, N, chunk):
        mu += X[i:i + chunk].to(device).double().sum(0)
    mu /= N
    if not cov:
        return mu, None
    C = torch.zeros(D, D, dtype=torch.float64, device=device)
    for i in range(0, N, chunk):
        xc = X[i:i + chunk].to(device).double() - mu
        C += xc.T @ xc
    return mu, C / (N - 1)


def apply_affine(E, mu_in, A, mu_out, device, chunk=65536):
    """(E − mu_in) A + mu_out  (A=None 이면 항등) → float32 CPU"""
    outs = []
    for i in range(0, E.size(0), chunk):
        x = E[i:i + chunk].to(device).double() - mu_in
        if A is not None:
            x = x @ A
        outs.append((x + mu_out).float().cpu())
    return torch.cat(outs)


# ── 닫힌해 기준선 ──────────────────────────────────────────────────────────────

def fit_closed(name, tr_f, tr_u, va_f, va_u, device):
    """반환: predict(E) -> E_r (CPU float32), info dict"""
    if name == "identity":
        return (lambda E: E.float().cpu().clone()), {"n_params": 0}

    if name == "meanshift":
        mu_u, _ = mean_cov(tr_u, device, cov=False)
        mu_f, _ = mean_cov(tr_f, device, cov=False)
        return (lambda E: apply_affine(E, mu_u, None, mu_f, device)), {"n_params": int(mu_f.numel())}

    if name == "coral":
        mu_u, C_u = mean_cov(tr_u, device)
        mu_f, C_f = mean_cov(tr_f, device)
        w_u, V_u = torch.linalg.eigh(C_u)
        w_f, V_f = torch.linalg.eigh(C_f)
        w_u, w_f = w_u.clamp_min(0), w_f.clamp_min(0)
        best = None
        for reg in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            eu, ef = reg * w_u.mean(), reg * w_f.mean()            # ε = reg × 평균 고유값
            whiten = (V_u * (w_u + eu).rsqrt()) @ V_u.T
            color = (V_f * (w_f + ef).sqrt()) @ V_f.T
            A = whiten @ color
            v = mse(apply_affine(va_u, mu_u, A, mu_f, device), va_f)
            logger.info(f"  coral reg={reg:g}: val restoration MSE={v:.6f}")
            if best is None or v < best[0]:
                best = (v, reg, A)
        v, reg, A = best
        logger.info(f"  → coral reg={reg:g}")
        return (lambda E: apply_affine(E, mu_u, A, mu_f, device)), {"n_params": int(A.numel() + 2 * mu_f.numel()),
                                                                    "reg": reg}

    if name == "ridge":
        mu_u, C_u = mean_cov(tr_u, device)
        mu_f, _ = mean_cov(tr_f, device, cov=False)
        N, D = tr_u.shape
        XtX = C_u * (N - 1)
        XtY = torch.zeros(D, D, dtype=torch.float64, device=device)
        for i in range(0, N, 65536):
            XtY += (tr_u[i:i + 65536].to(device).double() - mu_u).T @ (tr_f[i:i + 65536].to(device).double() - mu_f)
        eye = torch.eye(D, dtype=torch.float64, device=device)
        best = None
        for lam in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
            W = torch.linalg.solve(XtX + lam * N * eye, XtY)
            v = mse(apply_affine(va_u, mu_u, W, mu_f, device), va_f)
            logger.info(f"  ridge λ={lam:g}: val restoration MSE={v:.6f}")
            if best is None or v < best[0]:
                best = (v, lam, W)
        v, lam, W = best
        logger.info(f"  → ridge λ={lam:g}")
        return (lambda E: apply_affine(E, mu_u, W, mu_f, device)), {"n_params": int(W.numel() + 2 * mu_f.numel()),
                                                                    "lambda": lam}
    raise ValueError(name)


# ── 학습 기준선 (같은 레시피) ─────────────────────────────────────────────────

def build_model(method, emb_dim, hp):
    if method in ("diffusion", "direct"):
        return ColdDiffusionMLP(emb_dim=emb_dim, hidden=hp["hidden"], depth=hp["depth"], T=hp["T"])
    if method == "mlp":
        return ResidualMLP(emb_dim, hp["hidden"], hp["depth"])
    raise ValueError(method)


@torch.no_grad()
def restore(model, method, E, T, device, chunk=8192):
    """1단계 복원. E: (N, D) → E_r (N, D) CPU float32"""
    model.eval()
    outs = []
    for i in range(0, E.size(0), chunk):
        x = E[i:i + chunk].to(device).float()
        if method == "mlp":
            outs.append(model(x).float().cpu())
        else:
            t = torch.full((x.size(0),), T - 1, device=device, dtype=torch.long)
            outs.append((x - model(x, t, x)).float().cpu())
    return torch.cat(outs)


@torch.no_grad()
def val_loss_random_t(model, va_f, va_u, T, device, chunk=8192):
    """원고 규칙의 val 손실: 무작위 t에서 열화 → Δ 예측 MSE (raw 가중치, train_diffusion.train_stage1과 같은 정의)."""
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, va_f.size(0), chunk):
        ef, eu = va_f[i:i + chunk].to(device).float(), va_u[i:i + chunk].to(device).float()
        t = torch.randint(0, T, (ef.size(0),), device=device)
        d = model(model.degrade(ef, eu, t), t, eu) - (eu - ef)
        tot += float((d * d).sum())
        n += d.numel()
    return tot / n


def train_learned(method, tr_f, tr_u, va_f, va_u, emb_dim, hp, seed, device, use_amp):
    """반환: {variant: (state_dict_cpu, best_epoch, best_value, select_rule)}, curves, n_params, seconds"""
    set_seed(seed)
    model = build_model(method, emb_dim, hp).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    ema = EMA(model, decay=hp["ema"])
    optim = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    scaler = GradScaler("cuda") if use_amp else None
    T, bs, N = hp["T"], hp["bs"], tr_f.size(0)

    def loss_fn(ef, eu):
        if method == "diffusion":
            t = torch.randint(0, T, (ef.size(0),), device=device)
            return F.mse_loss(model(model.degrade(ef, eu, t), t, eu), eu - ef)
        if method == "direct":
            t = torch.full((ef.size(0),), T - 1, device=device, dtype=torch.long)
            return F.mse_loss(model(eu, t, eu), eu - ef)
        return F.mse_loss(model(eu), ef)                       # mlp: E_u − g(E_u) ≈ E_f  ⇔  g ≈ Δ

    curves = {"train": [], "val_restore_ema": []}
    if method == "diffusion":
        curves["val_loss_randt"] = []
    best = {}                                                   # variant → (value, epoch, state)
    t0 = time.time()
    for ep in range(1, hp["epochs"] + 1):
        model.train()
        perm, losses = torch.randperm(N), []
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            if tr_f.is_cuda:
                idx = idx.to(device)
            ef, eu = tr_f[idx].to(device).float(), tr_u[idx].to(device).float()
            optim.zero_grad()
            if scaler is not None:
                with autocast("cuda"):
                    loss = loss_fn(ef, eu)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                loss = loss_fn(ef, eu)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            ema.update()
            losses.append(loss.item())
        curves["train"].append(float(np.mean(losses)))

        # (a) 원고 규칙: raw 가중치의 무작위-t val 손실 → 그 에폭의 EMA 가중치 저장
        vl = val_loss_random_t(model, va_f, va_u, T, device) if method == "diffusion" else None
        # (b) 공정 비교 규칙: EMA 가중치의 val 복원 MSE (1단계)
        orig = {k: v.clone() for k, v in model.state_dict().items()}
        ema.apply_shadow()
        vr = mse(restore(model, method, va_u, T, device), va_f)
        ema_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        ema.restore(orig)
        curves["val_restore_ema"].append(vr)

        flags = []
        if vl is not None:
            curves["val_loss_randt"].append(vl)
            if "diffusion_vl" not in best or vl < best["diffusion_vl"][0]:
                best["diffusion_vl"] = (vl, ep, ema_sd)
                flags.append("best(val_loss)")
        if method not in best or vr < best[method][0]:
            best[method] = (vr, ep, ema_sd)
            flags.append("best(val_restore)")
        if ep == 1 or ep % 10 == 0 or ep == hp["epochs"]:
            logger.info(f"  {method} s{seed} ep {ep:3d}/{hp['epochs']}  train={curves['train'][-1]:.6f}  "
                        + (f"val_loss(t~U)={vl:.6f}  " if vl is not None else "")
                        + f"val_restore(EMA)={vr:.6f}  {' '.join(flags)}  ({time.time() - t0:.0f}s)")
    rules = {"diffusion_vl": "val_loss_random_t (manuscript rule)"}
    out = {k: (v[2], v[1], v[0], rules.get(k, "val_restoration_mse")) for k, v in best.items()}
    return out, curves, n_params, time.time() - t0


# ── 평가·저장 ─────────────────────────────────────────────────────────────────

def evaluate(er_test, er_val, ctx):
    probe, device = ctx["probe"], ctx["device"]
    p_rest = get_probs(probe, er_test, device).astype(np.float32)
    per = auroc_per_label(p_rest, ctx["y_test"], ctx["label_cols"])
    a_r = float(np.nanmean(list(per.values())))
    a_f, a_u = ctx["auroc_f"], ctx["auroc_u"]
    cos_r = cos_per_slice(ctx["te_f"], er_test).numpy().astype(np.float32)
    res = dict(auroc_f=a_f, auroc_u=a_u, auroc_r=a_r, delta=a_r - a_u,
               R_pct=100 * (a_r - a_u) / (a_f - a_u) if a_f > a_u else float("nan"),
               cos_u=float(ctx["cos_u"].mean()), cos_r=float(cos_r.mean()),
               per_label_r=per, per_label_f=ctx["per_f"], per_label_u=ctx["per_u"],
               val_restore_mse=mse(er_val, ctx["va_f"]), test_restore_mse=mse(er_test, ctx["te_f"]))
    return res, p_rest, cos_r


def save_result(ctx, method, seed, res, p_rest, cos_r, extra):
    key = f"{ctx['tag']}__{method}__s{seed}"
    np.savez_compressed(ctx["out"] / "preds" / f"{key}_test.npz", p_rest=p_rest, cos_r=cos_r)
    rec = dict(tag=ctx["tag"], method=method, seed=seed, **extra, **res, created=datetime.now().isoformat())
    tmp = ctx["out"] / "results" / f"{key}.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, ctx["out"] / "results" / f"{key}.json")      # 결과 파일은 마지막에 원자적으로 → 이어서 실행 판단 기준
    logger.info(f"[{method:<12} s{seed}] AUROC E_r={res['auroc_r']:.4f}  Δ={res['delta']:+.4f}  "
                f"R={res['R_pct']:.1f}%  cos(f,r)={res['cos_r']:.4f}  "
                f"val MSE={res['val_restore_mse']:.6f}  test MSE={res['test_restore_mse']:.6f}")


def done(out, tag, method, seed):
    return (out / "results" / f"{tag}__{method}__s{seed}.json").exists()


def run(cfg_path, a):
    opt = get_config(cfg_path)
    tag, pooling = opt.tag, getattr(opt, "pooling", "none")
    label_cols = getattr(opt, "label_cols", LABELS_DEFAULT)
    out = Path(a.out_dir)
    for sub in ("results", "preds", "weights", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    if a.train_frac < 1.0:
        tag = f"{tag}_frac{a.train_frac:g}"

    methods = a.methods
    jobs = [(m, 0) for m in methods if m in CLOSED and not done(out, tag, m, 0)]
    for s in a.seeds:
        for m in methods:
            if m in LEARNED:
                variants = ["diffusion", "diffusion_vl"] if m == "diffusion" else [m]
                if not all(done(out, tag, v, s) for v in variants):
                    jobs.append((m, s))
    if not jobs:
        logger.info(f"---- skip {tag} (모든 결과 있음: {out}/results)")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(out / "logs" / f"compare_{tag}_{stamp}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s", "%m/%d/%Y %H:%M:%S"))
    logger.addHandler(fh)
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_amp = device.type == "cuda" and not a.no_amp
        hp = dict(T=int(getattr(opt, "diffusion_T", 100)), hidden=int(getattr(opt, "diffusion_hidden", 1024)),
                  depth=int(getattr(opt, "diffusion_depth", 4)),
                  epochs=int(a.epochs or getattr(opt, "diffusion_epochs", 200)),
                  lr=float(getattr(opt, "diffusion_lr", 1e-4)), wd=float(getattr(opt, "diffusion_wd", 1e-4)),
                  bs=int(getattr(opt, "diffusion_batch", 256)), ema=float(getattr(opt, "diffusion_ema_decay", 0.999)))
        logger.info(f"==== {tag}  config={cfg_path}  device={device}  amp={use_amp}\n"
                    f"     jobs={jobs}\n     hp={hp}  train_frac={a.train_frac}")

        import pandas as pd
        df = pd.read_csv(opt.csv_path)
        ds = {s: CTFeatureMapDataset(df, opt.full_feature_dir, opt.under_feature_dir, label_cols, s, pooling)
              for s in ("train", "val", "test")}
        data = {}
        for s in ("train", "val", "test"):
            logger.info(f"Loading {s} ({len(ds[s])} slices)...")
            data[s] = load_to_ram(ds[s], num_workers=a.num_workers)
        (tr_f, tr_u, _), (va_f, va_u, _), (te_f, te_u, te_Y) = data["train"], data["val"], data["test"]

        if a.train_frac < 1.0:                                   # 검사 단위로 학습 데이터 일부만 (적은 데이터 실험용)
            sid = ds["train"].df[a.study_col].astype(str).values
            studies = np.unique(sid)
            keep = np.random.default_rng(0).choice(studies, max(1, int(round(a.train_frac * len(studies)))),
                                                    replace=False)
            m = torch.from_numpy(np.isin(sid, keep))
            tr_f, tr_u = tr_f[m], tr_u[m]
            logger.info(f"train_frac={a.train_frac}: {len(keep)}/{len(studies)} studies, {int(m.sum())} slices")

        emb_dim = tr_f.size(1)
        # 학습 데이터를 GPU에 fp16으로 올림 (feature 원본이 float16 .npy라 손실 없음). 레시피는 같고 속도만 다름.
        if device.type == "cuda" and a.data_on_gpu != "no":
            need = 2 * tr_f.numel() * 2
            free = torch.cuda.mem_get_info()[0]
            if a.data_on_gpu == "yes" or need < 0.6 * free:
                tr_f, tr_u = tr_f.half().to(device), tr_u.half().to(device)
                logger.info(f"train data on GPU (fp16, {need / 1e9:.1f} GB)")

        probe_ckpt = f"weights/linprobe_{opt.tag}_{pooling}.best.pt"
        if not os.path.exists(probe_ckpt):
            raise SystemExit(f"프로브 없음: {probe_ckpt} (train_diffusion.py로 먼저 학습)")
        probe = LinearProbe(emb_dim, len(label_cols)).to(device)
        probe.load_state_dict(torch.load(probe_ckpt, map_location=device, weights_only=True))
        probe.eval()

        y_test = te_Y.numpy().astype(np.uint8)
        p_full = get_probs(probe, te_f, device).astype(np.float32)
        p_under = get_probs(probe, te_u, device).astype(np.float32)
        per_f, per_u = auroc_per_label(p_full, y_test, label_cols), auroc_per_label(p_under, y_test, label_cols)
        cos_u = cos_per_slice(te_f, te_u).numpy().astype(np.float32)
        ctx = dict(tag=tag, out=out, probe=probe, device=device, label_cols=label_cols, y_test=y_test,
                   te_f=te_f, va_f=va_f, cos_u=cos_u, per_f=per_f, per_u=per_u,
                   auroc_f=float(np.nanmean(list(per_f.values()))), auroc_u=float(np.nanmean(list(per_u.values()))))
        logger.info(f"probe={probe_ckpt}  emb_dim={emb_dim}  train_N={tr_f.size(0)}  test_N={te_f.size(0)}  "
                    f"AUROC E_f={ctx['auroc_f']:.4f}  E_u={ctx['auroc_u']:.4f}  cos(f,u)={cos_u.mean():.4f}")
        ref = out / "preds" / f"{tag}__ref_test.npz"
        if not ref.exists():
            np.savez_compressed(ref, slice_id=np.asarray(ds["test"].df["slice_id"].astype(str), dtype=str),
                                study_id=np.asarray(ds["test"].df[a.study_col].astype(str), dtype=str),
                                label_cols=np.array(label_cols), y=y_test, p_full=p_full, p_under=p_under,
                                cos_u=cos_u)
        base = dict(encoder_tag=opt.tag, config=cfg_path, emb_dim=emb_dim, train_N=int(tr_f.size(0)),
                    train_frac=a.train_frac, hp=hp, amp=use_amp)

        for m, s in jobs:
            logger.info(f"\n---- {tag}  method={m}  seed={s if m in LEARNED else '-'}")
            if m in CLOSED:
                t0 = time.time()
                predict, info = fit_closed(m, tr_f, tr_u, va_f, va_u, device)
                res, p_rest, cos_r = evaluate(predict(te_u), predict(va_u), ctx)
                save_result(ctx, m, 0, res, p_rest, cos_r,
                            dict(base, selection="val_restoration_mse (hyperparameter)", seconds=time.time() - t0,
                                 **info))
                continue
            variants, curves, n_params, secs = train_learned(m, tr_f, tr_u, va_f, va_u, emb_dim, hp, s, device,
                                                             use_amp)
            model = build_model(m, emb_dim, hp).to(device)
            for v, (sd, ep, val, rule) in variants.items():
                if done(out, tag, v, s):
                    continue
                model.load_state_dict(sd)
                res, p_rest, cos_r = evaluate(restore(model, m, te_u, hp["T"], device),
                                              restore(model, m, va_u, hp["T"], device), ctx)
                save_result(ctx, v, s, res, p_rest, cos_r,
                            dict(base, selection=rule, best_epoch=ep, best_value=val, n_params=n_params,
                                 seconds=secs, curves=curves))
                if a.save_weights:
                    torch.save({k: t.half() for k, t in sd.items()}, out / "weights" / f"{tag}__{v}__s{s}.pt")
            del model, variants
            if device.type == "cuda":
                torch.cuda.empty_cache()
        logger.info(f"==== done {tag}")
    finally:
        logger.removeHandler(fh)
        fh.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+", help="train yaml (train_diffusion.py와 같은 것)")
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42], help="학습 seed (닫힌해는 seed 무관)")
    ap.add_argument("--out_dir", default="compare")
    ap.add_argument("--epochs", type=int, default=None, help="에폭 수 덮어쓰기 (동작 확인용; 기본 config의 200)")
    ap.add_argument("--train_frac", type=float, default=1.0, help="학습 검사 비율 (<1이면 tag에 _frac 붙음)")
    ap.add_argument("--study_col", default="study_id")
    ap.add_argument("--data_on_gpu", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--save_weights", action="store_true")
    ap.add_argument("--num_workers", type=int, default=8, help="feature 로딩 DataLoader worker 수")
    a = ap.parse_args()
    for c in a.configs:
        run(c, a)


if __name__ == "__main__":
    main()
