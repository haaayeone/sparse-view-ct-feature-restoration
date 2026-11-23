"""
exp_nocond/train_nocond.py — diffusion 마지막 시도 (A + C 변형), MedSigLIP 6-view · seed 42 한 번만.

원고 diffusion과 달라지는 점은 두 가지뿐이다 (네트워크 폭·깊이·time embedding·레시피는 같음).
  A. 조건 입력 제거 (InDI 방식, Delbracio & Milanfar 2023)
       원고:  D_θ(x_t, E_u, t)   입력 [x_t ‖ E_u ‖ τ(t)]  → x_t와 E_u로 E_f가 계산되는 지름길이 생김
       변형:  D_θ(x_t, t)        입력 [x_t ‖ τ(t)]
     조건이 없으면 작은 t에서 Δ는 알 수 없으므로, 오차를 E_f 공간에서 잰다(InDI의 clean 예측과 같음):
       Ê_f = x_t − λ_t·Δ̂,   loss = MSE(Ê_f, E_f)   (t = T−1에서는 direct와 똑같은 목적함수)
  C. λ 끝점 맞추기
       원고:  λ_t = t/T      → 학습 마지막 단계에서도 E_f가 1% 섞임
       변형:  λ_t = t/(T−1)  → t = T−1에서 x = E_u 정확히 (추론 입력과 같음). t는 1..T−1에서 고르게 뽑음
  샘플링 (S단계): x ← E_u (t = T−1)에서 시작, 각 단계에서 Ê_f = x − λ_t·Δ̂, 다음 단계 x ← Ê_f + λ_next·Δ̂.
       E_u는 시작점으로만 쓰고 다시 넣지 않는다. 1단계면 E_r = E_u − D_θ(E_u, T−1).

선택 규칙(미리 정함): 에폭은 val 1단계 복원 MSE 최소(EMA 가중치, 다른 방법과 같은 기준).
  그 체크포인트로 단계 수 S ∈ {1, 5, 10, 20}을 val 복원 MSE로 하나 고르고(S*), test는 네 개 모두 보고한다.

출력: 저장소의 compare/ (compare_restorers.py와 같은 형식) → collect_restorers.py가 그대로 읽는다.
  results/{tag}__diffusion_nc__s42.json, ..._nc_5step / _nc_10step / _nc_20step
  preds/{tag}__diffusion_nc*__s42_test.npz,  weights/{tag}__diffusion_nc__s42.pt,  logs/nocond_{tag}_{stamp}.log

실행 위치: 서버 (GPU). 이 폴더(exp_nocond/)를 ~/hayeon/ich_cf_clean/ 안에 그대로 두고 **저장소 루트에서** 실행.
학습 1회 200 에폭, 대략 30분–1시간 (+ 데이터 로딩).
Usage (서버):
  conda activate hayeon
  cd ~/hayeon/ich_cf_clean
  python exp_nocond/train_nocond.py configs/medsiglip_after/train_medsiglip_after_nv6.yaml --seed 42
  python collect_restorers.py --B 2000 --ref mlp        # MLP 대비 Δ와 95% CI
  # 동작 확인만: python exp_nocond/train_nocond.py configs/medsiglip_after/train_medsiglip_after_nv6.yaml --epochs 2 --out_dir compare_smoke
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # 저장소 루트 (ich_cf_clean/)
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)                                          # weights/linprobe_*, compare/ 경로를 루트 기준으로

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from data import CTFeatureMapDataset
from diffusion import EMA
from diffusion.model import SinusoidalTimeEmbedding, SwiGLUBlock
from probe import LinearProbe
from train_diffusion import load_to_ram, get_probs, set_seed
from utils import get_config
from compare_restorers import (auroc_per_label, cos_per_slice, mse, evaluate, save_result, done,
                               LABELS_DEFAULT)

logger = logging.getLogger("nocond")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _sh = logging.StreamHandler()
    _sh.setFormatter(logging.Formatter("%(asctime)s - %(message)s", "%m/%d/%Y %H:%M:%S"))
    logger.addHandler(_sh)

STEPS = (1, 5, 10, 20)
NAME = {1: "diffusion_nc", 5: "diffusion_nc_5step", 10: "diffusion_nc_10step", 20: "diffusion_nc_20step"}


class NoCondDiffusionMLP(nn.Module):
    """원고 ColdDiffusionMLP와 같은 블록·폭·time embedding, 입력에서 E_u만 뺌 (A). λ_t = t/(T−1) (C)."""

    def __init__(self, emb_dim, hidden=1024, depth=4, T=100, time_dim=256):
        super().__init__()
        self.T = T
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        layers = [SwiGLUBlock(emb_dim + time_dim, hidden, hidden)]
        for _ in range(depth - 1):
            layers.append(SwiGLUBlock(hidden, hidden, hidden))
        self.blocks = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, emb_dim)

    def lam(self, t):
        return (t.float() / (self.T - 1)).view(-1, 1)

    def degrade(self, e_f, e_u, t):
        lam = self.lam(t)
        return (1.0 - lam) * e_f + lam * e_u

    def forward(self, x_t, t):                          # → Δ̂ (= E_u − E_f 추정)
        return self.out(self.blocks(torch.cat([x_t, self.time_emb(t)], dim=-1)))

    @torch.no_grad()
    def sample(self, e_u, steps=1):
        x, B = e_u.clone(), e_u.size(0)
        ts = torch.linspace(self.T - 1, 0, steps + 1).long().tolist()
        for i in range(steps):
            t = torch.full((B,), ts[i], device=x.device, dtype=torch.long)
            d = self(x, t)
            ef_hat = x - (ts[i] / (self.T - 1)) * d
            x = ef_hat + (ts[i + 1] / (self.T - 1)) * d if ts[i + 1] > 0 else ef_hat
        return x


@torch.no_grad()
def restore(model, E, device, steps=1, chunk=8192):
    model.eval()
    return torch.cat([model.sample(E[i:i + chunk].to(device).float(), steps).float().cpu()
                      for i in range(0, E.size(0), chunk)])


def train(tr_f, tr_u, va_f, va_u, emb_dim, hp, seed, device, use_amp):
    set_seed(seed)
    model = NoCondDiffusionMLP(emb_dim, hp["hidden"], hp["depth"], hp["T"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    ema = EMA(model, decay=hp["ema"])
    optim = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    scaler = GradScaler("cuda") if use_amp else None
    T, bs, N = hp["T"], hp["bs"], tr_f.size(0)

    def loss_fn(ef, eu):
        t = torch.randint(1, T, (ef.size(0),), device=device)            # 1..T−1 (t = 0은 손실 0)
        x = model.degrade(ef, eu, t)
        return F.mse_loss(x - model.lam(t) * model(x, t), ef)             # E_f 공간 오차 (InDI)

    curves = {"train": [], "val_restore_ema": []}
    best = (float("inf"), 0, None)
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
        orig = {k: v.clone() for k, v in model.state_dict().items()}
        ema.apply_shadow()
        vr = mse(restore(model, va_u, device, 1), va_f)
        flag = ""
        if vr < best[0]:
            best = (vr, ep, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            flag = "best(val_restore)"
        ema.restore(orig)
        curves["val_restore_ema"].append(vr)
        if ep == 1 or ep % 10 == 0 or ep == hp["epochs"]:
            logger.info(f"  nocond s{seed} ep {ep:3d}/{hp['epochs']}  train={curves['train'][-1]:.6f}  "
                        f"val_restore(EMA, 1-step)={vr:.6f}  {flag}  ({time.time() - t0:.0f}s)")
    return model, best, curves, n_params, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="compare")
    ap.add_argument("--epochs", type=int, default=None, help="동작 확인용 (기본 config의 200)")
    ap.add_argument("--study_col", default="study_id")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--no_amp", action="store_true")
    a = ap.parse_args()

    opt = get_config(a.config)
    tag, pooling = opt.tag, getattr(opt, "pooling", "none")
    label_cols = getattr(opt, "label_cols", LABELS_DEFAULT)
    out = Path(a.out_dir)
    for sub in ("results", "preds", "weights", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    if all(done(out, tag, NAME[s], a.seed) for s in STEPS):
        logger.info(f"---- skip {tag} (결과 있음: {out}/results/{tag}__diffusion_nc*)")
        return
    fh = logging.FileHandler(out / "logs" / f"nocond_{tag}_{datetime.now():%Y%m%d_%H%M%S}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s", "%m/%d/%Y %H:%M:%S"))
    logger.addHandler(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not a.no_amp
    hp = dict(T=int(getattr(opt, "diffusion_T", 100)), hidden=int(getattr(opt, "diffusion_hidden", 1024)),
              depth=int(getattr(opt, "diffusion_depth", 4)), epochs=int(a.epochs or getattr(opt, "diffusion_epochs", 200)),
              lr=float(getattr(opt, "diffusion_lr", 1e-4)), wd=float(getattr(opt, "diffusion_wd", 1e-4)),
              bs=int(getattr(opt, "diffusion_batch", 256)), ema=float(getattr(opt, "diffusion_ema_decay", 0.999)))
    logger.info(f"==== {tag}  A+C (조건 입력 제거 + λ=t/(T−1))  seed={a.seed}  device={device}  amp={use_amp}\n     hp={hp}")

    import pandas as pd
    df = pd.read_csv(opt.csv_path)
    ds = {s: CTFeatureMapDataset(df, opt.full_feature_dir, opt.under_feature_dir, label_cols, s, pooling)
          for s in ("train", "val", "test")}
    data = {}
    for s in ("train", "val", "test"):
        logger.info(f"Loading {s} ({len(ds[s])} slices)...")
        data[s] = load_to_ram(ds[s], num_workers=a.num_workers)
    (tr_f, tr_u, _), (va_f, va_u, _), (te_f, te_u, te_Y) = data["train"], data["val"], data["test"]
    emb_dim = tr_f.size(1)
    if device.type == "cuda" and 2 * tr_f.numel() * 2 < 0.6 * torch.cuda.mem_get_info()[0]:
        tr_f, tr_u = tr_f.half().to(device), tr_u.half().to(device)
        logger.info("train data on GPU (fp16)")

    probe_ckpt = f"weights/linprobe_{tag}_{pooling}.best.pt"                 # 원고·compare와 같은 고정 프로브
    probe = LinearProbe(emb_dim, len(label_cols)).to(device)
    probe.load_state_dict(torch.load(probe_ckpt, map_location=device, weights_only=True))
    probe.eval()
    y_test = te_Y.numpy().astype(np.uint8)
    p_full = get_probs(probe, te_f, device).astype(np.float32)
    p_under = get_probs(probe, te_u, device).astype(np.float32)
    per_f, per_u = auroc_per_label(p_full, y_test, label_cols), auroc_per_label(p_under, y_test, label_cols)
    cos_u = cos_per_slice(te_f, te_u).numpy().astype(np.float32)
    ctx = dict(tag=tag, out=out, probe=probe, device=device, label_cols=label_cols, y_test=y_test, te_f=te_f,
               va_f=va_f, cos_u=cos_u, per_f=per_f, per_u=per_u,
               auroc_f=float(np.nanmean(list(per_f.values()))), auroc_u=float(np.nanmean(list(per_u.values()))))
    logger.info(f"probe={probe_ckpt}  AUROC E_f={ctx['auroc_f']:.4f}  E_u={ctx['auroc_u']:.4f}")
    ref = out / "preds" / f"{tag}__ref_test.npz"
    if not ref.exists():
        np.savez_compressed(ref, slice_id=np.asarray(ds["test"].df["slice_id"].astype(str), dtype=str),
                            study_id=np.asarray(ds["test"].df[a.study_col].astype(str), dtype=str),
                            label_cols=np.array(label_cols), y=y_test, p_full=p_full, p_under=p_under, cos_u=cos_u)

    model, (bv, bep, bsd), curves, n_params, secs = train(tr_f, tr_u, va_f, va_u, emb_dim, hp, a.seed, device, use_amp)
    model.load_state_dict(bsd)
    torch.save({k: v.half() for k, v in bsd.items()}, out / "weights" / f"{tag}__diffusion_nc__s{a.seed}.pt")
    val_mse = {s: mse(restore(model, va_u, device, s), va_f) for s in STEPS}
    s_star = min(val_mse, key=val_mse.get)
    logger.info(f"best epoch={bep} (val 1-step MSE={bv:.6f})  val MSE by steps: "
                + "  ".join(f"{s}:{v:.6f}" for s, v in val_mse.items()) + f"  → S*={s_star}")
    for s in STEPS:
        res, p_rest, cos_r = evaluate(restore(model, te_u, device, s), restore(model, va_u, device, s), ctx)
        save_result(ctx, NAME[s], a.seed, res, p_rest, cos_r,
                    dict(encoder_tag=tag, config=a.config, emb_dim=emb_dim, train_N=int(tr_f.size(0)), hp=hp,
                         amp=use_amp, variant="A+C: no E_u condition, lambda=t/(T-1), E_f-space loss",
                         selection="val_restoration_mse (1-step) for epoch; steps reported for all",
                         steps=s, steps_selected_on_val=s_star, is_selected=(s == s_star), val_mse_by_steps=val_mse,
                         best_epoch=bep, best_value=bv, n_params=n_params, seconds=secs, curves=curves))
    logger.info(f"==== done {tag}  (val로 고른 단계 수 S* = {s_star} → 판정에는 {NAME[s_star]} 사용)")
    logger.removeHandler(fh)
    fh.close()


if __name__ == "__main__":
    main()
