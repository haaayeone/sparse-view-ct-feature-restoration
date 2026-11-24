"""
exp_probe/train_probe_variants.py — 프로브를 무엇으로 학습하느냐에 따른 비교 (E_f / E_u / E_r).

질문
  1) E_u로 프로브를 직접 학습하면? → E_u 안의 정보를 선형 분류기가 바로 읽을 수 있는지
     (E_f 프로브로 E_u를 넣으면 0.58이었던 것이 "정보가 없어서"인지 "형태가 달라서"인지 구분)
  2) 복원한 E_r로 프로브를 학습하면? (10/28 미팅 할 일 1)
원고 프로브(P_f, E_f로 학습)와 똑같은 선형 프로브·학습 레시피를 쓴다 (train_diffusion.train_stage2와 같은 루프:
  AdamW 1e-4/1e-4, 배치 256, 최대 200 에폭, 20 에폭마다 val AUROC로 저장, 40 에폭 이후 개선 없으면 멈춤).
  선택은 각자 자기 표현의 val AUROC (P_u는 val E_u, P_r은 val E_r).
원고 프로브 weights/linprobe_*.best.pt 는 읽기만 하고 덮어쓰지 않는다. 새 프로브는 compare_probe/weights/ 에 저장.

표현 (train/val/test 모두 같은 식으로 만듦)
  f = E_f,  u = E_u,  r_mlp = MLP로 복원한 E_r,  r_diffusion = diffusion(복원 MSE 선택)으로 복원한 E_r
  복원기 가중치: compare/weights/{tag}__mlp__s42.pt, ..__diffusion__s42.pt (compare_restorers.py --save_weights)
프로브: P_f(기존), P_u, P_r_mlp, P_r_diffusion  →  모든 프로브 × 모든 test 표현을 평가 (행렬)
bootstrap 기준(--ref): Pf_on_r_mlp = "E_f 프로브 + MLP 복원" (지금 논문 방향의 파이프라인)
  검정(Holm)은 주요 조건만: 대각선(같은 표현으로 학습·평가)과 P_f 행. 나머지 교차 조건은 행렬 csv에만 적는다.

출력
  compare_probe/results/{tag}__P{학습}_on_{test}__s{seed}.json, preds/..._test.npz (collect_restorers.py와 같은 형식)
  compare_probe/weights/linprobe_{tag}__P{학습}__s{seed}.pt
  results_probe_matrix.csv      view × 프로브(학습 표현) 행, test 표현 열: 평균 AUROC
  results_probe_bootstrap.csv   (--B > 0) AUROC 95% CI, Δ = AUROC(ref) − AUROC(조건), Holm

실행 위치: 서버 (GPU). 이 폴더를 ~/hayeon/ich_cf_clean/ 안에 두고 실행. view 하나에 약 20–30분 (데이터 로딩 약 10분 포함).
Usage (서버):
  conda activate hayeon
  cd ~/hayeon/ich_cf_clean
  python exp_probe/train_probe_variants.py --views 6 --B 2000
  python exp_probe/train_probe_variants.py --views 6 23 90 --B 2000
결과 가져오기 (로컬 PowerShell):
  scp neumous-lab@100.73.97.101:/home/neumous-lab/hayeon/ich_cf_clean/results_probe_*.csv "C:\\Users\\user\\Desktop\\MIGT\\01. sparse ct project\\result"
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # 저장소 루트 (ich_cf_clean/)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import CTFeatureMapDataset
from probe import LinearProbe
from train_diffusion import load_to_ram, get_probs, set_seed, compute_auroc
from utils import get_config
from compare_restorers import auroc_per_label, build_model, restore, LABELS_DEFAULT
from collect_restorers import bootstrap_tag, write_csv
from bootstrap_study import ci, p_two_sided, holm

logger = logging.getLogger("probe_variants")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _sh = logging.StreamHandler()
    _sh.setFormatter(logging.Formatter("%(asctime)s - %(message)s", "%m/%d/%Y %H:%M:%S"))
    logger.addHandler(_sh)

NICE = {"f": "E_f", "u": "E_u", "r_mlp": "E_r(MLP)", "r_diffusion": "E_r(diffusion)"}


def train_cfg(enc, v):
    if enc == "medsiglip_after":
        return f"configs/medsiglip_after/train_medsiglip_after_nv{v}.yaml"
    return f"configs/{enc}/train_{enc}_window_nv{v}.yaml"


def train_probe(X_tr, Y_tr, X_va, Y_va, opt, device, seed, ckpt):
    """train_diffusion.train_stage2 와 같은 루프 (저장 경로만 다름)"""
    set_seed(seed)
    L = Y_tr.size(1)
    probe = LinearProbe(X_tr.size(1), L).to(device)
    optim = torch.optim.AdamW(probe.parameters(), lr=float(getattr(opt, "lr", 1e-4)),
                              weight_decay=float(getattr(opt, "weight_decay", 1e-4)))
    epochs, bs = int(getattr(opt, "num_epochs", 200)), int(getattr(opt, "diffusion_batch", 256))
    patience = int(getattr(opt, "patience", 20))
    Xg, Yg = X_tr.to(device), Y_tr.to(device)              # 선형 프로브라 GPU에 올려도 결과는 같고 빠름
    N, best, wait, best_ep = Xg.size(0), 0.0, 0, 0
    for ep in range(1, epochs + 1):
        probe.train()
        perm, losses = torch.randperm(N, device=device), []
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            logits = probe(Xg[idx])
            loss = sum(F.binary_cross_entropy_with_logits(logits[c], Yg[idx][:, c:c + 1]) for c in range(L)) / L
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
        if ep % 20 == 0 or ep in (1, epochs):
            auc = compute_auroc(probe, X_va, Y_va, device)
            if auc > best:
                best, wait, best_ep = auc, 0, ep
                torch.save(probe.state_dict(), ckpt)
            else:
                wait += 1
            logger.info(f"    ep {ep:3d}  loss={np.mean(losses):.5f}  val_AUROC={auc:.4f}  best={best:.4f}")
            if wait >= patience // 20 and ep > 40:
                logger.info(f"    early stop at ep {ep}")
                break
    probe.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    probe.eval()
    del Xg, Yg
    return probe, best, best_ep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="medsiglip_after")
    ap.add_argument("--views", nargs="+", type=int, default=[6])
    ap.add_argument("--restorers", nargs="+", default=["mlp", "diffusion"], choices=["mlp", "diffusion", "direct"])
    ap.add_argument("--seed", type=int, default=42, help="복원기 가중치 seed이자 프로브 학습 seed")
    ap.add_argument("--weights_dir", default="compare/weights")
    ap.add_argument("--out_dir", default="compare_probe")
    ap.add_argument("--csv_dir", default=".")
    ap.add_argument("--B", type=int, default=0, help="bootstrap 반복 수 (0이면 생략, 원고와 같게 2000)")
    ap.add_argument("--ref", default="Pf_on_r_mlp", help="bootstrap 비교 기준 조건")
    ap.add_argument("--study_col", default="study_id")
    ap.add_argument("--num_workers", type=int, default=8)
    a = ap.parse_args()

    out = Path(a.out_dir)
    for sub in ("results", "preds", "weights", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out / "logs" / f"probe_{datetime.now():%Y%m%d_%H%M%S}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s", "%m/%d/%Y %H:%M:%S"))
    logger.addHandler(fh)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reps = ["f", "u"] + [f"r_{m}" for m in a.restorers]
    logger.info(f"==== probe variants  encoder={a.encoder}  views={a.views}  reps={reps}  seed={a.seed}  device={device}")

    tags = []
    for v in a.views:
        cfg = train_cfg(a.encoder, v)
        opt = get_config(cfg)
        tag, pooling = opt.tag, getattr(opt, "pooling", "none")
        label_cols = getattr(opt, "label_cols", LABELS_DEFAULT)
        tags.append(tag)
        pf_ckpt = f"weights/linprobe_{tag}_{pooling}.best.pt"
        missing = [p for p in [pf_ckpt] + [f"{a.weights_dir}/{tag}__{m}__s{a.seed}.pt" for m in a.restorers]
                   if not os.path.exists(p)]
        if missing:
            raise SystemExit("!! 파일 없음: " + ", ".join(missing))
        todo = [(p, t) for p in reps for t in reps
                if not (out / "results" / f"{tag}__P{p}_on_{t}__s{a.seed}.json").exists()]
        if not todo:
            logger.info(f"---- skip {tag} (결과 있음)")
            continue
        logger.info(f"\n---- {tag}  ({cfg})")
        df = pd.read_csv(opt.csv_path)
        ds = {s: CTFeatureMapDataset(df, opt.full_feature_dir, opt.under_feature_dir, label_cols, s, pooling)
              for s in ("train", "val", "test")}
        X, Y = {}, {}
        for s in ("train", "val", "test"):
            logger.info(f"Loading {s} ({len(ds[s])} slices)...")
            ef, eu, y = load_to_ram(ds[s], num_workers=a.num_workers)
            X[("f", s)], X[("u", s)], Y[s] = ef, eu, y
        emb_dim = X[("f", "train")].size(1)
        hp = dict(T=int(getattr(opt, "diffusion_T", 100)), hidden=int(getattr(opt, "diffusion_hidden", 1024)),
                  depth=int(getattr(opt, "diffusion_depth", 4)))
        for m in a.restorers:                                     # E_r = 복원기(E_u), train/val/test 모두
            model = build_model(m, emb_dim, hp).to(device)
            model.load_state_dict(torch.load(f"{a.weights_dir}/{tag}__{m}__s{a.seed}.pt", map_location="cpu",
                                             weights_only=True))
            for s in ("train", "val", "test"):
                X[(f"r_{m}", s)] = restore(model, m, X[("u", s)], hp["T"], device)
            logger.info(f"  E_r({m}) 만듦 (train/val/test)")
            del model

        probes = {}
        pf = LinearProbe(emb_dim, len(label_cols)).to(device)
        pf.load_state_dict(torch.load(pf_ckpt, map_location=device, weights_only=True))
        pf.eval()
        probes["f"] = (pf, None, None, pf_ckpt)
        for p in reps[1:]:
            ck = out / "weights" / f"linprobe_{tag}__P{p}__s{a.seed}.pt"
            if ck.exists():
                pr = LinearProbe(emb_dim, len(label_cols)).to(device)
                pr.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
                pr.eval()
                probes[p] = (pr, None, None, str(ck))
                continue
            t0 = time.time()
            logger.info(f"  프로브 P_{p} 학습 ({NICE[p]}로 학습, val {NICE[p]} AUROC로 선택)")
            pr, best, bep = train_probe(X[(p, "train")], Y["train"], X[(p, "val")], Y["val"], opt, device, a.seed, ck)
            probes[p] = (pr, best, bep, str(ck))
            logger.info(f"  → P_{p}: best val AUROC={best:.4f} (ep {bep}, {time.time() - t0:.0f}s)")

        y = Y["test"].numpy().astype(np.uint8)
        p_full = get_probs(pf, X[("f", "test")], device).astype(np.float32)
        p_under = get_probs(pf, X[("u", "test")], device).astype(np.float32)
        ref = out / "preds" / f"{tag}__ref_test.npz"
        if not ref.exists():
            np.savez_compressed(ref, slice_id=np.asarray(ds["test"].df["slice_id"].astype(str), dtype=str),
                                study_id=np.asarray(ds["test"].df[a.study_col].astype(str), dtype=str),
                                label_cols=np.array(label_cols), y=y, p_full=p_full, p_under=p_under)
        for p, t in todo:
            pr, best, bep, ck = probes[p]
            prob = get_probs(pr, X[(t, "test")], device).astype(np.float32)
            per = auroc_per_label(prob, y, label_cols)
            auc = float(np.nanmean(list(per.values())))
            key = f"{tag}__P{p}_on_{t}__s{a.seed}"
            np.savez_compressed(out / "preds" / f"{key}_test.npz", p_rest=prob)
            rec = dict(tag=tag, method=f"P{p}_on_{t}", seed=a.seed, view=v, probe_trained_on=p, tested_on=t,
                       auroc=auc, per_label=per, probe_ckpt=ck, probe_best_val_auroc=best, probe_best_epoch=bep,
                       config=cfg, created=datetime.now().isoformat())
            with open(out / "results" / f"{key}.json", "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=1)
            logger.info(f"    P_{p:<12} on {NICE[t]:<15} AUROC={auc:.4f}")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    collect(a, out, reps)
    logger.removeHandler(fh)
    fh.close()


def collect(a, out, reps):
    runs = [json.load(open(f, encoding="utf-8")) for f in sorted((out / "results").glob("*.json"))]
    runs = [r for r in runs if r["tag"].startswith(a.encoder) and r["seed"] == a.seed]
    if not runs:
        return
    views = sorted({r["view"] for r in runs})
    mat = []
    print("\n평균 AUROC — 행: 프로브(무엇으로 학습), 열: test에 넣은 표현   (대각선 = 같은 표현으로 학습·평가)")
    for v in views:
        print(f"-- {v}-view")
        print(f"   {'probe':<18}" + "".join(f"{NICE[t]:>17}" for t in reps))
        for p in reps:
            row = dict(view=v, probe=f"P_{p}", trained_on=NICE[p])
            vals = ""
            for t in reps:
                r = next((r for r in runs if r["view"] == v and r["probe_trained_on"] == p and r["tested_on"] == t), None)
                row[f"test_{t}"] = round(r["auroc"], 4) if r else ""
                vals += f"{r['auroc']:>17.4f}" if r else f"{'':>17}"
            mat.append(row)
            print(f"   P_{p:<16}" + vals)
    write_csv(os.path.join(a.csv_dir, "results_probe_matrix.csv"), mat)
    if a.B <= 0:
        return
    rows = []
    for tag in sorted({r["tag"] for r in runs}):
        methods = {r["method"]: [r["seed"]] for r in runs if r["tag"] == tag     # 검정은 주요 조건만:
                   and (r["probe_trained_on"] == r["tested_on"] or r["probe_trained_on"] == "f")}  # 대각선 + P_f
        t0 = time.time()
        res, S = bootstrap_tag(str(out), tag, methods, a.B, 42, 100)
        v = next(r["view"] for r in runs if r["tag"] == tag)
        refr = res.get(a.ref)
        for m, (pt, bt) in res.items():
            if m in ("E_f", "E_u"):                                  # = Pf_on_f, Pf_on_u 와 같음
                continue
            lo, hi = ci(bt)
            row = dict(view=v, condition=m, n_studies=S, auroc=round(pt, 4), auroc_lo=round(lo, 4),
                       auroc_hi=round(hi, 4), ref=a.ref)
            if refr is not None and m != a.ref:
                dlt, db = refr[0] - pt, refr[1] - bt
                se, p_z, _ = p_two_sided(dlt, db)
                if se == 0:
                    p_z = 1.0 if dlt == 0 else 0.0
                dlo, dhi = ci(db)
                row.update(delta_ref_minus=round(dlt, 4), delta_lo=round(dlo, 4), delta_hi=round(dhi, 4),
                           delta_se=round(se, 5), p_z=p_z)
            rows.append(row)
        print(f"bootstrap {tag}: B={a.B} ({time.time() - t0:.0f}s)")
    tested = [r for r in rows if "p_z" in r]
    for r, p in zip(tested, holm([r["p_z"] for r in tested])):
        r["p_holm"] = p
    rows.sort(key=lambda r: (-r["view"], r["condition"]))
    write_csv(os.path.join(a.csv_dir, "results_probe_bootstrap.csv"), rows)
    print(f"\nΔ = AUROC({a.ref}) − AUROC(조건), 검사 단위 bootstrap 95% CI, Holm ({len(tested)}개)")
    for r in tested:
        print(f"  {r['view']:>3}-view  {r['condition']:<26} AUROC={r['auroc']:.4f}  Δ={r['delta_ref_minus']:+.4f} "
              f"[{r['delta_lo']:+.4f}, {r['delta_hi']:+.4f}]  p_holm={r['p_holm']:.3g}")


if __name__ == "__main__":
    main()
