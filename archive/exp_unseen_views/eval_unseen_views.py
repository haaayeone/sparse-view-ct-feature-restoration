"""
exp_unseen_views/eval_unseen_views.py — 학습에 쓰지 않은 view 수에서 복원기 평가 (교수님 제안 · 미팅 할 일 3).

질문: 45·90-view 등으로 학습한 복원기를 처음 보는 30·70-view test 영상에 그대로 쓰면 어떻게 되나?
  diffusion은 열화 정도(t)를 연속으로 배우므로 한 열화 수준만 배운 MLP보다 처음 보는 view 수에 강할 수 있다
  → diffusion을 쓸 남은 근거. 여기서도 MLP ≥ diffusion이면 그 근거도 없어진다. (판정 기준은 README에 미리 고정)

학습은 하지 않는다.
  - 학습 view s의 가중치 compare/weights/{tag_s}__{방법}__s{seed}.pt 를 불러와 test view v의 E_u에 1단계로 적용
    (compare_restorers.py와 같은 복원 식).
  - 프로브는 학습 view s의 고정 프로브 weights/linprobe_{tag_s}_none.best.pt (E_f로 학습, 원고와 같음).
  - R = (AUROC_r − AUROC_u) / (AUROC_f − AUROC_u), AUROC_u는 test view v의 E_u.
  - ridge는 가중치를 저장하지 않으므로 학습 view s의 train/val로 다시 맞춘다 (닫힌해, 데이터 로딩 포함 약 10분).
  방법: identity(= E_u), ridge, mlp, direct, diffusion, diffusion_vl, diffusion_nc(A+C, 가중치가 있을 때만)
  가중치 파일이 없는 방법은 건너뛰고 로그에 적는다.

필요한 것
  1) test view feature: exp_unseen_views/extract_test_views.py 로 추출 (…/feature_nv30/test/*.npy)
  2) 학습 view 가중치: compare_restorers.py … --save_weights 로 만든 compare/weights/*.pt
     45-view는 아직 compare를 안 돌렸으므로 먼저:
       python compare_restorers.py configs/medsiglip_after/train_medsiglip_after_nv45.yaml --seeds 42 --save_weights

출력 (끊겨도 같은 명령으로 이어서 실행: 결과 json이 있는 조건은 건너뜀)
  compare_unseen/results/{tag_s}_to_nv{v}__{방법}__s{seed}.json   (닫힌해는 s0)
  compare_unseen/preds/{tag_s}_to_nv{v}__ref_test.npz, ..__{방법}__s{seed}_test.npz   (collect_restorers.py와 같은 형식)
  results_unseen_runs.csv        학습 view × test view × 방법 1행: AUROC, Δ, R, cos, test MSE, 같은 view(s→s)에서의 AUROC
  results_unseen_table.csv       test view × 방법 행, 학습 view 열: "AUROC (R%)"  ← 슬라이드용
  results_unseen_bootstrap.csv   (--B > 0) 검사 단위 bootstrap: AUROC 95% CI, Δ = AUROC(ref) − AUROC(방법), Holm

실행 위치: 서버 (GPU 권장, CPU도 가능). 이 폴더를 ~/hayeon/ich_cf_clean/ 안에 두고 실행 (경로는 저장소 루트 기준).
Usage (서버):
  conda activate hayeon
  cd ~/hayeon/ich_cf_clean
  python exp_unseen_views/eval_unseen_views.py --train_views 90 45 --test_views 70 30 --B 2000
  # 빠른 확인 (학습 없음, ridge 생략, test 500검사, 같은 view도 같은 표본으로 같이):
  python exp_unseen_views/eval_unseen_views.py --train_views 90 23 --test_views 90 70 30 23 --methods identity mlp direct diffusion diffusion_vl --test_csv /data0/hayeon/master_df_test_sub500.csv --B 2000
  python exp_unseen_views/eval_unseen_views.py --train_views 90 45 23 6 --test_views 70 30 --B 2000
  # bootstrap만 다시 (결과는 그대로):  python exp_unseen_views/eval_unseen_views.py --train_views 90 45 --test_views 70 30 --B 2000 --ref diffusion
결과 가져오기 (로컬 PowerShell):
  scp neumous-lab@100.73.97.101:/home/neumous-lab/hayeon/ich_cf_clean/results_unseen_*.csv "C:\\Users\\user\\Desktop\\MIGT\\01. sparse ct project\\result"
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

from data import CTFeatureMapDataset
from probe import LinearProbe
from train_diffusion import load_to_ram, get_probs
from utils import get_config
from compare_restorers import auroc_per_label, cos_per_slice, mse, fit_closed, build_model, restore, LABELS_DEFAULT
from collect_restorers import bootstrap_tag, write_csv, NAME as ENC_NAME
from bootstrap_study import ci, p_two_sided, holm

logger = logging.getLogger("unseen")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _sh = logging.StreamHandler()
    _sh.setFormatter(logging.Formatter("%(asctime)s - %(message)s", "%m/%d/%Y %H:%M:%S"))
    logger.addHandler(_sh)

LEARNED = ["mlp", "direct", "diffusion", "diffusion_vl", "diffusion_nc"]
CLOSED = ["identity", "ridge"]
ORDER = CLOSED + LEARNED


def train_cfg(enc, v):
    """scripts/compare_all.sh 의 cfg_path 와 같은 규칙"""
    if enc == "medsiglip_after":
        return f"configs/medsiglip_after/train_medsiglip_after_nv{v}.yaml"
    return f"configs/{enc}/train_{enc}_window_nv{v}.yaml"


def swap_view(path, s, v):
    if v == s:                                           # 같은 view (s → s): 같은 검사 표본에서 비교 기준
        return path
    new = path.replace(f"feature_nv{s}", f"feature_nv{v}")
    if new == path:
        raise SystemExit(f"!! under_feature_dir에 'feature_nv{s}'가 없어 test view 경로를 못 만듦: {path}")
    return new


def load_learned(m, path, emb_dim, hp, device):
    """→ E ↦ E_r 함수 (1단계 복원, compare_restorers / train_nocond와 같은 식)"""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if m == "diffusion_nc":
        from exp_nocond.train_nocond import NoCondDiffusionMLP, restore as nc_restore
        model = NoCondDiffusionMLP(emb_dim, hp["hidden"], hp["depth"], hp["T"]).to(device)
        model.load_state_dict(sd)
        return lambda E: nc_restore(model, E, device, steps=1)
    base = "diffusion" if m == "diffusion_vl" else m
    model = build_model(base, emb_dim, hp).to(device)
    model.load_state_dict(sd)
    return lambda E: restore(model, base, E, hp["T"], device)


def result_path(out, key):
    return out / "results" / f"{key}.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="medsiglip_after", help="medsiglip_after siglip dinov2 resnet50 densenet121")
    ap.add_argument("--train_views", nargs="+", type=int, default=[90, 45])
    ap.add_argument("--test_views", nargs="+", type=int, default=[70, 30])
    ap.add_argument("--methods", nargs="+", default=ORDER, choices=ORDER)
    ap.add_argument("--seed", type=int, default=42, help="학습 seed (가중치 파일 이름의 s42)")
    ap.add_argument("--weights_dir", default="compare/weights")
    ap.add_argument("--same_view_dir", default="compare/results", help="같은 view(s→s) 결과 json (비교용 열)")
    ap.add_argument("--out_dir", default="compare_unseen")
    ap.add_argument("--csv_dir", default=".", help="results_unseen_*.csv 저장 위치")
    ap.add_argument("--B", type=int, default=0, help="bootstrap 반복 수 (0이면 생략, 원고와 같게 2000)")
    ap.add_argument("--ref", default="mlp", help="bootstrap 비교 기준 방법")
    ap.add_argument("--study_col", default="study_id")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--test_csv", default=None,
                    help="빠른 확인용: 이 csv의 test 슬라이스만 평가 (extract_test_views.py --n_studies N 이 만든 "
                         "master_df_test_sub{N}.csv). 결과는 compare_unseen_sub{N}/, results_unseen_sub{N}_*.csv 로 따로 저장")
    a = ap.parse_args()

    a.suffix = ""
    if a.test_csv:
        a.keep = set(pd.read_csv(a.test_csv, usecols=["slice_id"])["slice_id"].astype(str))
        stem = Path(a.test_csv).stem
        a.suffix = "_" + (stem.split("_test_")[-1] if "_test_" in stem else stem)      # master_df_test_sub500 → _sub500
        if a.out_dir == "compare_unseen":
            a.out_dir += a.suffix
    out = Path(a.out_dir)
    for sub in ("results", "preds", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out / "logs" / f"unseen_{datetime.now():%Y%m%d_%H%M%S}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s", "%m/%d/%Y %H:%M:%S"))
    logger.addHandler(fh)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"==== unseen views  encoder={a.encoder}  train={a.train_views}  test={a.test_views}  "
                f"methods={a.methods}  seed={a.seed}  device={device}")

    cache = {}                                                       # feature 경로 → 텐서 (test는 view마다 한 번만 읽음)

    def load_split(full_dir, under_dir, split, df, label_cols):
        key = (full_dir, under_dir, split)
        if key not in cache:
            ds = CTFeatureMapDataset(df, full_dir, under_dir, label_cols, split, "none")
            logger.info(f"Loading {split} ({len(ds)} slices)  E_u: {under_dir}")
            cache[key] = (ds, load_to_ram(ds, num_workers=a.num_workers))
        return cache[key]

    # 할 일 정리 + 입력 확인 (오래 돌기 전에 빠진 파일을 먼저 알려 줌)
    plan, missing = [], []
    for s in a.train_views:
        cfg = train_cfg(a.encoder, s)
        if not os.path.exists(cfg):
            raise SystemExit(f"!! config 없음: {cfg}")
        opt = get_config(cfg)
        probe = f"weights/linprobe_{opt.tag}_{getattr(opt, 'pooling', 'none')}.best.pt"
        if not os.path.exists(probe):
            raise SystemExit(f"!! 프로브 없음: {probe}")
        for v in a.test_views:
            d = Path(swap_view(opt.under_feature_dir, s, v)) / "test"
            if not d.is_dir() or not any(d.glob("*.npy")):
                raise SystemExit(f"!! test feature 없음: {d}  (먼저 exp_unseen_views/extract_test_views.py --views {v})")
        meths = []
        for m in a.methods:
            if m in LEARNED:
                w = Path(a.weights_dir) / f"{opt.tag}__{m}__s{a.seed}.pt"
                if not w.exists():
                    missing.append(str(w))
                    continue
            meths.append(m)
        plan.append((s, cfg, opt, probe, meths))
    for w in missing:
        logger.info(f"   가중치 없음 → 건너뜀: {w}")
    if any("__mlp__" in w or "__diffusion__" in w for w in missing):
        logger.info("   !! mlp·diffusion 가중치가 없는 학습 view는 판정을 못 함 → README의 '가중치 만들기' 참고")

    for s, cfg, opt, probe_ckpt, meths in plan:
        tag, label_cols = opt.tag, getattr(opt, "label_cols", LABELS_DEFAULT)
        todo = [(v, m) for v in a.test_views for m in meths
                if not result_path(out, f"{tag}_to_nv{v}__{m}__s{0 if m in CLOSED else a.seed}").exists()]
        if not todo:
            logger.info(f"---- skip {tag} (결과 있음)")
            continue
        logger.info(f"\n---- 학습 view {s}  ({cfg})  할 일 {len(todo)}개")
        df = pd.read_csv(opt.csv_path)
        if a.test_csv:                                   # test는 고른 검사만 (train/val은 그대로: ridge용)
            df = df[(df["split"] != "test") | df["slice_id"].astype(str).isin(a.keep)].reset_index(drop=True)
            logger.info(f"  test 검사 일부만: {a.test_csv}  ({int((df['split'] == 'test').sum())} 슬라이스)")
        hp = dict(T=int(getattr(opt, "diffusion_T", 100)), hidden=int(getattr(opt, "diffusion_hidden", 1024)),
                  depth=int(getattr(opt, "diffusion_depth", 4)))

        restorers = {"identity": lambda E: E.float().cpu().clone()}
        if "ridge" in meths and any(m == "ridge" for _, m in todo):
            t0 = time.time()
            _, (tr_f, tr_u, _) = load_split(opt.full_feature_dir, opt.under_feature_dir, "train", df, label_cols)
            _, (va_f, va_u, _) = load_split(opt.full_feature_dir, opt.under_feature_dir, "val", df, label_cols)
            restorers["ridge"], info = fit_closed("ridge", tr_f, tr_u, va_f, va_u, device)
            logger.info(f"  ridge (학습 view {s}) λ={info['lambda']:g}  ({time.time() - t0:.0f}s)")
            for k in [k for k in cache if k[2] in ("train", "val")]:  # 학습 데이터는 바로 비움 (메모리)
                del cache[k]
        emb_dim = None

        for v in a.test_views:
            under_v = swap_view(opt.under_feature_dir, s, v)
            ds_te, (te_f, te_u, te_Y) = load_split(opt.full_feature_dir, under_v, "test", df, label_cols)
            if emb_dim is None:
                emb_dim = te_f.size(1)
                probe = LinearProbe(emb_dim, len(label_cols)).to(device)
                probe.load_state_dict(torch.load(probe_ckpt, map_location=device, weights_only=True))
                probe.eval()
            tag_x = f"{tag}_to_nv{v}"
            y = te_Y.numpy().astype(np.uint8)
            p_full = get_probs(probe, te_f, device).astype(np.float32)
            p_under = get_probs(probe, te_u, device).astype(np.float32)
            per_f, per_u = auroc_per_label(p_full, y, label_cols), auroc_per_label(p_under, y, label_cols)
            a_f, a_u = float(np.nanmean(list(per_f.values()))), float(np.nanmean(list(per_u.values())))
            cos_u = cos_per_slice(te_f, te_u).numpy().astype(np.float32)
            ref = out / "preds" / f"{tag_x}__ref_test.npz"
            if not ref.exists():
                np.savez_compressed(ref, slice_id=np.asarray(ds_te.df["slice_id"].astype(str), dtype=str),
                                    study_id=np.asarray(ds_te.df[a.study_col].astype(str), dtype=str),
                                    label_cols=np.array(label_cols), y=y, p_full=p_full, p_under=p_under, cos_u=cos_u)
            logger.info(f"  [{s}→{v}] AUROC E_f={a_f:.4f}  E_u(nv{v})={a_u:.4f}  cos(f,u)={cos_u.mean():.4f}")

            for m in meths:
                sd = 0 if m in CLOSED else a.seed
                key = f"{tag_x}__{m}__s{sd}"
                if result_path(out, key).exists():
                    continue
                if m not in restorers:
                    restorers[m] = load_learned(m, Path(a.weights_dir) / f"{tag}__{m}__s{a.seed}.pt", emb_dim, hp,
                                                device)
                er = restorers[m](te_u)
                p_rest = get_probs(probe, er, device).astype(np.float32)
                per = auroc_per_label(p_rest, y, label_cols)
                a_r = float(np.nanmean(list(per.values())))
                cos_r = cos_per_slice(te_f, er).numpy().astype(np.float32)
                same = Path(a.same_view_dir) / f"{tag}__{m}__s{sd}.json"
                a_same = json.load(open(same, encoding="utf-8"))["auroc_r"] if same.exists() else None
                rec = dict(tag=tag_x, method=m, seed=sd, encoder_tag=tag, train_view=s, test_view=v,
                           auroc_f=a_f, auroc_u=a_u, auroc_r=a_r, delta=a_r - a_u,
                           R_pct=100 * (a_r - a_u) / (a_f - a_u) if a_f > a_u else float("nan"),
                           cos_u=float(cos_u.mean()), cos_r=float(cos_r.mean()), test_restore_mse=mse(er, te_f),
                           val_restore_mse=float("nan"), per_label_r=per, per_label_f=per_f, per_label_u=per_u,
                           auroc_r_same_view=a_same, config=cfg, probe=probe_ckpt,
                           weights=None if m in CLOSED else f"{a.weights_dir}/{tag}__{m}__s{a.seed}.pt",
                           created=datetime.now().isoformat())
                np.savez_compressed(out / "preds" / f"{key}_test.npz", p_rest=p_rest, cos_r=cos_r)
                tmp = result_path(out, key).with_suffix(".json.tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(rec, f, indent=1)
                os.replace(tmp, result_path(out, key))
                logger.info(f"    {m:<13} AUROC E_r={a_r:.4f}  Δ={a_r - a_u:+.4f}  R={rec['R_pct']:5.1f}%  "
                            f"cos={rec['cos_r']:.4f}  test MSE={rec['test_restore_mse']:.6f}"
                            + (f"   (같은 view {s}→{s}: {a_same:.4f})" if a_same is not None else ""))
        del restorers
        if device.type == "cuda":
            torch.cuda.empty_cache()

    collect(a, out)
    logger.removeHandler(fh)
    fh.close()


def collect(a, out):
    """compare_unseen/results 전체 → results_unseen_runs.csv · _table.csv · (B>0) _bootstrap.csv"""
    enc_name = ENC_NAME.get(f"{a.encoder}_r360", a.encoder) if a.encoder == "medsiglip_after" else \
        ENC_NAME.get(f"{a.encoder}_window_r360", a.encoder)
    runs = []
    for f in sorted((out / "results").glob("*.json")):
        r = json.load(open(f, encoding="utf-8"))
        if not r["encoder_tag"].startswith(a.encoder):
            continue
        row = dict(encoder=enc_name, train_view=r["train_view"], test_view=r["test_view"], method=r["method"],
                   seed=r["seed"], tag=r["tag"], auroc_f=round(r["auroc_f"], 4), auroc_u=round(r["auroc_u"], 4),
                   auroc_r=round(r["auroc_r"], 4), delta=round(r["delta"], 4), R_pct=round(r["R_pct"], 1),
                   cos_u=round(r["cos_u"], 4), cos_r=round(r["cos_r"], 4), test_mse=r["test_restore_mse"],
                   auroc_r_same_view=(round(r["auroc_r_same_view"], 4) if r["auroc_r_same_view"] is not None else ""))
        for lab, val in r["per_label_r"].items():
            row[f"r_{lab}"] = round(val, 4)
        runs.append(row)
    if not runs:
        logger.info("결과 없음")
        return
    om = {m: i for i, m in enumerate(ORDER)}
    runs.sort(key=lambda r: (-r["test_view"], -r["train_view"], om.get(r["method"], 99)))
    write_csv(os.path.join(a.csv_dir, f"results_unseen{a.suffix}_runs.csv"), runs)

    # 슬라이드용 표: 행 = (test view, 방법), 열 = 학습 view
    tvs = sorted({r["train_view"] for r in runs}, reverse=True)
    table, idx = [], {}
    for r in runs:                                       # E_f 행 (프로브는 학습 view마다 따로라 열마다 적음)
        k = (r["test_view"], "E_f")
        if k not in idx:
            idx[k] = dict(test_view=r["test_view"], method="E_f", **{f"train_nv{s}": "" for s in tvs})
            table.append(idx[k])
        idx[k][f"train_nv{r['train_view']}"] = f"{r['auroc_f']:.3f}"
    for r in runs:                                       # identity 행 = E_u (test view의 열화 feature 그대로)
        k = (r["test_view"], r["method"])
        if k not in idx:
            idx[k] = dict(test_view=r["test_view"], method=r["method"], **{f"train_nv{s}": "" for s in tvs})
            table.append(idx[k])
        idx[k][f"train_nv{r['train_view']}"] = f"{r['auroc_r']:.3f} ({r['R_pct']:.1f}%)"
    table.sort(key=lambda t: (-t["test_view"], -1 if t["method"] == "E_f" else om.get(t["method"], 99)))
    write_csv(os.path.join(a.csv_dir, f"results_unseen{a.suffix}_table.csv"), table)
    print("\nAUROC (R%) — 열: 학습 view, 행: test view × 방법   (identity = 그 test view의 E_u, R은 E_u 기준)")
    print(f"{'test':>5} {'method':<14}" + "".join(f"{'train ' + str(s):>17}" for s in tvs))
    for t in table:
        print(f"{t['test_view']:>5} {t['method']:<14}" + "".join(f"{t[f'train_nv{s}']:>17}" for s in tvs))

    if a.B <= 0:
        return
    by_tag = {}
    for r in runs:
        by_tag.setdefault(r["tag"], {})[r["method"]] = [r["seed"]]
    rows = []
    for tag_x, methods in by_tag.items():
        if not (out / "preds" / f"{tag_x}__ref_test.npz").exists():
            continue
        t0 = time.time()
        res, S = bootstrap_tag(str(out), tag_x, methods, a.B, 42, 100)
        s, v = next((r["train_view"], r["test_view"]) for r in runs if r["tag"] == tag_x)
        refr = res.get(a.ref)
        for m, (pt, bt) in res.items():
            lo, hi = ci(bt)
            row = dict(encoder=enc_name, train_view=s, test_view=v, method=m, n_studies=S, auroc=round(pt, 4),
                       auroc_lo=round(lo, 4), auroc_hi=round(hi, 4), ref=a.ref)
            if refr is not None and m not in (a.ref, "E_f", "E_u"):
                dlt, db = refr[0] - pt, refr[1] - bt
                se, p_z, _ = p_two_sided(dlt, db)
                if se == 0:
                    p_z = 1.0 if dlt == 0 else 0.0
                dlo, dhi = ci(db)
                row.update(delta_ref_minus=round(dlt, 4), delta_lo=round(dlo, 4), delta_hi=round(dhi, 4),
                           delta_se=round(se, 5), p_z=p_z)
            rows.append(row)
        print(f"bootstrap {tag_x}: {len(res)} representations, B={a.B} ({time.time() - t0:.0f}s)")
    tested = [r for r in rows if "p_z" in r]
    for r, p in zip(tested, holm([r["p_z"] for r in tested])):
        r["p_holm"] = p
    om2 = {m: i for i, m in enumerate(["E_f", "E_u"] + ORDER)}
    rows.sort(key=lambda r: (-r["test_view"], -r["train_view"], om2.get(r["method"], 99)))
    write_csv(os.path.join(a.csv_dir, f"results_unseen{a.suffix}_bootstrap.csv"), rows)
    print(f"\nΔ = AUROC({a.ref}) − AUROC(방법), 검사 단위 bootstrap 95% CI, Holm 보정 ({len(tested)}개 비교)")
    for r in tested:
        print(f"  {r['train_view']:>3}→{r['test_view']:<3} {r['method']:<13} Δ={r['delta_ref_minus']:+.4f} "
              f"[{r['delta_lo']:+.4f}, {r['delta_hi']:+.4f}]  p_holm={r['p_holm']:.3g}")


if __name__ == "__main__":
    main()
