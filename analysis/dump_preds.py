"""
dump_preds.py — 학습이 끝난 조건마다 슬라이스별 예측 확률을 저장 (study 단위 bootstrap·검사 단위 평가용).

run_eval.py와 같은 방식으로 E_r을 복원하고(기본 1단계), 고정 linear probe의 sigmoid 확률을
E_f(기준) · E_u(희소) · E_r(복원) 세 표현에 대해 저장한다. 로그를 만들지 않으므로
collect_test_results.py 결과와 섞이지 않는다.

저장 (--out_dir, 기본 preds/):
  {tag}_{split}.npz
    slice_id (N,) str     study_id (N,) str     label_cols (6,) str
    y        (N,6) uint8  p_full / p_under / p_rest (N,6) float32
    cos_u / cos_r (N,) float32   슬라이스별 cos(E_f,E_u), cos(E_f,E_r)
  이미 있는 파일은 건너뜀 (--overwrite 로 다시 계산). 끝나면 평균 AUROC를 출력하므로
  results_test_summary.csv 의 E_f/E_u/E_r 과 같은지 확인할 것.

실행 위치: 서버 (GPU, 저장소 루트). 조건당 데이터 로딩 포함 수 분.
Usage (서버):
  conda activate hayeon
  cd ~/hayeon/ich_cf_clean
  # 30개 조건 전부 (5 인코더 × 180/90/45/23/12/6 view) — tmux 권장
  python analysis/dump_preds.py --all --split test
  # 일부만
  python analysis/dump_preds.py configs/medsiglip_after/train_medsiglip_after_nv6.yaml --split test
  python analysis/dump_preds.py --all --encoders siglip dinov2 --views 6 12
다음 단계:
  python analysis/bootstrap_study.py --pred_dir preds --B 2000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data import CTFeatureMapDataset
from diffusion import ColdDiffusionMLP
from probe import LinearProbe
from train_diffusion import load_to_ram, get_probs, set_seed
from utils import get_config

ENCODERS = ["medsiglip_after", "siglip", "dinov2", "resnet50", "densenet121"]
VIEWS = [180, 90, 45, 23, 12, 6]


def cfg_path(enc, v):   # scripts/train_all.sh 의 cfg_path 와 동일
    if enc == "medsiglip_after":
        return f"configs/medsiglip_after/train_medsiglip_after_nv{v}.yaml"
    return f"configs/{enc}/train_{enc}_window_nv{v}.yaml"


def mean_auroc(probs, y):
    from sklearn.metrics import roc_auc_score
    return float(np.mean([roc_auc_score(y[:, c], probs[:, c]) for c in range(y.shape[1])
                          if 0 < y[:, c].sum() < len(y)]))


def dump(cfg, split, out_dir, steps, study_col, overwrite, device):
    opt = get_config(cfg)
    set_seed(int(getattr(opt, "seed", 42)))
    tag = opt.tag
    pooling = getattr(opt, "pooling", "none")
    label_cols = getattr(opt, "label_cols", ["epidural", "intraparenchymal", "intraventricular",
                                             "subarachnoid", "subdural", "any"])
    out = out_dir / f"{tag}_{split}.npz"
    if out.exists() and not overwrite:
        print(f"---- skip {tag} ({out} 있음)")
        return

    T = int(getattr(opt, "diffusion_T", 100))
    depth = int(getattr(opt, "diffusion_depth", 4))
    diff_ckpt = Path(f"weights/diffusion_{tag}_{pooling}_T{T}.best.pt")
    probe_ckpt = Path(f"weights/linprobe_{tag}_{pooling}.best.pt")
    for p in (diff_ckpt, probe_ckpt):
        if not p.exists():
            print(f"!! ckpt 없음: {p}  → {tag} 건너뜀")
            return

    df = pd.read_csv(opt.csv_path)
    if study_col not in df.columns:
        raise SystemExit(f"{opt.csv_path}에 '{study_col}' 열이 없음 (--study_col 로 지정)")
    ds = CTFeatureMapDataset(df, opt.full_feature_dir, opt.under_feature_dir, label_cols, split, pooling)
    print(f"==== {tag}  split={split}  N={len(ds)}  loading...")
    e_full, e_under, Y = load_to_ram(ds)
    emb_dim = e_full.size(1)

    sd = torch.load(diff_ckpt, map_location="cpu", weights_only=True)
    diff = ColdDiffusionMLP(emb_dim=emb_dim, hidden=sd["out.weight"].shape[1], depth=depth, T=T).to(device)
    diff.load_state_dict(sd)
    diff.eval()
    with torch.no_grad():
        e_rest = torch.cat([diff.sample(e_under[i:i + 512].to(device), steps=steps).cpu()
                            for i in range(0, e_under.size(0), 512)])

    probe = LinearProbe(emb_dim, len(label_cols)).to(device)
    probe.load_state_dict(torch.load(probe_ckpt, map_location=device, weights_only=True))
    probe.eval()
    p_full, p_under, p_rest = (get_probs(probe, e, device).astype(np.float32)
                               for e in (e_full, e_under, e_rest))
    y = Y.numpy().astype(np.uint8)
    cos = lambda a, b: (F.normalize(a.float(), dim=-1) * F.normalize(b.float(), dim=-1)).sum(-1).numpy()

    np.savez_compressed(
        out,
        slice_id=np.asarray(ds.df["slice_id"].astype(str), dtype=str),     # 고정 길이 문자열(object 아님)
        study_id=np.asarray(ds.df[study_col].astype(str), dtype=str),
        label_cols=np.array(label_cols),
        y=y, p_full=p_full, p_under=p_under, p_rest=p_rest,
        cos_u=cos(e_full, e_under).astype(np.float32), cos_r=cos(e_full, e_rest).astype(np.float32),
    )
    print(f"     saved {out}   studies={ds.df[study_col].nunique()}   mean AUROC "
          f"E_f={mean_auroc(p_full, y):.4f}  E_u={mean_auroc(p_under, y):.4f}  E_r={mean_auroc(p_rest, y):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", help="train yaml 경로 (--all 이면 생략)")
    ap.add_argument("--all", action="store_true", help="5 인코더 × 6 view 전체")
    ap.add_argument("--encoders", nargs="+", default=ENCODERS, help=f"--all 과 함께: {ENCODERS}")
    ap.add_argument("--views", nargs="+", type=int, default=VIEWS)
    ap.add_argument("--split", default="test")
    ap.add_argument("--steps", type=int, default=1, help="역방향 샘플링 단계 수 (원고 기본 1)")
    ap.add_argument("--study_col", default="study_id")
    ap.add_argument("--out_dir", default="preds")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    cfgs = list(a.configs)
    if a.all:
        cfgs += [cfg_path(e, v) for e in a.encoders for v in a.views]
    if not cfgs:
        ap.error("config 경로를 주거나 --all 을 지정")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for c in cfgs:
        if not Path(c).exists():
            print(f"!! config 없음: {c}")
            continue
        dump(c, a.split, out_dir, a.steps, a.study_col, a.overwrite, device)


if __name__ == "__main__":
    main()
