"""
ref360_check.py — 기준을 180 → 360 view로 바꾸면 결과가 달라지는가? (MedSigLIP, test split)

리뷰어 예상 질문 "화질은 360 view에서 포화되는데 왜 180을 기준으로 했나?"에 임베딩 수준에서 답한다.
  extract : test split을 n_view(기본 360)로 재구성해 MedSigLIP 임베딩 추출.
            논문 파이프라인과 동일 (preprocess_dicom → FBP(음수 clip, max 정규화 없음) → window_to_pil → MedSigLIP).
            시작 전에 test 슬라이스 일부를 180 view로 다시 추출해 기존 feature와 비교(코사인 ≈ 1)하여
            파이프라인이 논문과 같은지 검증한다. 다르면 중단.
  eval    : 논문의 고정 선형 프로브(E_180으로 학습)를 E_180 / E_360 / (선택) E_90 등에 그대로 적용.
            같은 슬라이스 집합에서 라벨별 AUROC, 평균 AUROC, DeLong p(대 E_180), E_180과의 코사인을 계산.

해석: E_360의 AUROC가 E_180과 거의 같고(차이 ≲ 0.005, 부트스트랩 CI 안) cos(E_180, E_360)이 cos(E_180, E_90)보다 훨씬 1에 가까우면
      → 180을 기준으로 둔 선택이 결과에 영향을 주지 않는다는 직접 근거.

실행 위치: 서버 (이 파일은 ~/hayeon/ich_cf_clean/analysis/ 안에 있어야 함, GPU 필요)
--train_config 에는 논문 MedSigLIP 결과를 만든 학습 config 아무거나(예: nv6) — full_feature_dir, tag,
pooling, label_cols, csv_path 를 여기서 읽는다. 프로브는 weights/linprobe_{tag}_{pooling}.best.pt.

Usage (서버):
  cd ~/hayeon/ich_cf_clean

  # 1) 360-view test 임베딩 추출 (재실행 시 이어서 진행, 약 1–2시간)
  python analysis/ref360_check.py extract --train_config configs/medsiglip_after/train_medsiglip_after_nv90.yaml \
      --out_dir /data0/hayeon/features/medsiglip_after/feature_nv360 --n_views 360 \
      --workers 32 --batch_size 64

  #    빠른 예비 확인: 일부 검사만 (경막외 양성 검사는 항상 포함)
  #    ... extract ... --n_studies 800

  #    저장된 feature가 어떤 전처리로 만들어졌는지만 판별 (몇 분, 추출 안 함):
  python analysis/ref360_check.py extract --train_config configs/medsiglip_after/train_medsiglip_after_nv90.yaml \
      --out_dir /tmp/ref360_probe --check_only --workers 32 --batch_size 64

  # 2) 평가 (180 vs 360, 비교용으로 90-view 포함)
  python analysis/ref360_check.py eval --train_config configs/medsiglip_after/train_medsiglip_after_nv90.yaml \
      --dense_dir /data0/hayeon/features/medsiglip_after/feature_nv360 \
      --compare nv90=/data0/hayeon/features_v2/medsiglip_after/feature_nv90 \
      --out /home/neumous-lab/hayeon/ich_cf_clean/plots/paper_figs/ref360_check.csv

결과를 로컬로 가져오기 (로컬 PowerShell):
  scp neumous-lab@100.73.97.101:/home/neumous-lab/hayeon/ich_cf_clean/plots/paper_figs/ref360_check.csv "C:\\Users\\user\\Desktop\\MIGT\\01. sparse ct project\\논문작성\\figures\\paper_figs"
"""

import argparse
import gc
import json
import os
import multiprocessing as mp
import sys
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore", message="Radon transform: image must be zero outside")


# ── 재구성 (논문 파이프라인과 동일) ──────────────────────────────────────────
def _recon(img01, nv, rs):
    """nv=None: full(180) / nv<=180: 180 격자 부분집합 — 둘 다 추출 러너가 쓰는 reconstruct_hu_proxy.
    nv>180: [0,180) 균등 분할, 같은 FBP 절차(ramp, 음수 clip, max 정규화 없음)."""
    from extraction.sparse_ct import N_ANGLES_FULL
    from extraction_3ch.windowing import reconstruct_hu_proxy
    if nv is None or nv <= N_ANGLES_FULL:
        return reconstruct_hu_proxy(img01, radon_size=rs, n_views=nv)
    from skimage.transform import iradon, radon
    from skimage.transform import resize as sk_resize
    small = sk_resize(img01, (rs, rs), anti_aliasing=True).astype(np.float32)
    th = np.linspace(0.0, 180.0, nv, endpoint=False)
    rec = iradon(radon(small, theta=th, circle=True), theta=th, circle=True, filter_name="ramp")
    return np.clip(rec, 0.0, None).astype(np.float32)


def _recon_windowed(hu, center, width, nv, rs):
    """window-BEFORE 경로 (run_extract_medsiglip_mp.py 의 reconstruct_windowed 와 동일).
    nv=None/≤180 은 원 함수를 그대로 호출, nv>180 은 같은 절차를 [0,180) 균등 분할 각도로."""
    from extraction.sparse_ct import N_ANGLES_FULL, apply_window, reconstruct_windowed
    if nv is None or nv <= N_ANGLES_FULL:
        return reconstruct_windowed(hu, center, width, n_views=nv, radon_size=rs)
    from skimage.transform import iradon, radon
    from skimage.transform import resize as sk_resize
    w = apply_window(hu, center, width)
    small = sk_resize(w, (rs, rs), anti_aliasing=True, preserve_range=True).astype(np.float32)
    th = np.linspace(0.0, 180.0, nv, endpoint=False)
    rec = iradon(radon(small, theta=th, circle=True), theta=th, circle=True, filter_name="ramp")
    return np.clip(rec, 0.0, 1.0).astype(np.float32)


def _prep(task):
    """pipeline = "after" : preprocess_dicom → FBP → window_to_pil  (run_extract_encoder_mp.py, 원고 §3.5)
                  "before": dicom_to_hu → 채널별 window → FBP → clip  (run_extract_medsiglip_mp.py,
                            brain/subdural/soft)"""
    warnings.filterwarnings("ignore", message="Radon transform: image must be zero outside")
    sid, fp, nv, rs, windows, pipeline = task
    try:
        import pydicom
        try:
            pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE
        except Exception:
            pass
        if pipeline == "before":
            from extraction.sparse_ct import WINDOWS, dicom_to_hu
            hu = dicom_to_hu(pydicom.dcmread(fp))
            chans = [_recon_windowed(hu, *WINDOWS[k], nv, rs) for k in ("brain", "subdural", "soft")]
            arr = (np.stack(chans, axis=-1) * 255.0).clip(0, 255).astype(np.uint8)
            return sid, arr, None
        from extraction.sparse_ct import preprocess_dicom
        from extraction_3ch.windowing import window_to_pil
        img = preprocess_dicom(pydicom.dcmread(fp))
        pil = window_to_pil(_recon(img, nv, rs), windows)
        return sid, np.asarray(pil, dtype=np.uint8), None
    except Exception as e:  # noqa: BLE001
        return sid, None, repr(e)[:150]


def _embed_stream(pool, tasks, extractor, workers, batch_size, on_batch, desc, stall_timeout):
    """bounded in-flight 전처리(pool) → GPU 배치 임베딩 → on_batch(sids, embeds).
    stall_timeout 초 동안 완료되는 작업이 하나도 없으면 멈춘 것으로 보고 오류로 종료."""
    from PIL import Image
    from tqdm import tqdm
    failed, meta, imgs = [], [], []
    pbar = tqdm(total=len(tasks), desc=desc, ncols=80)

    def flush():
        if imgs:
            on_batch(list(meta), extractor.extract_batch(list(imgs)))
            pbar.update(len(meta)); meta.clear(); imgs.clear()

    it = iter(tasks)
    inflight = set()
    for _ in range(max(workers * 3, 32)):
        t = next(it, None)
        if t is None:
            break
        inflight.add(pool.submit(_prep, t))
    while inflight:
        done, inflight = wait(inflight, timeout=stall_timeout, return_when=FIRST_COMPLETED)
        if not done:
            pbar.close()
            raise RuntimeError(f"{stall_timeout}s 동안 전처리 작업 완료 없음 (worker 정지). "
                               f"--workers 를 줄여 재실행하세요. 이미 저장된 임베딩은 이어서 진행됩니다.")
        inflight = set(inflight)
        for fut in done:
            nt = next(it, None)
            if nt is not None:
                inflight.add(pool.submit(_prep, nt))
            sid, arr, err = fut.result()
            if arr is None:
                failed.append((sid, err)); pbar.update(1); continue
            meta.append(sid); imgs.append(Image.fromarray(arr, "RGB"))
            if len(imgs) >= batch_size:
                flush()
    flush()
    pbar.close()
    return failed


def _unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def load_cfg(path):
    from utils import get_config
    return get_config(path)


# ── extract ──────────────────────────────────────────────────────────────────
def cmd_extract(a):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 필요")
    from extraction_3ch.backbone_medsiglip import MedSigLIPExtractor
    from extraction_3ch.windowing import parse_windows

    cfg = load_cfg(a.train_config)
    csv = a.master_csv or cfg.csv_path
    full_dir = Path(a.full_dir or cfg.full_feature_dir) / "test"
    out_dir = Path(a.out_dir) / "test"; out_dir.mkdir(parents=True, exist_ok=True)
    windows = parse_windows(None)
    device = torch.device("cuda")

    df = pd.read_csv(csv)
    test = df[df["split"] == "test"]
    if a.n_studies > 0:                                    # 예비 확인용 부분집합 (검사 단위)
        rng = np.random.default_rng(a.seed)
        edh = set(test.loc[test["epidural"] == 1, "study_id"])
        rest = sorted(set(test["study_id"]) - edh)
        k = max(0, a.n_studies - len(edh))
        keep = edh | set(rng.choice(rest, size=min(k, len(rest)), replace=False))
        test = test[test["study_id"].isin(keep)]
        print(f"부분집합: {len(keep)} studies (경막외 양성 {len(edh)} 전부 포함), {len(test):,} slices")

    print(f"csv={csv}\nfull(180) test={full_dir}\nout={out_dir}\nn_views={a.n_views}  windows={windows}")
    extractor = MedSigLIPExtractor(device, model_id=a.model_id)

    def save(sids, embs):
        for sid, e in zip(sids, embs):
            tmp = out_dir / f".{sid}.tmp.npy"
            np.save(str(tmp), e)
            os.replace(tmp, out_dir / f"{sid}.npy")

    # spawn: 작업 프로세스가 부모(GPU에 올라간 모델, transformers/tqdm 스레드)의 잠금 상태를
    # 물려받지 않도록 새로 시작 → fork 후 import 교착(이전 버전에서 128/200에서 멈춘 문제) 방지.
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as pool:
        # 0) 파이프라인 검증: 180 view 재추출 vs 저장된 feature.
        #    auto 이면 두 전처리(before/after)를 모두 시험해 저장 feature와 일치하는 쪽을 선택.
        pipeline = a.pipeline
        if a.check_n > 0:
            have = [s for s in test["slice_id"] if (full_dir / f"{s}.npy").exists()]
            rng = np.random.default_rng(0)
            pick = set(rng.choice(have, size=min(a.check_n, len(have)), replace=False))
            sub = test[test["slice_id"].isin(pick)]
            cands = ["before", "after"] if a.pipeline == "auto" else [a.pipeline]
            scores = {}
            for p in cands:
                got = {}
                _embed_stream(pool, [(r.slice_id, r.filepath, None, a.radon_size, windows, p)
                                     for r in sub.itertuples()],
                              extractor, a.workers, a.batch_size,
                              lambda sids, embs: got.update(zip(sids, embs)), f"[check 180 {p}]",
                              a.stall_timeout)
                s = sorted(got)
                new = _unit(np.stack([got[k] for k in s]).astype(np.float32))
                old = _unit(np.stack([np.load(full_dir / f"{k}.npy").astype(np.float32).reshape(-1)
                                      for k in s]))
                cos = (new * old).sum(1)
                scores[p] = float(cos.mean())
                print(f"[check] pipeline={p:6s}  180-view 재추출 vs 저장 feature: "
                      f"cos mean={cos.mean():.5f} min={cos.min():.5f} (n={len(s)})")
                del got; gc.collect()
            pipeline = max(scores, key=scores.get)
            if scores[pipeline] < a.check_min_cos and not a.force:
                raise SystemExit(f"어느 전처리도 저장된 feature와 일치하지 않음 {scores} → 360 추출 중단. "
                                 "full_feature_dir 경로 확인 (--force 로 무시 가능).")
            print(f"[check] 저장 feature와 일치하는 전처리: {pipeline}")
            if a.check_only:
                print("--check_only: 확인만 하고 종료 (추출 안 함)")
                return
            print(f"  → 이 방식으로 {a.n_views} view 추출")
        elif pipeline == "auto":
            raise SystemExit("--check_n 0 이면 --pipeline before|after 를 직접 지정해야 함")
        (Path(a.out_dir) / "PIPELINE.txt").write_text(
            f"pipeline={pipeline}\nn_views={a.n_views}\nfull_dir={full_dir}\n")

        # 1) n_view 추출 (resumable, atomic)
        done = {p.stem for p in out_dir.glob("*.npy")}
        todo = test[~test["slice_id"].isin(done)]
        print(f"대상 {len(test):,}  완료 {len(done):,}  남음 {len(todo):,}")
        failed = _embed_stream(pool, [(r.slice_id, r.filepath, a.n_views, a.radon_size, windows,
                                       pipeline) for r in todo.itertuples()],
                               extractor, a.workers, a.batch_size, save, f"[nv{a.n_views}]",
                               a.stall_timeout)
    print(f"완료: {len(list(out_dir.glob('*.npy'))):,} / {len(test):,}   실패: {len(failed)}")
    if failed:
        (Path(a.out_dir) / f"failed_nv{a.n_views}.txt").write_text(
            "\n".join(f"{s}\t{e}" for s, e in failed))


# ── eval ─────────────────────────────────────────────────────────────────────
def cmd_eval(a):
    import torch
    from probe import LinearProbe
    from sklearn.metrics import roc_auc_score
    from train_diffusion import delong_per_label

    cfg = load_cfg(a.train_config)
    labels = list(getattr(cfg, "label_cols", ["epidural", "intraparenchymal", "intraventricular",
                                              "subarachnoid", "subdural", "any"]))
    pooling = getattr(cfg, "pooling", "none")
    reps = {"ref180": Path(cfg.full_feature_dir), "dense": Path(a.dense_dir)}
    for c in a.compare or []:
        name, d = c.split("=", 1)
        reps[name] = Path(d)

    df = pd.read_csv(a.master_csv or cfg.csv_path)
    test = df[df["split"] == a.split]
    ids = [s for s in test["slice_id"]
           if all((d / a.split / f"{s}.npy").exists() for d in reps.values())]
    test = test.set_index("slice_id").loc[ids]
    print(f"공통 슬라이스 {len(ids):,}  (표현: {', '.join(reps)})")
    if not ids:
        raise SystemExit("공통 슬라이스가 없음 — 경로 확인")

    def load(d):
        with ThreadPoolExecutor(16) as ex:
            arrs = list(ex.map(lambda s: np.load(d / a.split / f"{s}.npy").astype(np.float32).reshape(-1), ids))
        return np.stack(arrs)

    E = {k: load(d) for k, d in reps.items()}
    Y = test[labels].to_numpy().astype(np.float32)

    probe_path = a.probe or f"weights/linprobe_{cfg.tag}_{pooling}.best.pt"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = LinearProbe(E["ref180"].shape[1], len(labels)).to(dev)
    probe.load_state_dict(torch.load(probe_path, map_location=dev, weights_only=True)); probe.eval()
    print(f"probe: {probe_path}")

    def probs(x):
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 1024):
                lg = probe(torch.from_numpy(x[i:i + 1024]).to(dev))
                out.append(torch.sigmoid(torch.cat(lg, 1)).cpu().numpy())
        return np.concatenate(out)

    P = {k: probs(v) for k, v in E.items()}
    ref_u = _unit(E["ref180"])
    rows = []
    for k in E:
        auc = {l: roc_auc_score(Y[:, i], P[k][:, i]) for i, l in enumerate(labels)}
        cos = (_unit(E[k]) * ref_u).sum(1)
        row = {"representation": k, "n": len(ids), "mean_auroc": float(np.mean(list(auc.values()))),
               "cos_to_ref180_mean": float(cos.mean()), "cos_to_ref180_p5": float(np.percentile(cos, 5))}
        row.update({f"auroc_{l}": v for l, v in auc.items()})
        if k != "ref180":
            dl = delong_per_label(P["ref180"], P[k], torch.from_numpy(Y), labels)
            row.update({f"delong_p_{l}": float(dl[l][2]) for l in labels})
        rows.append(row)

    res = pd.DataFrame(rows)
    res["delta_vs_ref180"] = res["mean_auroc"] - res.loc[res.representation == "ref180", "mean_auroc"].item()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    show = ["representation", "n", "mean_auroc", "delta_vs_ref180", "cos_to_ref180_mean", "cos_to_ref180_p5"]
    print("\n" + res[show].round(4).to_string(index=False))
    print(f"\nsaved → {out}")
    (out.with_suffix(".json")).write_text(json.dumps(
        {"probe": probe_path, "reps": {k: str(v) for k, v in reps.items()}}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--train_config", required=True)
    e.add_argument("--out_dir", required=True, help="{out_dir}/test/{slice_id}.npy 로 저장")
    e.add_argument("--n_views", type=int, default=360)
    e.add_argument("--master_csv", default=None, help="기본: train_config의 csv_path")
    e.add_argument("--full_dir", default=None, help="기본: train_config의 full_feature_dir")
    e.add_argument("--model_id", default="google/medsiglip-448")
    e.add_argument("--radon_size", type=int, default=256)
    e.add_argument("--batch_size", type=int, default=64)
    e.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 4))
    e.add_argument("--n_studies", type=int, default=0, help="0=test 전체")
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--check_n", type=int, default=200, help="180-view 재현 확인 슬라이스 수 (0=생략)")
    e.add_argument("--check_min_cos", type=float, default=0.995)
    e.add_argument("--force", action="store_true")
    e.add_argument("--check_only", action="store_true",
                   help="180 재현 확인(전처리 판별)만 하고 종료")
    e.add_argument("--pipeline", choices=["auto", "before", "after"], default="auto",
                   help="before=윈도우 후 FBP(run_extract_medsiglip_mp), after=FBP 후 윈도우(원고 §3.5). "
                        "auto=180 재현 확인으로 자동 선택")
    e.add_argument("--stall_timeout", type=int, default=600,
                   help="이 시간(초) 동안 진행이 없으면 멈춘 것으로 보고 오류 종료")
    v = sub.add_parser("eval")
    v.add_argument("--train_config", required=True)
    v.add_argument("--dense_dir", required=True)
    v.add_argument("--compare", action="append", help="name=feature_dir (여러 번 가능)")
    v.add_argument("--master_csv", default=None)
    v.add_argument("--probe", default=None)
    v.add_argument("--split", default="test")
    v.add_argument("--out", default="plots/paper_figs/ref360_check.csv")
    a = ap.parse_args()
    cmd_extract(a) if a.cmd == "extract" else cmd_eval(a)


if __name__ == "__main__":
    main()
