"""
make_mlp_preds.py — 검사 단위 평가(study_level_eval.py)를 MLP 복원 결과로 돌리기 위한 예측 파일 만들기.

왜: preds/{tag}_test.npz 의 p_rest 는 원고 v0.1 diffusion 가중치로 만든 것이다. 주 방법이 MLP로 바뀌었으므로
    compare_restorers.py 가 저장한 MLP 예측(compare/preds)을 study_level_eval.py 가 읽는 형식으로 합친다.
무엇을:
  test : compare/preds/{tag}__ref_test.npz (slice_id, study_id, label_cols, y, p_full, p_under, cos_u)
         + compare/preds/{tag}__{method}__s{seed}_test.npz (p_rest, cos_r)  →  {out_dir}/{tag}_test.npz
         (cos_u·cos_r 은 bootstrap_study.py 가 코사인 CI를 계산할 때 필요. 11-18 추가)
  val  : preds/{tag}_val.npz 를 그대로 복사 (study_level_eval 은 val 에서 E_f 확률만 써서 고정 임계값을 정함)
         없으면 건너뛰고 알려 줌 → 먼저 dump_preds.py --all --split val
재학습 없음, CPU, 몇 초.

Usage (서버):
  conda activate hayeon
  cd ~/hayeon/ich_cf_clean
  python analysis/dump_preds.py --all --split val
  python analysis/make_mlp_preds.py
  python analysis/study_level_eval.py --pred_dir preds_mlp --B 2000 --out_dir .
  다른 방법·seed로:  python analysis/make_mlp_preds.py --method diffusion --seed 42 --out_dir preds_diffusion
결과 가져오기 (로컬 PowerShell, 한 줄씩):
  scp neumous-lab@100.73.97.101:/home/neumous-lab/hayeon/ich_cf_clean/study_auroc_*.csv "C:\\Users\\user\\Desktop\\MIGT\\01. sparse ct project\\result"
  scp neumous-lab@100.73.97.101:/home/neumous-lab/hayeon/ich_cf_clean/sens_at_spec.csv "C:\\Users\\user\\Desktop\\MIGT\\01. sparse ct project\\result"
"""
import argparse
import glob
import os
import shutil

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare_dir", default="compare/preds")
    ap.add_argument("--val_dir", default="preds", help="dump_preds.py --split val 출력 위치")
    ap.add_argument("--out_dir", default="preds_mlp")
    ap.add_argument("--method", default="mlp")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    refs = sorted(glob.glob(os.path.join(a.compare_dir, "*__ref_test.npz")))
    if not refs:
        raise SystemExit(f"{a.compare_dir}에 *__ref_test.npz 없음 (compare_restorers.py 결과 필요)")
    n_test, no_rest, no_val, no_cos = 0, [], [], []
    for ref_path in refs:
        tag = os.path.basename(ref_path)[: -len("__ref_test.npz")]
        rest_path = os.path.join(a.compare_dir, f"{tag}__{a.method}__s{a.seed}_test.npz")
        if not os.path.exists(rest_path):
            no_rest.append(tag)
            continue
        r, m = np.load(ref_path, allow_pickle=True), np.load(rest_path, allow_pickle=True)
        assert m["p_rest"].shape == r["p_full"].shape, f"{tag}: 크기 다름 {m['p_rest'].shape} vs {r['p_full'].shape}"
        extra = {}
        for key, src in (("cos_u", r), ("cos_r", m)):
            if key in src.files:
                extra[key] = src[key]
            else:
                no_cos.append(f"{tag}:{key}")
        np.savez_compressed(os.path.join(a.out_dir, f"{tag}_test.npz"),
                            slice_id=r["slice_id"], study_id=r["study_id"], label_cols=r["label_cols"],
                            y=r["y"], p_full=r["p_full"], p_under=r["p_under"], p_rest=m["p_rest"], **extra)
        n_test += 1
        val_src = os.path.join(a.val_dir, f"{tag}_val.npz")
        if os.path.exists(val_src):
            shutil.copyfile(val_src, os.path.join(a.out_dir, f"{tag}_val.npz"))
        else:
            no_val.append(tag)

    print(f"test 예측 {n_test}개 → {a.out_dir}/  (복원: {a.method}, seed {a.seed})")
    if no_rest:
        print(f"  {a.method} s{a.seed} 결과 없음 {len(no_rest)}개: {', '.join(no_rest)}")
    if no_cos:
        print(f"  코사인 없음 {len(no_cos)}개 (bootstrap_study.py 실패 원인): {', '.join(no_cos)}")
    else:
        print("  cos_u·cos_r 포함")
    if no_val:
        print(f"  val 예측 없음 {len(no_val)}개 → 고정 특이도(fixed)는 빠짐. 먼저: python analysis/dump_preds.py --all --split val")
    else:
        print("  val 예측도 모두 복사함")


if __name__ == "__main__":
    main()
