# exp_unseen_views — 학습에 쓰지 않은 view 수(30·70)에서 복원기 평가

2026-10-30 작성. 10/28 미팅 할 일 3 ("30, 70view 등 추가로 test, MLP랑 성능 비교, MLP보다 높아야 굳")과
교수님 제안 ("90·45-view로 학습한 가중치로 70·30-view test")을 위한 실험이다. 기존 코드는 건드리지 않고 이 폴더에만 둔다.

## 왜 하나

- 실제 장비·프로토콜마다 view 수가 다를 수 있다. 한 view 수로 학습한 복원기를 다른 view 수에 그대로 써도 되는지 본다.
- diffusion은 열화 정도(t)를 연속으로 배우므로, 한 열화 수준만 배운 MLP보다 처음 보는 view 수에 강할 수 있다.
  이것이 diffusion을 쓸 남은 근거다. 같은 view 공정 비교(6·23·90)에서는 MLP가 이겼다.

## 무엇을 하나

학습은 하지 않는다. 학습 view s의 가중치를 그대로 불러와 test view v의 E_u에 1단계로 적용한다.

| | 내용 |
|---|---|
| 학습 view s | 90, 45 (주 비교). 23, 6은 보조 |
| test view v | 70, 30 (test 3,262 검사만 새로 추출) |
| 방법 | identity(= E_u), ridge, mlp, direct, diffusion, diffusion_vl, diffusion_nc(A+C, 6-view에만 있음) |
| 프로브 | 학습 view s의 고정 프로브 (E_f로 학습, 원고·compare와 같음) |
| 지표 | 6라벨 평균 AUROC, R(test view v의 E_u 기준), 검사 단위 bootstrap (B = 2000, Holm) |

## 미리 정한 판정 기준 (결과 보기 전에 고정)

- 주 비교: 90 → 70, 45 → 30 (교수님 제안 두 쌍). diffusion은 공정 비교와 같은 `diffusion`(복원 MSE 선택)으로 판정하고,
  원고 규칙 `diffusion_vl`은 함께 보고만 한다.
- **diffusion 성공**: 두 쌍 **모두**에서 AUROC(diffusion) > AUROC(mlp)이고,
  Δ = AUROC(mlp) − AUROC(diffusion)의 95% CI 상한 < 0 (Holm 보정 p < 0.05).
- **그 외(비슷하거나 낮음)**: 처음 보는 view 수에서도 diffusion의 이점은 없다 → MLP 주 모델 방향 유지.
  이때 이 결과는 "한 view 수로 학습한 MLP가 다른 view 수에도 쓸 수 있는가"라는 강건성 결과로 보고한다.
- 23·6-view 학습 가중치로 한 결과는 탐색용(판정에 쓰지 않음).

## 빠른 확인 (약 20–30분, 학습 없음)

결과 방향만 먼저 보는 탐색용. 위 판정 기준의 정식 판정은 아래 "실행 순서"(전체 test, 45-view 포함)로 한다.

- 45-view 학습을 건너뛰고 이미 있는 90·23-view 가중치만 쓴다 → 45 → 30 대신 23 → 30.
  (90 → 70은 깨끗한 쪽에서 배워 더 열화된 쪽에 적용, 23 → 30은 그 반대 방향)
- test 3,262검사 중 500검사(seed 0으로 고정, 약 1.7만 슬라이스)만 30·70-view로 추출 (view당 약 6–7분 예상).
  출력 폴더가 같아서 나중에 전체를 뽑으면 이미 뽑은 슬라이스는 건너뛴다.
- 같은 view(90 → 90, 23 → 23)도 같은 500검사로 같이 평가해 "처음 보는 view에서 얼마나 떨어지나"를 같은 표본에서 비교한다.
- ridge는 학습 데이터를 다시 읽어야 해서(약 10분) 뺀다.
- 검사 수가 1/6이라 CI가 전체의 약 2.5배 넓다.

```bash
ls compare/weights
```

```bash
python exp_unseen_views/extract_test_views.py --n_studies 500
```

```bash
python exp_unseen_views/eval_unseen_views.py --train_views 90 23 --test_views 90 70 30 23 --methods identity mlp direct diffusion diffusion_vl --test_csv /data0/hayeon/master_df_test_sub500.csv --B 2000
```

```bash
scp neumous-lab@100.73.97.101:/home/neumous-lab/hayeon/ich_cf_clean/results_unseen_sub500_*.csv "C:\Users\user\Desktop\MIGT\01. sparse ct project\result"
```

결과는 `compare_unseen_sub500/`, `results_unseen_sub500_*.csv`에 따로 저장된다(전체 실행 결과와 섞이지 않음).

## 실행 순서 (서버, `~/hayeon/ich_cf_clean`)

**0) 가중치 확인.** 학습 view 가중치가 있어야 한다 (`compare_restorers.py --save_weights`로 저장된 것).

```bash
ls compare/weights
```

**1) 45-view 공정 비교 + 가중치** (45는 아직 compare를 안 돌림; 메인 비교표의 빠진 45-view도 같이 채워짐, 약 2–3시간)

```bash
python compare_restorers.py configs/medsiglip_after/train_medsiglip_after_nv45.yaml --seeds 42 --save_weights
```

- 90-view 가중치(`medsiglip_after_r360_nv90__mlp__s42.pt` 등)가 없으면: compare는 결과가 있으면 건너뛰므로 다른 폴더로 다시 학습한다.
  평가할 때 `--weights_dir compare_w/weights`를 붙인다.
  `python compare_restorers.py configs/medsiglip_after/train_medsiglip_after_nv90.yaml --seeds 42 --methods mlp direct diffusion --save_weights --out_dir compare_w`

**2) 30·70-view test feature 추출** (약 40–50분 × 2, 1)과 다른 tmux 창에서 동시에 돌려도 됨)

```bash
python exp_unseen_views/extract_test_views.py
```

- 먼저 `--dry_run`으로 만들어질 config와 출력 경로만 확인할 수 있다.
- 출력: `/data0/hayeon/features_v2/medsiglip_after/feature_nv30/test/`, `feature_nv70/test/`

**3) 평가 + bootstrap** (학습 view마다 ridge용 데이터 로딩 약 10분 + 평가 몇 분)

```bash
python exp_unseen_views/eval_unseen_views.py --train_views 90 45 --test_views 70 30 --B 2000
```

- 끊기면 같은 명령으로 이어서 실행된다 (결과가 있는 조건은 건너뜀).
- 보조 비교까지: `--train_views 90 45 23 6`

**4) 결과 가져오기** (로컬 PowerShell)

```bash
scp neumous-lab@100.73.97.101:/home/neumous-lab/hayeon/ich_cf_clean/results_unseen_*.csv "C:\Users\user\Desktop\MIGT\01. sparse ct project\result"
```

## 출력

| 파일 | 내용 |
|---|---|
| `results_unseen_runs.csv` | 학습 view × test view × 방법 1행: AUROC, Δ, R, cos, test MSE, 같은 view(s → s)에서의 AUROC |
| `results_unseen_table.csv` | 행 = test view × 방법, 열 = 학습 view, 값 "AUROC (R%)" — 슬라이드용 |
| `results_unseen_bootstrap.csv` | AUROC 95% CI, Δ = AUROC(mlp) − AUROC(방법) 95% CI, Holm |
| `compare_unseen/` | 조건별 json·예측(npz)·로그 (collect_restorers.py와 같은 형식) |

## 읽는 법

- identity 행 = 그 test view의 E_u. 복원기 AUROC가 identity보다 높으면 처음 보는 view에서도 복원이 도움이 된다.
- `auroc_r_same_view`(s → s)와 비교하면, 학습 view와 다른 view에 썼을 때 얼마나 떨어지는지 보인다.
- 30·70-view로 직접 학습한 복원기(상한선)는 이 실험에 없다. 필요하면 30·70의 train/val까지 전부 추출해야 한다 (view당 4–5시간).
