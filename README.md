# ich_cf_clean — sparse-view CT 임베딩 복원 (최종 파이프라인만)

2026-09-30 정리, 2026-10-01 기준 360 view로 변경. 서버 `~/hayeon/ich_cf` (9/30 코드 묶음) 중 **논문에 쓰는 최종 파이프라인**만 모은 폴더.
기존 `~/hayeon/ich_cf`는 그대로 두고, 이 폴더를 `~/hayeon/ich_cf_clean`으로 올려 여기서 다시 학습한다.

## 1. 최종 파이프라인 (다섯 인코더 모두 동일)

```
DICOM → HU → clip[−1024, 2048] → (HU+1024)/3072 ∈ [0,1]          extraction/sparse_ct.py  preprocess_dicom
      → 256×256 → Radon → FBP 각도 선택                              extraction_3ch/windowing.py  reconstruct_hu_proxy
        기준(reference) 360 view: [0°,180°) 0.5° 간격
        sparse n_v ∈ {180,90,45,23,12,6}: 1° 격자에서 sparse_view_indices = [0°,180°)를 n_v 균등 분할
          (예 6 view → 0,30,…,150°; 180 view = 1° 격자 전체)
      → FBP(ramp) → 음수만 clip (영상별 최댓값 정규화 없음)
      → HU로 되돌림 → 윈도우 R brain 40/80, G subdural 80/200, B bone 600/2800   window_to_pil
      → 인코더 → 고유 pooled 임베딩 (.npy, float16)                  extraction_3ch/run_extract_encoder_mp.py
```

- **재구성 먼저, 윈도우 나중.** (예전 MedSigLIP 전용 코드 `run_extract_medsiglip*.py`는 윈도우를 재구성 *전에*
  적용해 뼈 줄무늬가 약해지는 문제가 있어 이 폴더에서 뺐다.)
- **view 선택 수정 (2026-09-30)**: 예전 규칙 `linspace(0, 179, n_v)`는 0°와 179°를 함께 골라(평행빔에서
  거의 같은 방향) 사실상 1 view가 중복됐다(6 view → 0,35,71,107,143,179°). `sparse_view_indices`로 바꾸고
  sparse feature는 `/data0/hayeon/features_v2/`에 다시 추출한다. full은 180각을 모두 써서 영향이 없으므로
  기존 `/data0/hayeon/features/*/feature_full`을 그대로 쓴다.
- **기준 360 view (2026-10-01)**: 평행 빔에서 aliasing 없는 재구성에는 약 π×(물체 반경) view가 필요하다
  (256 격자에서 머리 반경 ≈110 px → ≈346). 180은 이에 못 미치고, 360은 충족하며 720과 SSIM 차이가 0.001.
  그래서 기준을 360으로 올리고 180은 sparse 조건(×2)으로 둔다. 1° 격자 각도는 모두 0.5° 격자에 포함되고
  Radon 투영은 각도별로 독립이므로 sparse feature(nv90–6)와 180(= 예전 feature_full)은 다시 뽑지 않는다.
  새로 뽑는 것은 `features_v2/<인코더>/feature_ref360` 하나뿐. tag는 `<인코더>_r360_nv<V>`.
- 추출기의 `input_mode: adjacent`는 레거시(영상별 최댓값 정규화) 경로라 막아 두었다(`window`만 허용).
- 인코더와 임베딩: MedSigLIP `google/medsiglip-448`, SigLIP `google/siglip-so400m-patch14-384`
  (attention pooling, 1152), DINOv2 `facebook/dinov2-large` (LayerNorm CLS, 1024), ResNet50 IMAGENET1K_V2
  (GAP, 2048), DenseNet121 IMAGENET1K_V1 (GAP, 1024).
- 참고: MedSigLIP의 사전학습 CT 윈도우 규약(R wide, G soft, B brain)과는 다르다. 모든 인코더에 같은 입력을
  주기 위한 선택이며 원고 §3.5에 이유를 적는다.

### 복원·평가
- `train_diffusion.py`: Stage 1 콜드 디퓨전(T=100, SwiGLU 4블록×1024, AdamW 1e-4, 배치 256, EMA 0.999,
  200 에폭) → Stage 2 선형 프로브(E_f = 360-view 기준 임베딩으로 학습, 이후 고정) → Stage 3 val 평가.
- `run_eval.py --split test`: 같은 가중치로 test 평가 (라벨별 AUROC, DeLong, 코사인, t-SNE).
- **시드 고정**: config의 `seed`(없으면 42)로 python/numpy/torch 난수를 고정 (`train_diffusion.set_seed`).
  같은 인코더의 config들은 같은 E_f·같은 시드로 프로브를 학습하므로 투영 수와 관계없이 같은 프로브가 된다.

## 2. 폴더 구조

| 경로 | 역할 |
|---|---|
| `extraction/sparse_ct.py` | 각도 격자(`ANGLES_FULL`), `preprocess_dicom` |
| `extraction_3ch/windowing.py` | `reconstruct_hu_proxy`(FBP), `window_to_pil`(윈도우 → RGB), 기본 윈도우 |
| `extraction_3ch/run_extract_encoder_mp.py` | 추출 러너 (ProcessPool, 이어서 실행 가능, atomic 저장) |
| `extraction_3ch/backbone_*.py` | MedSigLIP / SigLIP / DINOv2 / CNN 래퍼 |
| `data/ diffusion/ probe/ pooling/ utils/` | 데이터셋, 확산 모델, 프로브, pooling(`none` 사용), config 로더 |
| `train_diffusion.py`, `run_eval.py` | 학습(+시드), 평가 |
| `collect_test_results.py` | test 로그 → `results_test_summary.csv`(표 2), `results_test_perlabel.csv`(표 3) |
| `train_baselines.py`, `collect_baselines.py` | (옛) 회귀 기준선 identity / ridge / residual MLP. 레시피·선택 기준이 디퓨전과 달라 **아래 `compare_restorers.py`로 대체** |
| `compare_restorers.py`, `scripts/compare_all.sh`, `collect_restorers.py` | 복원기 공정 비교 + 학습 seed 반복 (피드백 1.1·1.2·2.3): identity, mean shift, CORAL, ridge, 같은 구조 직접 회귀(direct), residual MLP, diffusion(선택 규칙 2개). 같은 프로브·레시피·선택 기준. 결과는 `compare/`에만 씀 |
| `analysis/study_level_eval.py` | 검사 단위 AUROC + 고정 특이도(0.90/0.95) 민감도, any·EDH (피드백 3.2). `preds/*.npz` 사용, 재학습 없음 |
| `analysis/fig_views_psnr.py` | Fig. 2 (view 수별 재구성 격자) + PSNR/SSIM 곡선 |
| `analysis/ref360_check.py` | 저장 feature의 전처리 판별(`--check_only`). 360 확인 실험은 기준을 360으로 바꿔 불필요 |
| `analysis/fig_view_artifact_factors.py` | 같은 view 수에서 표시 창·뼈 clip·해상도·필터·윈도우 순서·SART별 줄무늬 비교, `--auto_pick`으로 출혈 큰 슬라이스 선택 |
| `analysis/fig_overview_panels.py` | Fig. 1(a) 패널용 중간 영상(입력·사이노그램·FBP·RGB) npz |
| `analysis/fig_tsne_cosine.py` | Fig. 5: t-SNE + 슬라이스별 코사인 분포 + top-1 대응 비율 |
| `scripts/steps_sweep.sh`, `collect_steps_results.py` | 샘플링 단계 수 1/5/10/20 평가 → `results_steps.csv`(표 4) |
| `configs/{medsiglip_after,siglip,dinov2,resnet50,densenet121}/` | 인코더별 추출(full=예전 180, nv90–6, ref360) + 학습 6개(nv180–6) config |
| `scripts/train_all.sh` | 5 인코더 × 6 sparse 조건 학습 → test 평가 → 결과 수집 |
| `scripts/extract_all.sh` | sparse 5조건(nv90–6) 추출 → `features_v2` (이미 있는 .npy는 건너뜀) |
| `scripts/extract_ref360.sh` | 기준 360 view 추출 → `features_v2/<인코더>/feature_ref360` |

로컬 전용 그림 스크립트(서버 불필요)는 `Desktop\MIGT\01. sparse ct project\논문작성\figures\`에 있다:
`make_figures_134.py`(Fig. 1·3·4, 원고 표 2를 읽음), `make_graphical_abstract.py`.

### 뺀 것
예전 MedSigLIP 추출기·config(윈도우 먼저, soft), CT-FM(`run_extract_3ch*`, `backbone_3ch`, `extraction/run_extract.py`),
MedGemma, 무작위 각도(`run_extract_encoder_mp_v2`, `medsiglip_rand`), ThreadPool 구버전(`run_extract_encoder.py`),
중복 config(`configs/Dino`), 옛 비교·그림 스크립트, 옛 `collect_results.py`(`…_90view_…` 로그 이름만 읽음),
옛 `CLAUDE.md`("윈도우를 재구성 전에" 규칙이 적혀 있어 최종 파이프라인과 반대).

## 3. 실행 순서 (서버)

### 0) 올리기 — 로컬 PowerShell
```powershell
scp "C:\Users\user\Desktop\MIGT\01. sparse ct project\code\ich_cf_clean_20260930.tgz" neumous-lab@100.73.97.101:/home/neumous-lab/hayeon/ich_cf_clean_20260930.tgz
```
서버에서:
```bash
cd ~/hayeon && tar -xzf ich_cf_clean_20260930.tgz && ls ich_cf_clean
```

### 1) sparse 재추출 — view 선택 수정 반영 (5 인코더 × 5 조건)
full은 기존 `/data0/hayeon/features/<인코더>/feature_full` 그대로. sparse만 새로 뽑는다(약 45 GB, 조건당 4–5시간).
```bash
df -h /data0
tmux new -s extract_v2
cd ~/hayeon/ich_cf_clean
bash scripts/extract_all.sh 2>&1 | tee logs/extract_all_$(date +%Y%m%d_%H%M).log
```
끊겨도 같은 명령으로 이어서 실행(이미 있는 .npy는 건너뜀). 인코더를 나눠 돌리려면 `bash scripts/extract_all.sh siglip`.

### 1-b) 기준 360 view 추출 (5 인코더) — extract_all.sh와 다른 tmux에서 함께 돌려도 됨
```bash
tmux new -s ref360
cd ~/hayeon/ich_cf_clean
bash scripts/extract_ref360.sh 2>&1 | tee logs/extract_ref360_$(date +%Y%m%d_%H%M).log
```
※ 실행 중인 `extract_all.sh` 파일은 서버에서 덮어쓰지 말 것.

### 2) 학습 + test 평가 — 30개 조건 (해당 인코더의 ref360과 sparse 추출이 모두 끝난 뒤)
```bash
tmux new -s train_all
cd ~/hayeon/ich_cf_clean
bash scripts/train_all.sh 2>&1 | tee logs/train_all_$(date +%Y%m%d_%H%M).log
```
끊겨도 같은 명령으로 이어서 실행 (test 로그가 있는 조건은 건너뜀). 특정 인코더만: `bash scripts/train_all.sh siglip`.

### 3) 결과 모으기
```bash
cd ~/hayeon/ich_cf_clean && python collect_test_results.py
```
로컬로:
```powershell
scp neumous-lab@100.73.97.101:/home/neumous-lab/hayeon/ich_cf_clean/results_test_*.csv "C:\Users\user\Desktop\MIGT\01. sparse ct project\논문작성\figures\paper_figs"
```

### 4) (선택) 회귀 기준선, 확인 실험
```bash
python train_baselines.py configs/medsiglip_after/train_medsiglip_after_nv6.yaml --baseline all --split test
python analysis/fig_views_psnr.py --master_csv /data0/hayeon/master_df.csv --split test --n 2000 --views 6,12,23,45,90,180,360,720 --workers 32 --out_dir plots/paper_figs
```

### 5) 그림·표 4용 (해당 인코더 학습이 끝난 뒤)
```bash
python analysis/fig_overview_panels.py --master_csv /data0/hayeon/master_df.csv --views 45 --out_dir plots/paper_figs/overview   # CPU, 추출 중에도 가능
bash scripts/steps_sweep.sh                     # MedSigLIP 6·12 view × 1/5/10/20 단계 → results_steps.csv
python analysis/fig_tsne_cosine.py --train_config configs/medsiglip_after/train_medsiglip_after_nv6.yaml --out_dir plots/paper_figs/tsne
```

### 6) 복원기 공정 비교 + seed 반복 (피드백 1.1·1.2·2.3, 2026-10-26 추가)
같은 (E_f, E_u) 쌍·같은 고정 프로브(`weights/linprobe_{tag}_none.best.pt`)·같은 레시피(AdamW, 배치 256, 200 에폭, AMP, EMA 0.999)로
비교한다. 학습 방법의 체크포인트는 모두 **val 복원 MSE(1단계, t = T−1)** 로 고르고, 디퓨전은 원고 규칙(무작위 t val 손실)으로 고른
`diffusion_vl`도 같은 학습에서 함께 저장한다. `direct`는 디퓨전과 **같은 네트워크(파라미터 수 동일)**를 t = T−1, 입력 E_u에서만 학습한 직접 회귀.
원고용 `weights/diffusion_*`는 읽지도 덮어쓰지도 않는다.
```bash
tmux new -s compare
cd ~/hayeon/ich_cf_clean
SEEDS=42 bash scripts/compare_all.sh medsiglip_after 2>&1 | tee logs/compare_all_$(date +%Y%m%d_%H%M).log   # 1단계
bash scripts/compare_all.sh 2>&1 | tee logs/compare_all_$(date +%Y%m%d_%H%M).log                           # 2단계: 5 인코더 × 6·23·90 view × seed 42·43·44
python collect_restorers.py --B 2000
```
- 참고: mean shift는 선형 프로브의 로짓에 상수만 더하므로 AUROC가 E_u와 정확히 같다(임계값을 고정한 민감도에서만 차이).
- 출력: `compare/results/*.json`, `compare/preds/*.npz` → `results_restorers_{runs,summary,bootstrap}.csv`

### 7) 검사 단위 평가 + 고정 특이도 민감도 (피드백 3.2, 2026-10-26 추가)
```bash
python analysis/dump_preds.py --all --split val
python analysis/study_level_eval.py --pred_dir preds --B 2000 --out_dir .
```
- 출력: `study_auroc_perlabel.csv`, `study_auroc_summary.csv`, `sens_at_spec.csv`
- 민감도는 두 방식: `roc`(각 표현의 test ROC에서 특이도 목표점), `fixed`(val의 E_f로 임계값을 한 번 정해 test의 E_f/E_u/E_r에 그대로 적용, 실제 특이도도 보고)

## 4. 결과 위치 (모두 이 폴더 기준 상대 경로)
- 가중치 `weights/diffusion_{tag}_none_T100.best.pt`, `weights/linprobe_{tag}_none.best.pt`
- 로그 `logs/{tag}_none_{stamp}.log`(학습), `logs/{tag}_none_test_{stamp}.log`(test 평가),
  `logs/{tag}_none_test_steps{S}_{stamp}.log`(단계 수 평가, `collect_test_results.py`는 무시)
- 그림 `plots/` · 요약 CSV `results_test_summary.csv`, `results_test_perlabel.csv`
- tag 예: `medsiglip_after_r360_nv6`, `siglip_window_r360_nv180`
