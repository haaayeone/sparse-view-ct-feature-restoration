# exp_nocond — diffusion 마지막 시도 (A + C 변형)

2026-10-29 작성. 1단계 공정 비교(MedSigLIP, seed 42)에서 MLP·direct가 diffusion보다 높게 나와서
(6-view: mlp 0.846, direct 0.845, diffusion 0.822, diffusion_vl 0.803),
diffusion을 유지할 수 있는지 **한 번만** 확인하는 실험이다. 기존 코드는 건드리지 않고 이 폴더에만 둔다.

## 무엇이 달라지나 (원고 diffusion 대비)

| | 원고 diffusion | 이 변형 |
|---|---|---|
| A. 네트워크 입력 | [x_t ‖ E_u ‖ τ(t)] | [x_t ‖ τ(t)] — E_u 조건 입력 제거 (InDI 방식) |
| C. 섞는 비율 | λ_t = t/T (최대 0.99) | λ_t = t/(T−1) (t = T−1에서 x = E_u 정확히) |
| 손실 | MSE(Δ̂, Δ) | MSE(x_t − λ_t·Δ̂, E_f): E_f 공간 오차 (조건이 없으면 작은 t의 Δ는 알 수 없어서 InDI처럼 clean 예측 오차로 잰다. t = T−1에서는 direct와 같은 목적함수) |
| 샘플링 | 매 단계 E_u를 다시 조건으로 넣음 | E_u는 시작점으로만. Ê_f = x − λ_t·Δ̂ → x ← Ê_f + λ_next·Δ̂ |

네트워크 폭·깊이(SwiGLU 4블록 × 1024)·time embedding·학습 레시피(AdamW 1e-4, 배치 256, 200 에폭, AMP, EMA 0.999)·
고정 프로브·데이터는 모두 같다.

## 미리 정한 판정 기준 (결과 보기 전에 고정)

- 조건: MedSigLIP, 6-view, seed 42, 한 번.
- 에폭 선택: val 1단계 복원 MSE 최소 (다른 방법과 같은 기준). 단계 수 S ∈ {1, 5, 10, 20}은 val 복원 MSE로 하나 고름(S*).
- **성공**: S*에서의 test AUROC가 mlp보다 높고, 검사 단위 bootstrap 95% CI가 0을 넘는다
  (`collect_restorers.py --ref mlp` 출력의 Δ = AUROC(mlp) − AUROC(diffusion_nc…)의 CI 상한 < 0).
- **그 외(비슷하거나 낮음)**: diffusion 개선 시도는 여기서 끝내고, 기여를 "feature-space restoration"(주 모델 MLP)으로 다시 잡는다.

## 실행 (서버)

이 폴더를 서버 `~/hayeon/ich_cf_clean/exp_nocond/`에 그대로 두고 저장소 루트에서 실행한다.

```bash
cd ~/hayeon/ich_cf_clean
python exp_nocond/train_nocond.py configs/medsiglip_after/train_medsiglip_after_nv6.yaml --seed 42
python collect_restorers.py --B 2000 --ref mlp
```

- 결과는 `compare/`에 기존 결과와 같은 형식으로 저장된다: `diffusion_nc`(1단계), `diffusion_nc_5step`, `_10step`, `_20step`.
  로그 끝에 `S* = …`가 찍히고, 판정에는 그 단계 수의 결과를 쓴다.
- 이미 결과가 있으면 건너뛴다. 가중치: `compare/weights/medsiglip_after_r360_nv6__diffusion_nc__s42.pt`.
