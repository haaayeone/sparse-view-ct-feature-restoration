# exp_probe — 프로브를 무엇으로 학습하느냐에 따른 비교 (E_f / E_u / E_r)

2026-10-30 작성. 10/28 미팅 할 일 1 ("restored로 probe 학습 후 비교")과,
"E_f 프로브에 E_u를 넣었을 때 0.58인 건 정보가 없어서인가, 형태가 달라서인가"를 확인하는 실험이다.
기존 코드와 원고 프로브(`weights/linprobe_*.best.pt`)는 건드리지 않는다.

## 무엇을 하나

원고 프로브와 **똑같은 선형 프로브·학습 레시피**로, 학습에 쓰는 표현만 바꿔 새 프로브를 학습한다.
학습 레시피는 `train_diffusion.train_stage2`와 같다: AdamW 1e-4/1e-4, 배치 256, 최대 200 에폭, 20 에폭마다 val AUROC로 저장, 40 에폭 이후 개선이 없으면 멈춘다.

| 프로브 | 학습·선택에 쓰는 표현 |
|---|---|
| P_f (기존) | E_f |
| P_u | E_u (희소 투영 그대로) |
| P_r_mlp | MLP로 복원한 E_r |
| P_r_diffusion | diffusion으로 복원한 E_r (복원 MSE 선택 체크포인트) |

모든 프로브를 모든 test 표현(E_f, E_u, E_r)에 적용해 행렬로 보고한다.
검정(bootstrap, Holm)은 주요 조건만 한다: 대각선(같은 표현으로 학습·평가)과 P_f 행. 기준은 `Pf_on_r_mlp`(E_f 프로브 + MLP 복원)이다.

## 읽는 법 (결과 보기 전에 적어 둠)

| 결과 | 뜻 |
|---|---|
| P_u(E_u) ≈ P_f(E_r) | 선형 분류기도 E_u에서 정보를 바로 읽는다. 복원의 가치는 정보를 늘리는 것이 아니라, **E_f로 학습한 분류기 하나를 모든 view에 그대로 쓰게 해 주는 것**(희소 투영 데이터의 라벨 불필요)이다 |
| P_u(E_u) < P_f(E_r) | 정보는 E_u에 있지만 선형으로는 못 읽는다. 복원(비선형 MLP)이 **선형으로 읽을 수 있는 정보**를 되살린다 (usable information, Xu et al. 2020) |
| P_r(E_r) > P_u(E_u) | 라벨이 있어도 "복원 후 학습"이 "E_u로 바로 학습"보다 낫다 |

- E_r은 E_u만으로 계산하므로 E_r의 정보량은 E_u를 넘을 수 없다(데이터 처리 부등식). 달라질 수 있는 것은 선형 분류기가 읽을 수 있는 정보뿐이다.
- 공정성: P_u·P_r은 희소 투영 데이터의 라벨로 학습하지만, 복원기는 라벨 없이 (E_u, E_f) 쌍만으로 학습한다.

## 실행 (서버, `~/hayeon/ich_cf_clean`)

필요한 것: 원고 프로브 `weights/linprobe_{tag}_none.best.pt`, 복원기 가중치 `compare/weights/{tag}__mlp__s42.pt`·`__diffusion__s42.pt`.

먼저 6-view만 (약 20–30분: 데이터 로딩 약 10분 + 프로브 3개 학습):

```bash
python exp_probe/train_probe_variants.py --views 6 --B 2000
```

세 view 모두:

```bash
python exp_probe/train_probe_variants.py --views 6 23 90 --B 2000
```

- 끊기면 같은 명령으로 이어서 실행된다 (결과와 학습한 프로브가 있으면 건너뜀).

결과 가져오기 (로컬 PowerShell):

```bash
scp neumous-lab@100.73.97.101:/home/neumous-lab/hayeon/ich_cf_clean/results_probe_*.csv "C:\Users\user\Desktop\MIGT\01. sparse ct project\result"
```

## 출력

| 파일 | 내용 |
|---|---|
| `results_probe_matrix.csv` | view × 프로브(학습 표현) 행, test 표현 열: 평균 AUROC |
| `results_probe_bootstrap.csv` | 주요 조건의 AUROC 95% CI, Δ = AUROC(Pf_on_r_mlp) − AUROC(조건), Holm |
| `compare_probe/` | 조건별 json·예측(npz)·학습한 프로브·로그 |
