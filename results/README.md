# results/

논문의 각 표를 만든 수치입니다. 모든 AUROC는 테스트 분할 기준이고,
신뢰구간은 검사(study) 단위 paired bootstrap 2,000회에서 나왔습니다.

| 파일 | 논문 | 내용 |
|---|---|---|
| `table2_summary.csv` | Table 2, Fig. 3 | 인코더 x 투영 수 평균 AUROC (E_f / E_u / E_r), Δ, R |
| `table3_tableS1_perlabel.csv` | Table 3, Table S1 | 라벨별 AUROC, Δ, 95% CI, Holm 보정 p |
| `table4_restorers.csv` | Table 4 | 복원기 비교, seed 42/43/44 평균 ± 표준편차 |
| `table4_restorers_runs.csv` | Table 4 | seed별 개별 실행값 |
| `table4_restorers_bootstrap.csv` | Table 4 | 복원기 간 차이의 CI 와 Holm p |
| `table5_study_auroc.csv` | Table 5 | 검사 단위 AUROC (슬라이스 확률 최댓값으로 집계) |
| `table5_study_auroc_perlabel.csv` | Table 5 | 위의 라벨별 분해 |
| `table6_probe_matrix.csv` | Table 6 | 프로브를 무엇으로 학습했는지에 따른 행렬 (고정 / 재학습 / 복원) |
| `table6_probe_bootstrap.csv` | Table 6 | 위 조건들의 CI 와 Holm p |
| `tableS2_sens_at_spec.csv` | Table S2 | 고정 동작점에서의 민감도·특이도 (목표 특이도 0.90 / 0.95) |
| `raw_test_perlabel.csv` | — | 부트스트랩 이전의 원시 라벨별 AUROC |
| `raw_test_summary.csv` | — | 부트스트랩 이전의 원시 요약 |
| `steps_sweep.csv` | §3.4 | 확산 복원기의 샘플링 단계 수 비교 |
| `per_condition_json/` | — | 조건별 원본 JSON (지표, 학습 곡선, 선택된 epoch) |
| `per_condition_probe_json/` | Table 6 | 프로브 변형 실험의 조건별 JSON |

## 주의

`table3_tableS1_perlabel.csv` 는 **MLP 복원기 기준**입니다. 과거 확산 복원기 기준으로
만든 동명 파일이 저장소에 함께 있었으나 혼동을 피하기 위해 제거했습니다.
논문 Table 3 과 일치하는 것은 이 파일입니다.
