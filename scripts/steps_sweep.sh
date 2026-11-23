#!/usr/bin/env bash
# steps_sweep.sh — 역방향 샘플링 단계 수(1, 5, 10, 20)에 따른 test AUROC (원고 표 4).
# 학습된 가중치를 그대로 쓰고 평가만 다시 한다 (run_eval.py --steps S).
# 로그: logs/{tag}_none_test_steps{S}_{stamp}.log  → 기본 test 결과(collect_test_results.py)와 섞이지 않음.
#
# 실행 위치: 서버. 해당 인코더의 train_all.sh 가 끝난 뒤 (weights 필요).
# Usage (서버):
#   cd ~/hayeon/ich_cf_clean
#   bash scripts/steps_sweep.sh                         # 기본: medsiglip_after, 6·12 view, 1·5·10·20 단계
#   bash scripts/steps_sweep.sh siglip                  # 다른 인코더
#   VIEWS="6 12 23" STEPS="1 5 10 20" bash scripts/steps_sweep.sh medsiglip_after
#   끝나면 자동으로: python collect_steps_results.py  → results_steps.csv

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
E="${1:-medsiglip_after}"
VIEWS="${VIEWS:-6 12}"
STEPS="${STEPS:-1 5 10 20}"

for V in $VIEWS; do
  if [ "$E" = "medsiglip_after" ]; then CFG="configs/medsiglip_after/train_medsiglip_after_nv${V}.yaml"
  else CFG="configs/$E/train_${E}_window_nv${V}.yaml"; fi
  [ -f "$CFG" ] || { echo "!! config 없음: $CFG" >&2; exit 1; }
  for S in $STEPS; do
    echo "==================== $E nv$V  steps=$S  $(date '+%F %T')"
    python run_eval.py "$CFG" --split test --steps "$S"
  done
done
python collect_steps_results.py
