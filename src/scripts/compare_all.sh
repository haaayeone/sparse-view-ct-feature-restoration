#!/usr/bin/env bash
# compare_all.sh — 복원기 공정 비교 + 학습 seed 반복 (피드백 1.2·2.3): 인코더 × view × seed 전부 compare_restorers.py
#
# 기본: 5 인코더 × 6·23·90 view × seed 42·43·44 × 방법 7개(identity meanshift coral ridge diffusion direct mlp).
# config 하나마다 데이터를 한 번만 읽고 그 config의 모든 seed·방법을 돌린다. 결과는 compare/ 에만 쓰며
# 원고용 가중치(weights/diffusion_*)는 건드리지 않는다. 끝난 (tag, 방법, seed)는 건너뛰므로 끊겨도 같은 명령으로 이어서 실행.
#
# 실행 위치: 서버 (GPU). 오래 걸리므로 tmux 안에서. 필요한 것: 각 config의 프로브 weights/linprobe_{tag}_none.best.pt (이미 있음)
# Usage (서버):
#   tmux new -s compare
#   conda activate hayeon
#   cd ~/hayeon/ich_cf_clean
#   SEEDS=42 bash scripts/compare_all.sh medsiglip_after 2>&1 | tee logs/compare_all_$(date +%Y%m%d_%H%M).log
#     ↑ 1단계: MedSigLIP만, seed 42 (3 view × 학습 3회). 결과 보고 원고 방향 결정
#   bash scripts/compare_all.sh 2>&1 | tee logs/compare_all_$(date +%Y%m%d_%H%M).log
#     ↑ 2단계: 전체 (이미 끝난 것은 건너뜀)
#   GPU가 여러 개면 인코더를 나눠 다른 tmux 창에서 동시에:
#     CUDA_VISIBLE_DEVICES=1 bash scripts/compare_all.sh siglip dinov2
#   환경변수: VIEWS="6 23 90"  SEEDS="42 43 44"  METHODS="identity meanshift coral ridge diffusion direct mlp"  EXTRA="--save_weights"
#   끝난 뒤:  python collect_restorers.py --B 2000

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

if [ $# -gt 0 ]; then ENCODERS="$*"; else ENCODERS="medsiglip_after siglip dinov2 resnet50 densenet121"; fi
VIEWS="${VIEWS:-6 23 90}"
SEEDS="${SEEDS:-42 43 44}"
METHODS="${METHODS:-identity meanshift coral ridge diffusion direct mlp}"
EXTRA="${EXTRA:-}"

cfg_path() {   # scripts/train_all.sh 와 동일
  if [ "$1" = "medsiglip_after" ]; then echo "configs/medsiglip_after/train_medsiglip_after_nv$2.yaml"
  else echo "configs/$1/train_$1_window_nv$2.yaml"; fi
}

for E in $ENCODERS; do
  for V in $VIEWS; do
    CFG=$(cfg_path "$E" "$V")
    [ -f "$CFG" ] || { echo "!! config 없음: $CFG" >&2; exit 1; }
    echo "==================== compare $CFG  seeds=[$SEEDS]  $(date '+%F %T')"
    # shellcheck disable=SC2086
    python compare_restorers.py "$CFG" --seeds $SEEDS --methods $METHODS $EXTRA
  done
done
echo "==================== done  $(date '+%F %T')"
python collect_restorers.py
