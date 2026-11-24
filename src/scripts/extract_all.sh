#!/usr/bin/env bash
# extract_all.sh — 5개 인코더 × sparse 5조건(nv90, nv45, nv23, nv12, nv6) feature 추출.
#
# 2026-09-30: view 선택 규칙 수정(sparse_view_indices, [0°,180°) 균등 분할) 후 sparse만 다시 추출한다.
#   출력: /data0/hayeon/features_v2/<인코더>/feature_nv*   (기존 /data0/hayeon/features/ 는 건드리지 않음)
#   full(180 view)은 모든 각도를 쓰므로 규칙 수정과 무관 → 기존 feature_full 그대로 사용.
# 최종 전처리(모든 인코더 동일): DICOM → HU → [−1024,2048]→[0,1] → 256×256 → Radon(180각) → view 선택
#   → FBP(ramp, 음수만 clip) → HU 복원 → 윈도우 brain 40/80, subdural 80/200, bone 600/2800 → RGB → 인코더.
# 이미 저장된 .npy는 건너뛰므로 끊겨도 같은 명령으로 이어서 실행하면 된다.
#
# 실행 위치: 서버(GPU). 조건 하나에 약 4–5시간, 전체 25조건 → tmux 필수.
# 먼저 용량 확인 (25조건 × 약 1.7 GB ≈ 45 GB):   df -h /data0
# Usage (서버):
#   tmux new -s extract_v2
#   cd ~/hayeon/ich_cf_clean
#   bash scripts/extract_all.sh 2>&1 | tee logs/extract_all_$(date +%Y%m%d_%H%M).log
#   특정 인코더만:   bash scripts/extract_all.sh medsiglip_after siglip
#   (이름: medsiglip_after siglip dinov2 resnet50 densenet121)
#   full까지 다시 뽑으려면(보통 불필요):  VIEWS="full nv90 nv45 nv23 nv12 nv6" bash scripts/extract_all.sh

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

if [ $# -gt 0 ]; then ENCODERS="$*"; else ENCODERS="medsiglip_after siglip dinov2 resnet50 densenet121"; fi
VIEWS="${VIEWS:-nv90 nv45 nv23 nv12 nv6}"

for E in $ENCODERS; do
  for V in $VIEWS; do
    if [ "$E" = "medsiglip_after" ]; then CFG="configs/medsiglip_after/extract_medsiglip_after_${V}.yaml"
    else CFG="configs/$E/extract_${E}_window_${V}.yaml"; fi
    [ -f "$CFG" ] || { echo "!! config 없음: $CFG" >&2; exit 1; }
    echo "==================== extract $E $V  $(date '+%F %T')"
    python extraction_3ch/run_extract_encoder_mp.py "$CFG"
  done
done
echo "==================== done  $(date '+%F %T')"
