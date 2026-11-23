#!/usr/bin/env bash
# extract_ref360.sh — 기준(reference) 360 view feature 추출, 5개 인코더.
#
# 실험 조건 (2026-10-01 확정): 기준 360 view, sparse 180·90·45·23·12·6 view.
#   ref360 : 이 스크립트 → /data0/hayeon/features_v2/<인코더>/feature_ref360
#   nv90–6 : scripts/extract_all.sh (진행 중인 것 그대로 사용 — 1° 격자 각도는 0.5° 격자에 포함되고
#            Radon 투영은 각도별로 독립이라 기준을 바꿔도 영상이 같다)
#   nv180  : 예전 기준 /data0/hayeon/features/<인코더>/feature_full 을 그대로 사용 (1° 격자의 모든 각도)
# 이미 저장된 .npy는 건너뛰므로 끊겨도 같은 명령으로 이어서 실행하면 된다.
#
# 실행 위치: 서버(GPU). extract_all.sh 가 돌고 있는 중이면 다른 tmux 창에서 함께 돌려도 된다
#           (CPU를 나눠 쓰므로 둘 다 느려지지만 전체 소요 시간은 비슷).
# ※ 실행 중인 extract_all.sh 파일은 덮어쓰지 말 것 (bash는 실행 중 스크립트를 읽어가며 실행함).
# Usage (서버):
#   tmux new -s ref360
#   cd ~/hayeon/ich_cf_clean
#   bash scripts/extract_ref360.sh 2>&1 | tee logs/extract_ref360_$(date +%Y%m%d_%H%M).log
#   특정 인코더만:   bash scripts/extract_ref360.sh medsiglip_after
#   (이름: medsiglip_after siglip dinov2 resnet50 densenet121)

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

if [ $# -gt 0 ]; then ENCODERS="$*"; else ENCODERS="medsiglip_after siglip dinov2 resnet50 densenet121"; fi

for E in $ENCODERS; do
  if [ "$E" = "medsiglip_after" ]; then CFG="configs/medsiglip_after/extract_medsiglip_after_ref360.yaml"
  else CFG="configs/$E/extract_${E}_window_ref360.yaml"; fi
  [ -f "$CFG" ] || { echo "!! config 없음: $CFG" >&2; exit 1; }
  echo "==================== extract $E ref360  $(date '+%F %T')"
  python extraction_3ch/run_extract_encoder_mp.py "$CFG"
done
echo "==================== done  $(date '+%F %T')"
