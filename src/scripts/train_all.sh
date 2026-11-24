#!/usr/bin/env bash
# train_all.sh — 5개 인코더 × 6개 sparse 조건: 확산 모델 학습(train_diffusion.py) → test 평가(run_eval.py)
#
# 실험 조건: 기준(reference) 360 view, sparse 180·90·45·23·12·6 view.  tag = <인코더>_r360_nv<V>
# 필요한 feature (인코더별):
#   기준  /data0/hayeon/features_v2/<인코더>/feature_ref360   ← scripts/extract_ref360.sh
#   sparse /data0/hayeon/features_v2/<인코더>/feature_nv{90,45,23,12,6}  ← scripts/extract_all.sh
#          /data0/hayeon/features/<인코더>/feature_full (= 180 view)
# 두 추출이 해당 인코더에 대해 모두 끝난 뒤 실행할 것 (폴더가 없으면 그 조건에서 멈춤).
# 이미 test 로그가 있는 조건은 건너뛰므로, 끊겨도 같은 명령으로 이어서 실행하면 된다.
#
# 실행 위치: 서버. 오래 걸리므로 tmux 안에서 실행.
# Usage (서버):
#   tmux new -s train_all
#   cd ~/hayeon/ich_cf_clean
#   bash scripts/train_all.sh 2>&1 | tee logs/train_all_$(date +%Y%m%d_%H%M).log
#
#   특정 인코더만:   bash scripts/train_all.sh medsiglip_after siglip
#   (이름: medsiglip_after siglip dinov2 resnet50 densenet121)
#   끝난 뒤 결과 모으기:  python collect_test_results.py

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

if [ $# -gt 0 ]; then ENCODERS="$*"; else ENCODERS="medsiglip_after siglip dinov2 resnet50 densenet121"; fi
VIEWS="${VIEWS:-180 90 45 23 12 6}"

cfg_path() {   # $1=encoder $2=views
  if [ "$1" = "medsiglip_after" ]; then echo "configs/medsiglip_after/train_medsiglip_after_nv$2.yaml"
  else echo "configs/$1/train_$1_window_nv$2.yaml"; fi
}
cfg_get() {    # $1=config $2=key
  grep -E "^$2:" "$1" | awk '{print $2}'
}

for E in $ENCODERS; do
  for V in $VIEWS; do
    CFG=$(cfg_path "$E" "$V")
    [ -f "$CFG" ] || { echo "!! config 없음: $CFG" >&2; exit 1; }
    TAG=$(cfg_get "$CFG" tag)
    if ls logs/${TAG}_none_test_*.log >/dev/null 2>&1; then
      echo "---- skip $TAG (test 로그 있음)"; continue
    fi
    for D in "$(cfg_get "$CFG" full_feature_dir)" "$(cfg_get "$CFG" under_feature_dir)"; do
      [ -d "$D/test" ] || { echo "!! feature 폴더 없음: $D  (추출이 끝났는지 확인)" >&2; exit 1; }
    done
    echo "==================== train $TAG  $(date '+%F %T')"
    python train_diffusion.py "$CFG"
    echo "==================== eval  $TAG (test)  $(date '+%F %T')"
    python run_eval.py "$CFG" --split test
  done
done
echo "==================== done  $(date '+%F %T')"
python collect_test_results.py
