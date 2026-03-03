#!/usr/bin/env bash
# =======================================================================
# Final 配置多 seed 验证（D2）
#
# 用法：
#   bash scripts/run_multiseed_final.sh 1234
#   bash scripts/run_multiseed_final.sh all
# =======================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PROJECT_DIR}/.venv-5090/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python"
fi
CONFIG="configs/person/vit_clipreid.yml"
ANNOTATION="annotations/market1501_train.json"
DATASETS_ROOT="${CLIPREID_DATA_ROOT:-${PROJECT_DIR}/DATASETS}"
OUTPUT_ROOT="${CLIPREID_OUTPUT_ROOT:-${PROJECT_DIR}/OUTPUT}"
OUTPUT_BASE="${OUTPUT_ROOT}/final_multiseed"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_BASE"

run_seed() {
  local seed="$1"
  local out="${OUTPUT_BASE}/seed_${seed}"
  echo "[INFO] Running final config with seed=${seed}"

  $PYTHON train_text_reid.py \
    --config_file $CONFIG \
    --annotation_file $ANNOTATION \
    DATASETS.NAMES market1501 \
    DATASETS.ROOT_DIR "$DATASETS_ROOT" \
    SOLVER.SEED "$seed" \
    SOLVER.STAGE1.IMS_PER_BATCH 32 \
    SOLVER.STAGE1.MAX_EPOCHS 60 \
    SOLVER.STAGE1.CHECKPOINT_PERIOD 60 \
    SOLVER.STAGE2.IMS_PER_BATCH 32 \
    SOLVER.STAGE2.MAX_EPOCHS 60 \
    OUTPUT_DIR "$out" \
    MODEL.TEXT_ENCODER_TYPE clip_native \
    MODEL.TEXT_LOSS_TYPE none \
    MODEL.I2T_LOSS_WEIGHT 0.2 \
    MODEL.SIE_CAMERA True \
    MODEL.SIE_COE 3.0 \
    MODEL.STRIDE_SIZE "[12, 12]" \
    INPUT.SIZE_TRAIN "[256, 128]" \
    INPUT.SIZE_TEST "[256, 128]"
}

STEP="${1:-all}"
case "$STEP" in
  1234|777|4321)
    run_seed "$STEP"
    ;;
  all)
    run_seed 1234
    run_seed 777
    run_seed 4321
    ;;
  *)
    echo "Usage: $0 [1234|777|4321|all]"
    exit 1
    ;;
esac
