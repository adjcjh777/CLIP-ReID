#!/usr/bin/env bash
# =======================================================================
# B4 最优组合实验
#
# 当前采用最小成本可复现组合：
#   TextQueryEncoder + I2T_LOSS_WEIGHT=0.2 + 无额外 text loss
#
# 用法：
#   bash scripts/run_b4_best_combo.sh
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
OUT="${OUTPUT_ROOT}/phase2/B4_best_combo"
S1_CKPT="${OUTPUT_ROOT}/phase2/B1_query_encoder/ViT-B-16_stage1_60.pth"

cd "$PROJECT_DIR"
mkdir -p "$OUT"

if [ ! -f "$S1_CKPT" ]; then
  echo "ERROR: missing stage1 checkpoint: $S1_CKPT"
  exit 1
fi

echo "[INFO] Running B4 with query_encoder + I2T=0.2"

$PYTHON train_text_reid.py \
  --config_file $CONFIG \
  --annotation_file $ANNOTATION \
  --skip_stage1 \
  --stage1_checkpoint "$S1_CKPT" \
  DATASETS.NAMES market1501 \
  DATASETS.ROOT_DIR "$DATASETS_ROOT" \
  SOLVER.SEED 1234 \
  SOLVER.STAGE1.IMS_PER_BATCH 32 \
  SOLVER.STAGE1.MAX_EPOCHS 60 \
  SOLVER.STAGE2.IMS_PER_BATCH 32 \
  SOLVER.STAGE2.MAX_EPOCHS 60 \
  OUTPUT_DIR "$OUT" \
  MODEL.TEXT_ENCODER_TYPE query_encoder \
  MODEL.TEXT_LOSS_TYPE none \
  MODEL.I2T_LOSS_WEIGHT 0.2 \
  INPUT.SIZE_TRAIN "[256, 128]" \
  INPUT.SIZE_TEST "[256, 128]"
