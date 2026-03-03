#!/bin/bash
# scripts/train_text_guided_msmt17.sh
# MSMT17 上的 Text-Guided ReID 训练

set -e

GPU_ID=${1:-0}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${CLIPREID_DATA_ROOT:-${PROJECT_DIR}/DATASETS}"
OUTPUT_ROOT="${CLIPREID_OUTPUT_ROOT:-${PROJECT_DIR}/OUTPUT}"
OUTPUT_DIR="${OUTPUT_ROOT}/text_guided/msmt17"
ANNOTATION_FILE="${PROJECT_DIR}/annotations/msmt17_train.json"
PYTHON="${PROJECT_DIR}/.venv-5090/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python"
fi

echo "=== Text-Guided ReID on MSMT17 ==="
echo "Output: $OUTPUT_DIR"
echo "GPU: $GPU_ID"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR"

# 训练 (Stage 1 + Stage 2)
CUDA_VISIBLE_DEVICES=${GPU_ID} "$PYTHON" train_text_reid.py \
    --config_file configs/person/vit_clipreid.yml \
    --annotation_file "$ANNOTATION_FILE" \
    DATASETS.NAMES "('msmt17')" \
    DATASETS.ROOT_DIR "$DATA_ROOT" \
    OUTPUT_DIR "$OUTPUT_DIR" \
    SOLVER.STAGE1.IMS_PER_BATCH 32 \
    SOLVER.STAGE1.MAX_EPOCHS 60 \
    SOLVER.STAGE2.IMS_PER_BATCH 32 \
    SOLVER.STAGE2.MAX_EPOCHS 60 \
    MODEL.DEVICE_ID "('${GPU_ID}')" \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "Training completed."
