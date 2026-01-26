#!/bin/bash
# scripts/train_text_guided_msmt17.sh
# MSMT17 上的 Text-Guided ReID 训练

set -e

GPU_ID=${1:-0}
OUTPUT_DIR="/root/autodl-tmp/CLIP_REID/OUTPUT/text_guided/msmt17"
ANNOTATION_FILE="annotations/msmt17_train.json"

echo "=== Text-Guided ReID on MSMT17 ==="
echo "Output: $OUTPUT_DIR"
echo "GPU: $GPU_ID"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 训练 (Stage 1 + Stage 2)
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_text_reid.py \
    --config_file configs/person/vit_clipreid.yml \
    --annotation_file "$ANNOTATION_FILE" \
    DATASETS.NAMES "('msmt17')" \
    DATASETS.ROOT_DIR "/root/autodl-tmp/CLIP_REID/DATASETS" \
    OUTPUT_DIR "$OUTPUT_DIR" \
    SOLVER.STAGE1.IMS_PER_BATCH 32 \
    SOLVER.STAGE1.MAX_EPOCHS 60 \
    SOLVER.STAGE2.IMS_PER_BATCH 32 \
    SOLVER.STAGE2.MAX_EPOCHS 60 \
    MODEL.DEVICE_ID "('${GPU_ID}')" \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "Training completed."
