#!/bin/bash
# scripts/train_text_guided_pipeline.sh
# 完整的 Text-Guided ReID 训练流程

set -e

DATASET=${1:-"market1501"}
GPU_ID=${2:-0}

OUTPUT_DIR="/root/autodl-tmp/CLIP_REID/OUTPUT/text_guided/${DATASET}"
ANNOTATION_FILE="annotations/${DATASET}_train.json"
ATTR_FILE="datasets/attributes/market1501_attribute.mat"

echo "=== Text-Guided ReID Pipeline ==="
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo "GPU: $GPU_ID"

# 1. 准备数据
if [ ! -f "$ATTR_FILE" ]; then
    bash scripts/download_attributes.sh
fi

if [ ! -f "$ANNOTATION_FILE" ]; then
    echo "Generating text annotations..."
    python datasets/attribute_parser.py \
        --attribute_file "$ATTR_FILE" \
        --image_dir "/root/autodl-tmp/CLIP_REID/DATASETS/Market-1501-v15.09.15/bounding_box_train" \
        --output_file "$ANNOTATION_FILE"
fi

# 2. 训练
echo "Starting training..."
mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=${GPU_ID} python train_text_reid.py \
    --config_file configs/person/vit_clipreid.yml \
    --annotation_file "$ANNOTATION_FILE" \
    DATASETS.NAMES "('${DATASET}')" \
    DATASETS.ROOT_DIR "/root/autodl-tmp/CLIP_REID/DATASETS" \
    OUTPUT_DIR "$OUTPUT_DIR" \
    SOLVER.STAGE1.IMS_PER_BATCH 32 \
    SOLVER.STAGE1.MAX_EPOCHS 60 \
    SOLVER.STAGE2.IMS_PER_BATCH 32 \
    SOLVER.STAGE2.MAX_EPOCHS 60 \
    MODEL.DEVICE_ID "('${GPU_ID}')" \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "Training completed."
echo "Shutting down system..."
shutdown -h now
