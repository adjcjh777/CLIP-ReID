#!/bin/bash
# scripts/train_text_guided_pipeline.sh
# 完整的 Text-Guided ReID 训练流程

set -e

DATASET=${1:-"market1501"}
GPU_ID=${2:-0}
CAPTION_SOURCE=${3:-"attribute"}  # attribute | blip
BLIP_MODEL=${BLIP_MODEL:-"Salesforce/blip-image-captioning-base"}
BLIP_BATCH_SIZE=${BLIP_BATCH_SIZE:-8}

OUTPUT_DIR="/root/autodl-tmp/CLIP_REID/OUTPUT/text_guided/${DATASET}"
ANNOTATION_FILE="annotations/${DATASET}_train.json"
ATTR_FILE="datasets/attributes/market1501_attribute.mat"

if [[ "${DATASET,,}" == *"msmt17"* ]]; then
    IMAGE_DIR="/root/autodl-tmp/CLIP_REID/DATASETS/MSMT17/train"
else
    IMAGE_DIR="/root/autodl-tmp/CLIP_REID/DATASETS/Market-1501-v15.09.15/bounding_box_train"
fi

echo "=== Text-Guided ReID Pipeline ==="
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo "GPU: $GPU_ID"
echo "Caption source: $CAPTION_SOURCE"

# 1. 准备数据
if [ "$CAPTION_SOURCE" = "blip" ]; then
    ANNOTATION_FILE="annotations/${DATASET}_train_blip.json"
    if [ ! -f "$ANNOTATION_FILE" ]; then
        echo "Generating BLIP captions..."
        python scripts/generate_captions_blip.py \
            --image_dir "$IMAGE_DIR" \
            --output_file "$ANNOTATION_FILE" \
            --model_name "$BLIP_MODEL" \
            --batch_size "$BLIP_BATCH_SIZE" \
            --device "cuda"
    fi
else
    if [ ! -f "$ATTR_FILE" ]; then
        bash scripts/download_attributes.sh
    fi

    if [ ! -f "$ANNOTATION_FILE" ]; then
        echo "Generating attribute-based annotations..."
        python datasets/attribute_parser.py \
            --attribute_file "$ATTR_FILE" \
            --image_dir "$IMAGE_DIR" \
            --output_file "$ANNOTATION_FILE"
    fi
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
