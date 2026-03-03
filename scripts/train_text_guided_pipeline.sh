#!/bin/bash
# scripts/train_text_guided_pipeline.sh
# 完整的 Text-Guided ReID 训练流程

set -e

DATASET=${1:-"market1501"}
GPU_ID=${2:-0}
CAPTION_SOURCE=${3:-"attribute"}  # attribute | blip
BLIP_MODEL=${BLIP_MODEL:-"Salesforce/blip-image-captioning-base"}
BLIP_BATCH_SIZE=${BLIP_BATCH_SIZE:-8}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${CLIPREID_DATA_ROOT:-${PROJECT_DIR}/DATASETS}"
OUTPUT_ROOT="${CLIPREID_OUTPUT_ROOT:-${PROJECT_DIR}/OUTPUT}"
OUTPUT_DIR="${OUTPUT_ROOT}/text_guided/${DATASET}"
ANNOTATION_FILE="${PROJECT_DIR}/annotations/${DATASET}_train.json"
ATTR_FILE="${PROJECT_DIR}/datasets/attributes/market1501_attribute.mat"
PYTHON="${PROJECT_DIR}/.venv-5090/bin/python"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python"
fi

if [[ "${DATASET,,}" == *"msmt17"* ]]; then
    IMAGE_DIR="${DATA_ROOT}/MSMT17/train"
else
    IMAGE_DIR="${DATA_ROOT}/Market-1501-v15.09.15/bounding_box_train"
fi

echo "=== Text-Guided ReID Pipeline ==="
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo "GPU: $GPU_ID"
echo "Caption source: $CAPTION_SOURCE"
echo "Data root: $DATA_ROOT"

cd "$PROJECT_DIR"

# 1. 准备数据
if [ "$CAPTION_SOURCE" = "blip" ]; then
    ANNOTATION_FILE="${PROJECT_DIR}/annotations/${DATASET}_train_blip.json"
    if [ ! -f "$ANNOTATION_FILE" ]; then
        echo "Generating BLIP captions..."
        "$PYTHON" scripts/generate_captions_blip.py \
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
        "$PYTHON" datasets/attribute_parser.py \
            --attribute_file "$ATTR_FILE" \
            --image_dir "$IMAGE_DIR" \
            --output_file "$ANNOTATION_FILE"
    fi
fi

# 2. 训练
echo "Starting training..."
mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=${GPU_ID} "$PYTHON" train_text_reid.py \
    --config_file configs/person/vit_clipreid.yml \
    --annotation_file "$ANNOTATION_FILE" \
    DATASETS.NAMES "('${DATASET}')" \
    DATASETS.ROOT_DIR "$DATA_ROOT" \
    OUTPUT_DIR "$OUTPUT_DIR" \
    SOLVER.STAGE1.IMS_PER_BATCH 32 \
    SOLVER.STAGE1.MAX_EPOCHS 60 \
    SOLVER.STAGE2.IMS_PER_BATCH 32 \
    SOLVER.STAGE2.MAX_EPOCHS 60 \
    MODEL.DEVICE_ID "('${GPU_ID}')" \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "Training completed."
