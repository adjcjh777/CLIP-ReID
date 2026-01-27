#!/bin/bash
# Train Text-Guided ReID then shutdown
set -e

DATASET=${1:-"market1501"}
GPU_ID=${2:-0}
CAPTION_SOURCE=${3:-"attribute"}  # attribute | blip

OUTPUT_DIR="/root/autodl-tmp/CLIP_REID/OUTPUT/text_guided/${DATASET}"
LOG_FILE="${OUTPUT_DIR}/train_shutdown.log"

mkdir -p "$OUTPUT_DIR"

echo "=== Text-Guided ReID (auto-shutdown) ===" | tee "$LOG_FILE"
echo "Dataset: $DATASET" | tee -a "$LOG_FILE"
echo "GPU: $GPU_ID" | tee -a "$LOG_FILE"
echo "Caption source: $CAPTION_SOURCE" | tee -a "$LOG_FILE"

bash scripts/train_text_guided_pipeline.sh "$DATASET" "$GPU_ID" "$CAPTION_SOURCE" 2>&1 | tee -a "$LOG_FILE"

sync
shutdown -h now
