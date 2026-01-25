#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/root/autodl-tmp/CLIP_REID/DATASETS"
OUT_BASE="/root/autodl-tmp/CLIP_REID/OUTPUT/multi_granularity"

mkdir -p "${OUT_BASE}/market1501"
mkdir -p "${OUT_BASE}/msmt17"

# Market-1501
CUDA_VISIBLE_DEVICES=0 python train_clipreid.py \
  --config_file configs/person/vit_clipreid.yml \
  MODEL.MULTI_GRANULARITY.ENABLED True \
  MODEL.MULTI_GRANULARITY.NUM_PARTS 4 \
  MODEL.MULTI_GRANULARITY.PART_DIM 512 \
  TENSORBOARD.ENABLED True \
  TENSORBOARD.LOG_DIR "${OUT_BASE}/tensorboard/market1501" \
  DATASETS.NAMES '("market1501")' \
  DATASETS.ROOT_DIR "${DATA_ROOT}" \
  OUTPUT_DIR "${OUT_BASE}/market1501"

# MSMT17
CUDA_VISIBLE_DEVICES=0 python train_clipreid.py \
  --config_file configs/person/vit_clipreid.yml \
  MODEL.MULTI_GRANULARITY.ENABLED True \
  MODEL.MULTI_GRANULARITY.NUM_PARTS 4 \
  MODEL.MULTI_GRANULARITY.PART_DIM 512 \
  TENSORBOARD.ENABLED True \
  TENSORBOARD.LOG_DIR "${OUT_BASE}/tensorboard/msmt17" \
  DATASETS.NAMES '("msmt17")' \
  DATASETS.ROOT_DIR "${DATA_ROOT}" \
  OUTPUT_DIR "${OUT_BASE}/msmt17"

echo "Training finished. Shutting down in 3 minutes..."
if command -v shutdown >/dev/null 2>&1; then
  shutdown -h +3
elif command -v poweroff >/dev/null 2>&1; then
  sleep 180
  poweroff
else
  echo "No shutdown command found. Please shut down manually."
fi
