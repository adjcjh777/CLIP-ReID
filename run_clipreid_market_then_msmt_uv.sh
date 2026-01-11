#!/usr/bin/env bash
set -e

#####################################
# Auto shutdown on exit (safe)
#####################################
trap 'echo "[INFO] All tasks finished or aborted. Shutting down in 60s..."; sleep 60; sudo shutdown -h now' EXIT

echo "======================================="
echo " CLIP-ReID Sequential Runner (uv)"
echo " Market1501  ->  MSMT17"
echo "======================================="

#####################################
# Paths (STRICTLY match your structure)
#####################################
PROJECT_ROOT=$(pwd)
DATA_ROOT="/root/autodl-tmp/CLIP_REID/DATASETS"
OUTPUT_ROOT="/root/autodl-tmp/CLIP_REID/OUTPUT"
CONFIG_FILE="configs/person/vit_clipreid.yml"

cd ${PROJECT_ROOT}

echo "[INFO] Python:"
uv run python -V
nvidia-smi || true

#####################################
# Function: train + test
#####################################
run_dataset () {
  DATASET_NAME=$1
  OUTPUT_DIR=$2

  echo "---------------------------------------"
  echo "[INFO] Running dataset: ${DATASET_NAME}"
  echo "---------------------------------------"

  mkdir -p ${OUTPUT_DIR}

  echo "[INFO] Training ${DATASET_NAME}..."
  CUDA_VISIBLE_DEVICES=0 \
  uv run python train_clipreid.py \
    --config_file ${CONFIG_FILE} \
    DATASETS.NAMES "('${DATASET_NAME}')" \
    DATASETS.ROOT_DIR ${DATA_ROOT} \
    OUTPUT_DIR ${OUTPUT_DIR} \
    2>&1 | tee ${OUTPUT_DIR}/train.log

  if [ ! -f "${OUTPUT_DIR}/model_best.pth" ]; then
    echo "[ERROR] model_best.pth not found for ${DATASET_NAME}"
    exit 1
  fi

  echo "[INFO] Testing ${DATASET_NAME}..."
  CUDA_VISIBLE_DEVICES=0 \
  uv run python test_clipreid.py \
    --config_file ${CONFIG_FILE} \
    TEST.WEIGHT ${OUTPUT_DIR}/model_best.pth \
    DATASETS.NAMES "('${DATASET_NAME}')" \
    DATASETS.ROOT_DIR ${DATA_ROOT} \
    2>&1 | tee ${OUTPUT_DIR}/test.log
}

#####################################
# 1️⃣ Market1501
#####################################
run_dataset "market1501" "${OUTPUT_ROOT}/vit_clipreid_market1501"

#####################################
# 2️⃣ MSMT17
#####################################
run_dataset "msmt17" "${OUTPUT_ROOT}/vit_clipreid_msmt17"

#####################################
# Finish
#####################################
echo "======================================="
echo " All experiments finished successfully "
echo "======================================="

##关机##
echo "[INFO] Training finished, shutting down in 1 minute..."
sleep 60
sudo shutdown -h now
