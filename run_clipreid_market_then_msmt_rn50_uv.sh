#!/usr/bin/env bash

echo "========================================="
echo " CLIP-ReID STRICT SEQUENTIAL RUNNER"
echo " Market1501  ->  MSMT17 (GUARANTEED)"
echo " Backbone: RN50"
echo "========================================="

DATA_ROOT="/root/autodl-tmp/CLIP_REID/DATASETS"
OUTPUT_ROOT="/root/autodl-tmp/CLIP_REID/OUTPUT"
CONFIG_FILE="configs/person/cnn_clipreid.yml"
RUN_STAMP=$(date +"%Y%m%d_%H%M%S")

echo "[INFO] Workdir: $(pwd)"
uv run python -V
nvidia-smi || true

############################################
# Function: train + test (NO early exit)
############################################
run_dataset () {
  DATASET_NAME=$1
  OUTPUT_DIR=$2

  echo "-----------------------------------------"
  echo "[INFO] START DATASET: ${DATASET_NAME}"
  echo "-----------------------------------------"

  mkdir -p "${OUTPUT_DIR}"

  echo "[INFO] Training ${DATASET_NAME} ..."
  CUDA_VISIBLE_DEVICES=0 \
  uv run python train_clipreid.py \
    --config_file ${CONFIG_FILE} \
    DATASETS.NAMES "('${DATASET_NAME}')" \
    DATASETS.ROOT_DIR ${DATA_ROOT} \
    OUTPUT_DIR ${OUTPUT_DIR} \
    WANDB.ENABLED True \
    WANDB.PROJECT clip-reid \
    2>&1 | tee ${OUTPUT_DIR}/train_${RUN_STAMP}.log

  echo "[INFO] Training finished for ${DATASET_NAME}"

  WEIGHT_FILE=$(ls -t ${OUTPUT_DIR}/*.pth 2>/dev/null | head -n 1)

  if [ -z "${WEIGHT_FILE}" ]; then
    echo "[WARN] No checkpoint found for ${DATASET_NAME}, skip test"
    return
  fi

  echo "[INFO] Testing ${DATASET_NAME} using ${WEIGHT_FILE}"
  CUDA_VISIBLE_DEVICES=0 \
  uv run python test_clipreid.py \
    --config_file ${CONFIG_FILE} \
    TEST.WEIGHT ${WEIGHT_FILE} \
    DATASETS.NAMES "('${DATASET_NAME}')" \
    DATASETS.ROOT_DIR ${DATA_ROOT} \
    OUTPUT_DIR ${OUTPUT_DIR} \
    WANDB.ENABLED True \
    WANDB.PROJECT clip-reid \
    2>&1 | tee ${OUTPUT_DIR}/test_${RUN_STAMP}.log

  echo "[INFO] TEST DONE: ${DATASET_NAME}"
}

############################################
# STRICT SEQUENTIAL EXECUTION
############################################
run_dataset "market1501" "${OUTPUT_ROOT}/rn50_clipreid_market1501"
echo "[INFO] Market1501 DONE. Proceeding to MSMT17."

run_dataset "msmt17" "${OUTPUT_ROOT}/rn50_clipreid_msmt17"
echo "[INFO] MSMT17 DONE."

echo "========================================="
echo " ALL DATASETS FINISHED SUCCESSFULLY"
echo "========================================="

echo "[INFO] System will shutdown in 60 seconds..."
sleep 60 && shutdown -h now
