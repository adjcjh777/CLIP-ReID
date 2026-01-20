#!/usr/bin/env bash

echo "========================================="
echo " CLIP-ReID TEST ONLY"
echo " Market1501  ->  MSMT17 (SEQUENTIAL)"
echo " ViT-B-16 + SIE + OLP"
echo "========================================="

DATA_ROOT="/root/autodl-tmp/CLIP_REID/DATASETS"
OUTPUT_ROOT="/root/autodl-tmp/CLIP_REID/OUTPUT"
CONFIG_FILE="configs/person/vit_clipreid.yml"
RUN_STAMP=$(date +"%Y%m%d_%H%M%S")

TRAIN_SIZE="[384, 128]"
TEST_SIZE="[384, 128]"
TEST_BATCH=256

echo "[INFO] Workdir: $(pwd)"
uv run python -V
nvidia-smi || true

############################################
# Function: test only
############################################
run_test () {
  DATASET_NAME=$1
  OUTPUT_DIR=$2

  echo "-----------------------------------------"
  echo "[INFO] TEST DATASET: ${DATASET_NAME}"
  echo "-----------------------------------------"

  if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "[WARN] Output dir not found: ${OUTPUT_DIR}"
    return
  fi

  WEIGHT_FILE=$(ls -t ${OUTPUT_DIR}/*.pth 2>/dev/null | grep -v stage1 | head -n 1)
  if [ -z "${WEIGHT_FILE}" ]; then
    WEIGHT_FILE=$(ls -t ${OUTPUT_DIR}/*.pth 2>/dev/null | head -n 1)
  fi

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
    INPUT.SIZE_TRAIN "${TRAIN_SIZE}" \
    INPUT.SIZE_TEST "${TEST_SIZE}" \
    TEST.IMS_PER_BATCH ${TEST_BATCH} \
    MODEL.SIE_CAMERA True \
    MODEL.SIE_COE 1.0 \
    MODEL.STRIDE_SIZE "[12, 12]" \
    WANDB.ENABLED True \
    WANDB.PROJECT clip-reid \
    2>&1 | tee ${OUTPUT_DIR}/test_${RUN_STAMP}.log
}

############################################
# SEQUENTIAL EXECUTION
############################################
run_test "market1501" "${OUTPUT_ROOT}/vit_clipreid_sie_olp_market1501"
run_test "msmt17" "${OUTPUT_ROOT}/vit_clipreid_sie_olp_msmt17"

echo "========================================="
echo " ALL TESTS FINISHED"
echo "========================================="
