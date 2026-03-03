#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PROJECT_DIR}/.venv-5090/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python"
fi

GPU_ID="${1:-0}"
DATA_ROOT="${CLIPREID_DATA_ROOT:-${PROJECT_DIR}/DATASETS}"
OUTPUT_ROOT="${CLIPREID_OUTPUT_ROOT:-${PROJECT_DIR}/OUTPUT}"
RAW_MACHINE_TAG="${CLIPREID_MACHINE_TAG:-$(hostname -s 2>/dev/null || echo local)}"
MACHINE_TAG="$(echo "${RAW_MACHINE_TAG}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')"
if [[ -z "${MACHINE_TAG}" ]]; then
  MACHINE_TAG="local"
fi
EXPERIMENT_NAME="B5_i2t_w0.2_sie_olp_${MACHINE_TAG}"
RUN_DIR="${OUTPUT_ROOT}/phase2/${EXPERIMENT_NAME}"
LEGACY_ALIAS="${OUTPUT_ROOT}/today_plan_20260302_b5"
CONFIG_FILE="${PROJECT_DIR}/configs/person/vit_clipreid.yml"
ANNOTATION_FILE="${PROJECT_DIR}/annotations/market1501_train.json"

mkdir -p "${RUN_DIR}"
if [[ -L "${LEGACY_ALIAS}" || ! -e "${LEGACY_ALIAS}" ]]; then
  ln -sfn "${RUN_DIR}" "${LEGACY_ALIAS}"
fi
cd "${PROJECT_DIR}"

echo "[PLAN] Start B5 local rebuild at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[PLAN] GPU=${GPU_ID}"
echo "[PLAN] DATA_ROOT=${DATA_ROOT}"
echo "[PLAN] MACHINE_TAG=${MACHINE_TAG}"
echo "[PLAN] EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "[PLAN] RUN_DIR=${RUN_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" train_text_reid.py \
  --config_file "${CONFIG_FILE}" \
  --annotation_file "${ANNOTATION_FILE}" \
  DATASETS.NAMES market1501 \
  DATASETS.ROOT_DIR "${DATA_ROOT}" \
  OUTPUT_DIR "${RUN_DIR}" \
  SOLVER.SEED 1234 \
  SOLVER.STAGE1.IMS_PER_BATCH 32 \
  SOLVER.STAGE1.MAX_EPOCHS 60 \
  SOLVER.STAGE1.CHECKPOINT_PERIOD 60 \
  SOLVER.STAGE2.IMS_PER_BATCH 32 \
  SOLVER.STAGE2.MAX_EPOCHS 60 \
  MODEL.TEXT_ENCODER_TYPE clip_native \
  MODEL.TEXT_LOSS_TYPE none \
  MODEL.I2T_LOSS_WEIGHT 0.2 \
  MODEL.SIE_CAMERA True \
  MODEL.SIE_COE 3.0 \
  MODEL.STRIDE_SIZE "[12, 12]" \
  INPUT.SIZE_TRAIN "[256, 128]" \
  INPUT.SIZE_TEST "[256, 128]" \
  TEST.EVAL True 2>&1 | tee "${RUN_DIR}/train.log"

CKPT="${RUN_DIR}/ViT-B-16_stage2_60.pth"
if [[ ! -f "${CKPT}" ]]; then
  echo "[PLAN] Stage2 checkpoint missing: ${CKPT}"
  exit 1
fi

echo "[PLAN] Start baseline eval at $(date '+%Y-%m-%d %H:%M:%S')"
"${PYTHON}" scripts/eval_checkpoint.py \
  --config_file "${CONFIG_FILE}" \
  --checkpoint "${CKPT}" \
  DATASETS.NAMES market1501 \
  DATASETS.ROOT_DIR "${DATA_ROOT}" \
  TEST.FEAT_NORM yes \
  TEST.RE_RANKING False \
  MODEL.PRETRAIN_PATH "" \
  MODEL.TEXT_ENCODER_TYPE clip_native \
  MODEL.STRIDE_SIZE "[12, 12]" \
  MODEL.SIE_CAMERA True \
  MODEL.SIE_COE 3.0 \
  > "${RUN_DIR}/eval_baseline.log" 2>&1

echo "[PLAN] Start reranking eval at $(date '+%Y-%m-%d %H:%M:%S')"
"${PYTHON}" scripts/eval_checkpoint.py \
  --config_file "${CONFIG_FILE}" \
  --checkpoint "${CKPT}" \
  --reranking \
  DATASETS.NAMES market1501 \
  DATASETS.ROOT_DIR "${DATA_ROOT}" \
  TEST.FEAT_NORM yes \
  MODEL.PRETRAIN_PATH "" \
  MODEL.TEXT_ENCODER_TYPE clip_native \
  MODEL.STRIDE_SIZE "[12, 12]" \
  MODEL.SIE_CAMERA True \
  MODEL.SIE_COE 3.0 \
  > "${RUN_DIR}/eval_rerank.log" 2>&1

echo "[PLAN] Finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[PLAN] Output directory: ${RUN_DIR}"
