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

EXPERIMENT_NAME="D1c_combined_local_s2e30_${MACHINE_TAG}"
RUN_DIR="${OUTPUT_ROOT}/phase4/${EXPERIMENT_NAME}"
CONFIG_FILE="${PROJECT_DIR}/configs/person/vit_clipreid.yml"
ANNOTATION_FILE="${PROJECT_DIR}/annotations/market1501_train.json"

S1_CKPT="${2:-${CLIPREID_STAGE1_CKPT:-}}"
if [[ -z "${S1_CKPT}" ]]; then
  S1_CKPT="$(ls -1dt "${OUTPUT_ROOT}"/phase2/B5_i2t_w0.2_sie_olp_*/ViT-B-16_stage1_60.pth 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${S1_CKPT}" || ! -f "${S1_CKPT}" ]]; then
  echo "[PLAN] Missing stage1 checkpoint. Provide via arg2 or CLIPREID_STAGE1_CKPT."
  exit 1
fi

mkdir -p "${RUN_DIR}"
cd "${PROJECT_DIR}"

echo "[PLAN] Start D1c combined screen at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[PLAN] GPU=${GPU_ID}"
echo "[PLAN] DATA_ROOT=${DATA_ROOT}"
echo "[PLAN] MACHINE_TAG=${MACHINE_TAG}"
echo "[PLAN] S1_CKPT=${S1_CKPT}"
echo "[PLAN] RUN_DIR=${RUN_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" train_text_reid.py \
  --config_file "${CONFIG_FILE}" \
  --annotation_file "${ANNOTATION_FILE}" \
  --skip_stage1 \
  --stage1_checkpoint "${S1_CKPT}" \
  DATASETS.NAMES market1501 \
  DATASETS.ROOT_DIR "${DATA_ROOT}" \
  OUTPUT_DIR "${RUN_DIR}" \
  SOLVER.SEED 1234 \
  SOLVER.STAGE2.IMS_PER_BATCH 32 \
  SOLVER.STAGE2.MAX_EPOCHS 30 \
  SOLVER.STAGE2.CHECKPOINT_PERIOD 30 \
  SOLVER.STAGE2.EVAL_PERIOD 30 \
  MODEL.TEXT_ENCODER_TYPE clip_native \
  MODEL.TEXT_LOSS_TYPE combined \
  MODEL.TEXT_LOSS_WEIGHT 0.5 \
  MODEL.TEXT_TRIPLET_WEIGHT 0.5 \
  MODEL.TEXT_CONTRASTIVE_WEIGHT 1.0 \
  MODEL.I2T_LOSS_WEIGHT 0.2 \
  MODEL.SIE_CAMERA True \
  MODEL.SIE_COE 3.0 \
  MODEL.STRIDE_SIZE "[12, 12]" \
  INPUT.SIZE_TRAIN "[256, 128]" \
  INPUT.SIZE_TEST "[256, 128]" \
  TEST.EVAL True 2>&1 | tee "${RUN_DIR}/train.log"

CKPT="${RUN_DIR}/ViT-B-16_stage2_30.pth"
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

echo "[PLAN] Baseline metrics:"
grep -E "mAP|Rank-1|Rank-5|Rank-10" "${RUN_DIR}/eval_baseline.log" | tail -n 4 || true
echo "[PLAN] Reranking metrics:"
grep -E "mAP|Rank-1|Rank-5|Rank-10" "${RUN_DIR}/eval_rerank.log" | tail -n 4 || true
