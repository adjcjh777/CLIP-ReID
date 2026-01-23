#!/usr/bin/env bash
set -euo pipefail

echo "========================================="
echo " CLIP-ReID GRID RUNNER (BOTH)"
echo " Seeds + Train-PCT (ids)"
echo "========================================="

PROJECT_DIR="/root/CLIP-ReID"
DATA_ROOT="/root/autodl-tmp/CLIP_REID/DATASETS"
OUTPUT_ROOT="/root/autodl-tmp/CLIP_REID/OUTPUT"

SEEDS_STR="1234 1111 7 77 4321"
TRAIN_PCTS_STR="1.0 0.75 0.5"
TRAIN_PCT_MODE="ids"
AUTO_SHUTDOWN=1

export SEEDS_STR TRAIN_PCTS_STR TRAIN_PCT_MODE AUTO_SHUTDOWN

echo "[CHECK] Project dir: ${PROJECT_DIR}"
if [ ! -d "${PROJECT_DIR}" ]; then
  echo "[ERROR] Project dir not found: ${PROJECT_DIR}"
  exit 1
fi

echo "[CHECK] Dataset root: ${DATA_ROOT}"
if [ ! -d "${DATA_ROOT}" ]; then
  echo "[ERROR] Dataset root not found: ${DATA_ROOT}"
  exit 1
fi

echo "[CHECK] Output root: ${OUTPUT_ROOT}"
if [ ! -d "${OUTPUT_ROOT}" ]; then
  mkdir -p "${OUTPUT_ROOT}"
fi
touch "${OUTPUT_ROOT}/.write_test" 2>/dev/null || { echo "[ERROR] Output root not writable: ${OUTPUT_ROOT}"; exit 1; }
rm -f "${OUTPUT_ROOT}/.write_test"

if [ ! -x "${PROJECT_DIR}/run_clipreid_market_then_msmt_uv_grid.sh" ]; then
  echo "[ERROR] Missing or not executable: ${PROJECT_DIR}/run_clipreid_market_then_msmt_uv_grid.sh"
  exit 1
fi
if [ ! -x "${PROJECT_DIR}/run_clipreid_market_then_msmt_sie_olp_uv_grid.sh" ]; then
  echo "[ERROR] Missing or not executable: ${PROJECT_DIR}/run_clipreid_market_then_msmt_sie_olp_uv_grid.sh"
  exit 1
fi

cd "${PROJECT_DIR}"

./run_clipreid_market_then_msmt_uv_grid.sh
./run_clipreid_market_then_msmt_sie_olp_uv_grid.sh
