#!/usr/bin/env bash

echo "========================================="
echo " CLIP-ReID GRID RUNNER (BASELINE)"
echo " Market1501  ->  MSMT17"
echo " Seeds + Train-PCT"
echo "========================================="

DATA_ROOT="/root/autodl-tmp/CLIP_REID/DATASETS"
OUTPUT_ROOT="/root/autodl-tmp/CLIP_REID/OUTPUT"
CONFIG_FILE="configs/person/vit_clipreid.yml"

STAGE1_BATCH=256
STAGE2_BATCH=256
TRAIN_SIZE="[384, 128]"
TEST_SIZE="[384, 128]"
TEST_BATCH=256

SEEDS_STR="${SEEDS_STR:-"777 2345 3456"}"
TRAIN_PCTS_STR="${TRAIN_PCTS_STR:-"1.0"}"
TRAIN_PCT_MODE="${TRAIN_PCT_MODE:-ids}"
TRAIN_PCT_SEED="${TRAIN_PCT_SEED:-}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"

SCRIPT_NAME="$(basename "$0" .sh)"
RUN_LOG="${OUTPUT_ROOT}/${SCRIPT_NAME}_run.log"

log() {
  echo "[$(date +\"%Y-%m-%d %H:%M:%S\")] $*" | tee -a "${RUN_LOG}"
}

trap 'log \"[INTERRUPTED] signal received\"; exit 130' INT TERM

read -r -a SEEDS <<< "${SEEDS_STR}"
read -r -a TRAIN_PCTS <<< "${TRAIN_PCTS_STR}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log "Workdir: $(pwd)"
uv run python -V
nvidia-smi || true

run_one () {
  local dataset_name=$1
  local seed=$2
  local train_pct=$3
  local train_pct_seed=$4

  local pct_tag=${train_pct//./p}
  local output_dir="${OUTPUT_ROOT}/vit_clipreid_${dataset_name}_seed${seed}_pct${pct_tag}"
  local run_stamp=$(date +"%Y%m%d_%H%M%S")
  local status_file="${output_dir}/status_seed${seed}_pct${pct_tag}.txt"
  local train_done_flag="${output_dir}/.train_done_seed${seed}_pct${pct_tag}"
  local done_flag="${output_dir}/.done_seed${seed}_pct${pct_tag}"

  log "-----------------------------------------"
  log "DATASET=${dataset_name} SEED=${seed} TRAIN_PCT=${train_pct}"
  log "OUTPUT_DIR=${output_dir}"
  log "-----------------------------------------"

  mkdir -p "${output_dir}"

  if [ -f "${done_flag}" ]; then
    log "[SKIP] already done: ${dataset_name} seed=${seed} pct=${train_pct}"
    return
  fi

  if [ ! -f "${train_done_flag}" ]; then
    log "[TRAIN] start"
    CUDA_VISIBLE_DEVICES=0 \
    uv run python train_clipreid.py \
      --config_file ${CONFIG_FILE} \
      DATASETS.NAMES "('${dataset_name}')" \
      DATASETS.ROOT_DIR ${DATA_ROOT} \
      DATASETS.TRAIN_PCT ${train_pct} \
      DATASETS.TRAIN_PCT_MODE ${TRAIN_PCT_MODE} \
      DATASETS.TRAIN_PCT_SEED ${train_pct_seed} \
      SOLVER.SEED ${seed} \
      OUTPUT_DIR ${output_dir} \
      SOLVER.STAGE1.IMS_PER_BATCH ${STAGE1_BATCH} \
      SOLVER.STAGE2.IMS_PER_BATCH ${STAGE2_BATCH} \
      INPUT.SIZE_TRAIN "${TRAIN_SIZE}" \
      INPUT.SIZE_TEST "${TEST_SIZE}" \
      TEST.IMS_PER_BATCH ${TEST_BATCH} \
      WANDB.ENABLED True \
      WANDB.PROJECT clip-reid \
      2>&1 | tee ${output_dir}/train_${run_stamp}_seed${seed}_pct${pct_tag}.log
    train_rc=${PIPESTATUS[0]}
    echo "train_rc=${train_rc} time=$(date +\"%Y-%m-%d %H:%M:%S\")" >> "${status_file}"
    if [ "${train_rc}" -ne 0 ]; then
      log "[TRAIN] failed rc=${train_rc}"
      return
    fi
    touch "${train_done_flag}"
    log "[TRAIN] done"
  else
    log "[TRAIN] already done, skip training"
  fi

  WEIGHT_FILE=$(ls -t ${output_dir}/*.pth 2>/dev/null | head -n 1)

  if [ -z "${WEIGHT_FILE}" ]; then
    log "[WARN] No checkpoint found for ${dataset_name}, skip test"
    return
  fi

  log "[TEST] start"
  CUDA_VISIBLE_DEVICES=0 \
  uv run python test_clipreid.py \
    --config_file ${CONFIG_FILE} \
    TEST.WEIGHT ${WEIGHT_FILE} \
    DATASETS.NAMES "('${dataset_name}')" \
    DATASETS.ROOT_DIR ${DATA_ROOT} \
    DATASETS.TRAIN_PCT ${train_pct} \
    DATASETS.TRAIN_PCT_MODE ${TRAIN_PCT_MODE} \
    DATASETS.TRAIN_PCT_SEED ${train_pct_seed} \
    OUTPUT_DIR ${output_dir} \
    INPUT.SIZE_TRAIN "${TRAIN_SIZE}" \
    INPUT.SIZE_TEST "${TEST_SIZE}" \
    TEST.IMS_PER_BATCH ${TEST_BATCH} \
    WANDB.ENABLED True \
    WANDB.PROJECT clip-reid \
    2>&1 | tee ${output_dir}/test_${run_stamp}_seed${seed}_pct${pct_tag}.log
  test_rc=${PIPESTATUS[0]}
  echo "test_rc=${test_rc} time=$(date +\"%Y-%m-%d %H:%M:%S\")" >> "${status_file}"
  if [ "${test_rc}" -ne 0 ]; then
    log "[TEST] failed rc=${test_rc}"
    return
  fi
  touch "${done_flag}"
  log "[TEST] done"
}

for seed in "${SEEDS[@]}"; do
  for train_pct in "${TRAIN_PCTS[@]}"; do
    if [ -n "${TRAIN_PCT_SEED}" ]; then
      pct_seed=${TRAIN_PCT_SEED}
    else
      pct_seed=${seed}
    fi

    run_one "market1501" "${seed}" "${train_pct}" "${pct_seed}"
    log "Market1501 DONE. Proceeding to MSMT17."

    run_one "msmt17" "${seed}" "${train_pct}" "${pct_seed}"
    log "MSMT17 DONE."
  done
done

log "========================================="
log " ALL DATASETS FINISHED SUCCESSFULLY"
log "========================================="

if [ "${AUTO_SHUTDOWN}" = "1" ]; then
  log "System will shutdown in 60 seconds..."
  sleep 60 && shutdown -h now
fi
