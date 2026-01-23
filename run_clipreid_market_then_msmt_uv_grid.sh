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

read -r -a SEEDS <<< "${SEEDS_STR}"
read -r -a TRAIN_PCTS <<< "${TRAIN_PCTS_STR}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[INFO] Workdir: $(pwd)"
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

  echo "-----------------------------------------"
  echo "[INFO] DATASET=${dataset_name} SEED=${seed} TRAIN_PCT=${train_pct}"
  echo "[INFO] OUTPUT_DIR=${output_dir}"
  echo "-----------------------------------------"

  mkdir -p "${output_dir}"

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

  WEIGHT_FILE=$(ls -t ${output_dir}/*.pth 2>/dev/null | head -n 1)

  if [ -z "${WEIGHT_FILE}" ]; then
    echo "[WARN] No checkpoint found for ${dataset_name}, skip test"
    return
  fi

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
}

for seed in "${SEEDS[@]}"; do
  for train_pct in "${TRAIN_PCTS[@]}"; do
    if [ -n "${TRAIN_PCT_SEED}" ]; then
      pct_seed=${TRAIN_PCT_SEED}
    else
      pct_seed=${seed}
    fi

    run_one "market1501" "${seed}" "${train_pct}" "${pct_seed}"
    echo "[INFO] Market1501 DONE. Proceeding to MSMT17."

    run_one "msmt17" "${seed}" "${train_pct}" "${pct_seed}"
    echo "[INFO] MSMT17 DONE."
  done
done

echo "========================================="
echo " ALL DATASETS FINISHED SUCCESSFULLY"
echo "========================================="

if [ "${AUTO_SHUTDOWN}" = "1" ]; then
  echo "[INFO] System will shutdown in 60 seconds..."
  sleep 60 && shutdown -h now
fi
