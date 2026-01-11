#!/usr/bin/env bash
set -e

echo "=============================="
echo " CLIP-ReID One-Click Launcher "
echo "=============================="

############################
# 1. 基本路径配置（按你目录）
############################
PROJECT_ROOT="/root/CLIP_REID/CLIP-ReID"
DATA_ROOT="/root/CLIP_REID/DATASETS"
OUTPUT_ROOT="/root/CLIP_REID/OUTPUT"

DATASET_NAME="market1501"
CONFIG_FILE="configs/person/vit_clipreid.yml"
OUTPUT_DIR="${OUTPUT_ROOT}/vit_clipreid_market1501"

############################
# 2. 进入项目目录
############################
cd ${PROJECT_ROOT}

############################
# 3. Python & GPU 检查
############################
echo "[INFO] Python:" $(which python)
echo "[INFO] CUDA devices:" ${CUDA_VISIBLE_DEVICES}

nvidia-smi || echo "[WARN] nvidia-smi not available"

############################
# 4. 创建输出目录
############################
mkdir -p ${OUTPUT_DIR}

############################
# 5. 开始训练
############################
echo "[INFO] Start Training..."

CUDA_VISIBLE_DEVICES=0 \
python train_clipreid.py \
  --config_file ${CONFIG_FILE} \
  DATASETS.NAMES "('${DATASET_NAME}')" \
  DATASETS.ROOT_DIR ${DATA_ROOT} \
  OUTPUT_DIR ${OUTPUT_DIR} \
  2>&1 | tee ${OUTPUT_DIR}/train.log

############################
# 6. 查找最优模型
############################
BEST_MODEL=$(ls ${OUTPUT_DIR}/model_best.pth 2>/dev/null || true)

if [ -z "${BEST_MODEL}" ]; then
  echo "[ERROR] model_best.pth not found!"
  exit 1
fi

############################
# 7. 开始测试
############################
echo "[INFO] Start Testing..."

CUDA_VISIBLE_DEVICES=0 \
python test_clipreid.py \
  --config_file ${CONFIG_FILE} \
  TEST.WEIGHT ${BEST_MODEL} \
  DATASETS.NAMES "('${DATASET_NAME}')" \
  DATASETS.ROOT_DIR ${DATA_ROOT} \
  2>&1 | tee ${OUTPUT_DIR}/test.log

############################
# 8. 完成提示
############################
echo "=============================="
echo " CLIP-ReID Finished Successfully "
echo " Results saved to:"
echo " ${OUTPUT_DIR}"
echo "=============================="
