#!/usr/bin/env bash
set -e

echo "=============================="
echo " CLIP-ReID (uv) One-Click Run "
echo "=============================="

############################
# 路径配置（严格对齐你目录）
############################
PROJECT_ROOT="/root/CLIP_REID/CLIP-ReID"
DATA_ROOT="/root/CLIP_REID/DATASETS"
OUTPUT_ROOT="/root/CLIP_REID/OUTPUT"

DATASET_NAME="market1501"
CONFIG_FILE="configs/person/vit_clipreid.yml"
OUTPUT_DIR="${OUTPUT_ROOT}/vit_clipreid_market1501"

############################
# 进入项目目录
############################
cd ${PROJECT_ROOT}

############################
# 基本信息
############################
echo "[INFO] Using uv python:"
uv run python -V

echo "[INFO] CUDA devices:" ${CUDA_VISIBLE_DEVICES}
nvidia-smi || true

############################
# 创建输出目录
############################
mkdir -p ${OUTPUT_DIR}

############################
# 训练
############################
echo "[INFO] Start Training..."

CUDA_VISIBLE_DEVICES=0 \
uv run python train_clipreid.py \
  --config_file ${CONFIG_FILE} \
  DATASETS.NAMES "('${DATASET_NAME}')" \
  DATASETS.ROOT_DIR ${DATA_ROOT} \
  OUTPUT_DIR ${OUTPUT_DIR} \
  2>&1 | tee ${OUTPUT_DIR}/train.log

############################
# 检查模型
############################
if [ ! -f "${OUTPUT_DIR}/model_best.pth" ]; then
  echo "[ERROR] model_best.pth not found!"
  exit 1
fi

############################
# 测试
############################
echo "[INFO] Start Testing..."

CUDA_VISIBLE_DEVICES=0 \
uv run python test_clipreid.py \
  --config_file ${CONFIG_FILE} \
  TEST.WEIGHT ${OUTPUT_DIR}/model_best.pth \
  DATASETS.NAMES "('${DATASET_NAME}')" \
  DATASETS.ROOT_DIR ${DATA_ROOT} \
  2>&1 | tee ${OUTPUT_DIR}/test.log

############################
# 结束
############################
echo "=============================="
echo " CLIP-ReID Finished (uv) "
echo " Results in ${OUTPUT_DIR}"
echo "=============================="
## 系统关机##
echo "[INFO] Training finished, shutting down in 1 minute..."
sleep 60
sudo shutdown -h now
echo "[INFO] Shutdown command issued."