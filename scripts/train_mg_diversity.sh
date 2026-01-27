#!/bin/bash
# 训练多粒度融合模型 - 带有 Part 多样性损失
# 用法: bash scripts/train_mg_diversity.sh [DATASET] [GPU_ID]

set -e

DATASET=${1:-"market1501"}
GPU_ID=${2:-0}

# 输出目录
OUTPUT_DIR="/root/autodl-tmp/CLIP_REID/OUTPUT/multi_granularity_v2/${DATASET}"
TB_LOG_DIR="/root/autodl-tmp/CLIP_REID/OUTPUT/multi_granularity_v2/tensorboard/${DATASET}"

# 数据集路径
DATA_ROOT="/root/autodl-tmp/CLIP_REID/DATASETS"

echo "=============================================="
echo "Multi-Granularity Training with Diversity Loss"
echo "=============================================="
echo "Dataset: ${DATASET}"
echo "GPU: ${GPU_ID}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# 创建输出目录
mkdir -p ${OUTPUT_DIR}
mkdir -p ${TB_LOG_DIR}

# 训练命令
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_clipreid.py \
    --config_file configs/person/vit_clipreid.yml \
    DATASETS.NAMES "('${DATASET}')" \
    DATASETS.ROOT_DIR "${DATA_ROOT}" \
    OUTPUT_DIR "${OUTPUT_DIR}" \
    SOLVER.STAGE1.IMS_PER_BATCH 256 \
    SOLVER.STAGE2.IMS_PER_BATCH 256 \
    INPUT.SIZE_TRAIN "[384, 128]" \
    INPUT.SIZE_TEST "[384, 128]" \
    TEST.IMS_PER_BATCH 256 \
    MODEL.MULTI_GRANULARITY.ENABLED True \
    MODEL.MULTI_GRANULARITY.NUM_PARTS 4 \
    MODEL.MULTI_GRANULARITY.PART_DIM 512 \
    MODEL.PART_ID_LOSS_WEIGHT 0.25 \
    MODEL.PART_TRIPLET_LOSS_WEIGHT 0.5 \
    MODEL.PART_DIVERSITY_LOSS_WEIGHT 0.5 \
    MODEL.DIVERSITY_ORTHO_WEIGHT 1.0 \
    MODEL.DIVERSITY_CONTRAST_WEIGHT 0.5 \
    MODEL.DIVERSITY_CONTRAST_MARGIN 0.3 \
    TENSORBOARD.ENABLED True \
    TENSORBOARD.LOG_DIR "${TB_LOG_DIR}" \
    2>&1 | tee ${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Training completed!"
echo "Output saved to: ${OUTPUT_DIR}"
