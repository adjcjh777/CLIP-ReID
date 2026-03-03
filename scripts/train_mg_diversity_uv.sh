#!/bin/bash
# ============================================================
# Multi-Granularity Training with Part Diversity Loss
# 使用 uv 环境配置
# ============================================================
# 用法: bash scripts/train_mg_diversity_uv.sh [DATASET] [GPU_ID]
# 
# 训练完成后自动关机
# ============================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 参数设置
DATASET=${1:-"market1501"}
GPU_ID=${2:-0}

# 路径配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -d "${PROJECT_ROOT}/.venv-5090" ]; then
    VENV_PATH="${PROJECT_ROOT}/.venv-5090"
else
    VENV_PATH="${PROJECT_ROOT}/.venv"
fi
DATA_ROOT="${CLIPREID_DATA_ROOT:-${PROJECT_ROOT}/DATASETS}"
OUTPUT_ROOT="${CLIPREID_OUTPUT_ROOT:-${PROJECT_ROOT}/OUTPUT}"
OUTPUT_BASE="${OUTPUT_ROOT}/multi_granularity_v2"
OUTPUT_DIR="${OUTPUT_BASE}/${DATASET}"
TB_LOG_DIR="${OUTPUT_BASE}/tensorboard/${DATASET}"

echo -e "${GREEN}=============================================="
echo "Multi-Granularity Training with Diversity Loss"
echo -e "==============================================${NC}"
echo ""

# ============================================================
# 1. 环境检查
# ============================================================
echo -e "${YELLOW}[Step 1/5] Checking environment...${NC}"

# 检查 uv 虚拟环境
if [ ! -d "${VENV_PATH}" ]; then
    echo -e "${RED}Error: Virtual environment not found at ${VENV_PATH}${NC}"
    exit 1
fi
echo "  ✓ Virtual environment found: ${VENV_PATH}"

# 激活虚拟环境
source "${VENV_PATH}/bin/activate"
echo "  ✓ Virtual environment activated"

# 检查 Python
PYTHON_PATH=$(which python)
echo "  ✓ Python: ${PYTHON_PATH}"
echo "  ✓ Python version: $(python --version 2>&1)"

# ============================================================
# 2. 路径验证
# ============================================================
echo ""
echo -e "${YELLOW}[Step 2/5] Validating paths...${NC}"

# 检查项目目录
if [ ! -f "${PROJECT_ROOT}/train_clipreid.py" ]; then
    echo -e "${RED}Error: train_clipreid.py not found in ${PROJECT_ROOT}${NC}"
    exit 1
fi
echo "  ✓ Project root: ${PROJECT_ROOT}"

# 检查数据集
if [ "${DATASET}" == "market1501" ]; then
    DATASET_PATH="${DATA_ROOT}/Market-1501-v15.09.15"
elif [ "${DATASET}" == "msmt17" ]; then
    DATASET_PATH="${DATA_ROOT}/MSMT17"
else
    echo -e "${RED}Error: Unknown dataset ${DATASET}${NC}"
    exit 1
fi

if [ ! -d "${DATASET_PATH}" ]; then
    echo -e "${RED}Error: Dataset not found at ${DATASET_PATH}${NC}"
    exit 1
fi
echo "  ✓ Dataset path: ${DATASET_PATH}"

# 检查 GPU
if ! nvidia-smi -i ${GPU_ID} &> /dev/null; then
    echo -e "${RED}Error: GPU ${GPU_ID} not available${NC}"
    exit 1
fi
echo "  ✓ GPU ${GPU_ID} is available"

# ============================================================
# 3. 创建输出目录
# ============================================================
echo ""
echo -e "${YELLOW}[Step 3/5] Creating output directories...${NC}"

mkdir -p ${OUTPUT_DIR}
mkdir -p ${TB_LOG_DIR}
echo "  ✓ Output directory: ${OUTPUT_DIR}"
echo "  ✓ TensorBoard log: ${TB_LOG_DIR}"

# ============================================================
# 4. 显示训练配置
# ============================================================
echo ""
echo -e "${YELLOW}[Step 4/5] Training configuration:${NC}"
echo "  Dataset: ${DATASET}"
echo "  GPU: ${GPU_ID}"
echo "  Input size: [384, 128]"
echo "  Multi-granularity: Enabled (4 parts)"
echo "  Part ID Loss Weight: 0.25"
echo "  Part Triplet Loss Weight: 0.5"
echo "  Part Diversity Loss Weight: 0.5"
echo ""

# 记录开始时间
START_TIME=$(date +%s)
LOG_FILE="${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

echo -e "${GREEN}Starting training at $(date)${NC}"
echo "Log file: ${LOG_FILE}"
echo ""

# ============================================================
# 5. 开始训练
# ============================================================
echo -e "${YELLOW}[Step 5/5] Training...${NC}"

cd ${PROJECT_ROOT}

CUDA_VISIBLE_DEVICES=${GPU_ID} python train_clipreid.py \
    --config_file configs/person/vit_clipreid.yml \
    DATASETS.NAMES "('${DATASET}')" \
    DATASETS.ROOT_DIR "${DATA_ROOT}" \
    OUTPUT_DIR "${OUTPUT_DIR}" \
    SOLVER.STAGE1.IMS_PER_BATCH 256 \
    SOLVER.STAGE2.IMS_PER_BATCH 256 \
    SOLVER.STAGE1.MAX_EPOCHS 120 \
    SOLVER.STAGE2.MAX_EPOCHS 60 \
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
    2>&1 | tee ${LOG_FILE}

# 记录结束时间
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo -e "${GREEN}=============================================="
echo "Training completed!"
echo "==============================================${NC}"
echo "Total time: ${HOURS}h ${MINUTES}m"
echo "Output saved to: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo ""

# ============================================================
# 6. 训练完成后关机
# ============================================================
echo -e "${YELLOW}System will shutdown in 60 seconds...${NC}"
echo "Press Ctrl+C to cancel shutdown"

sleep 60

echo -e "${RED}Shutting down...${NC}"
sudo shutdown -h now
