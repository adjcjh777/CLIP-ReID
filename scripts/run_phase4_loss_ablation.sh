#!/usr/bin/env bash
# =======================================================================
# Phase 4 跨模态损失消融（优先执行版）
#
# 目标：在 B4 最优配置基础上，优先验证 triplet/cmpm 两种损失
#
# 用法：
#   bash scripts/run_phase4_loss_ablation.sh D1_triplet
#   bash scripts/run_phase4_loss_ablation.sh D1_cmpm
#   bash scripts/run_phase4_loss_ablation.sh all
# =======================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PROJECT_DIR}/.venv-5090/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python"
fi
CONFIG="configs/person/vit_clipreid.yml"
ANNOTATION="annotations/market1501_train.json"
DATASETS_ROOT="${CLIPREID_DATA_ROOT:-${PROJECT_DIR}/DATASETS}"
OUTPUT_ROOT="${CLIPREID_OUTPUT_ROOT:-${PROJECT_DIR}/OUTPUT}"
OUTPUT_BASE="${OUTPUT_ROOT}/phase4"

# B1 的 Stage1 checkpoint 对 query_encoder 可直接复用
S1_CKPT="${OUTPUT_ROOT}/phase2/B1_query_encoder/ViT-B-16_stage1_60.pth"

COMMON_OPTS="DATASETS.NAMES market1501 DATASETS.ROOT_DIR ${DATASETS_ROOT} SOLVER.SEED 1234 SOLVER.STAGE1.IMS_PER_BATCH 32 SOLVER.STAGE1.MAX_EPOCHS 60 SOLVER.STAGE2.IMS_PER_BATCH 32 SOLVER.STAGE2.MAX_EPOCHS 60 INPUT.SIZE_TRAIN [256,128] INPUT.SIZE_TEST [256,128]"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_BASE"

log_msg() {
  echo ""
  echo "========================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
  echo "========================================"
  echo ""
}

run_d1_triplet() {
  local OUT="${OUTPUT_BASE}/D1_triplet"
  if [ ! -f "$S1_CKPT" ]; then
    echo "ERROR: missing stage1 checkpoint: $S1_CKPT"
    return 1
  fi
  log_msg "Starting D1_triplet: query_encoder + I2T=0.2 + TEXT_LOSS_TYPE=triplet"
  $PYTHON train_text_reid.py \
    --config_file $CONFIG \
    --annotation_file $ANNOTATION \
    --skip_stage1 \
    --stage1_checkpoint "$S1_CKPT" \
    $COMMON_OPTS \
    OUTPUT_DIR "$OUT" \
    MODEL.TEXT_ENCODER_TYPE query_encoder \
    MODEL.I2T_LOSS_WEIGHT 0.2 \
    MODEL.TEXT_LOSS_TYPE triplet \
    MODEL.TEXT_LOSS_WEIGHT 0.5
  log_msg "D1_triplet completed"
}

run_d1_cmpm() {
  local OUT="${OUTPUT_BASE}/D1_cmpm"
  if [ ! -f "$S1_CKPT" ]; then
    echo "ERROR: missing stage1 checkpoint: $S1_CKPT"
    return 1
  fi
  log_msg "Starting D1_cmpm: query_encoder + I2T=0.2 + TEXT_LOSS_TYPE=cmpm"
  $PYTHON train_text_reid.py \
    --config_file $CONFIG \
    --annotation_file $ANNOTATION \
    --skip_stage1 \
    --stage1_checkpoint "$S1_CKPT" \
    $COMMON_OPTS \
    OUTPUT_DIR "$OUT" \
    MODEL.TEXT_ENCODER_TYPE query_encoder \
    MODEL.I2T_LOSS_WEIGHT 0.2 \
    MODEL.TEXT_LOSS_TYPE cmpm \
    MODEL.TEXT_LOSS_WEIGHT 0.5
  log_msg "D1_cmpm completed"
}

STEP="${1:-all}"
case "$STEP" in
  D1_triplet) run_d1_triplet ;;
  D1_cmpm) run_d1_cmpm ;;
  all)
    run_d1_triplet
    run_d1_cmpm
    ;;
  *)
    echo "Usage: $0 [D1_triplet|D1_cmpm|all]"
    exit 1
    ;;
esac
