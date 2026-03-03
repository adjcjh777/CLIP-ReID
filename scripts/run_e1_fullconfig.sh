#!/usr/bin/env bash
# E1: Full-config Recovery —— 恢复论文完整训练配置
#   batch=64, Stage1=120ep + Stage2=60ep（单次运行，内部衔接）
#   SIE_CAMERA=True, SIE_COE=3.0, STRIDE_SIZE=[12,12], I2T=0.2, clip_native
# 预期效果: 关闭与原文 0.8% mAP 差距 → 达到 ~89.4-89.6%
# 注意: train_text_reid.py 单次调用完成 Stage1+Stage2，无需分两次运行
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PROJECT_DIR}/.venv-5090/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python"
fi
DATA_ROOT="${CLIPREID_DATA_ROOT:-${PROJECT_DIR}/DATASETS}"
OUTPUT_ROOT="${CLIPREID_OUTPUT_ROOT:-${PROJECT_DIR}/OUTPUT}"
OUT_ROOT="${OUTPUT_ROOT}/E1_fullconfig"
CONFIG="configs/person/vit_clipreid.yml"
ANNOTATION="annotations/market1501_train.json"

mkdir -p "$OUT_ROOT"
cd "$PROJECT_DIR"

echo "============================================================"
echo "[E1] Full-config Recovery"
echo "     Stage1: 120ep, batch=64 → Stage2: 60ep, batch=64"
echo "     SIE_CAMERA=True, SIE_COE=3.0, STRIDE=[12,12], I2T=0.2"
echo "Output: $OUT_ROOT"
echo "Start : $(date)"
echo "============================================================"

$PYTHON train_text_reid.py \
    --config_file "$CONFIG" \
    --annotation_file "$ANNOTATION" \
    DATASETS.NAMES market1501 \
    DATASETS.ROOT_DIR "$DATA_ROOT" \
    MODEL.TEXT_ENCODER_TYPE clip_native \
    MODEL.SIE_CAMERA True \
    MODEL.SIE_COE 3.0 \
    MODEL.STRIDE_SIZE "[12, 12]" \
    MODEL.I2T_LOSS_WEIGHT 0.2 \
    SOLVER.SEED 1234 \
    SOLVER.STAGE1.IMS_PER_BATCH 64 \
    SOLVER.STAGE1.MAX_EPOCHS 120 \
    SOLVER.STAGE2.IMS_PER_BATCH 64 \
    SOLVER.STAGE2.MAX_EPOCHS 60 \
    INPUT.SIZE_TRAIN "[256, 128]" \
    INPUT.SIZE_TEST "[256, 128]" \
    OUTPUT_DIR "$OUT_ROOT" \
    TEST.EVAL True

echo ""
echo "[E1] 全部完成！End: $(date)"
echo "结果在 ${OUT_ROOT}/train_log.txt"
