#!/usr/bin/env bash
# E3: Scheduled I2T Weight —— 高权重预热 → 低权重精调，两阶段 Stage2
#   思路：Stage2 前 30ep 用 I2T=1.0 强监督对齐，后 30ep 用 I2T=0.2 精调分类
#   无需修改训练代码，以 checkpoint 串联两次 Stage2 实现
# 预期效果：vs B5 mAP +0.2~0.5%
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PROJECT_DIR}/.venv-5090/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python"
fi
DATA_ROOT="${CLIPREID_DATA_ROOT:-${PROJECT_DIR}/DATASETS}"
OUTPUT_ROOT="${CLIPREID_OUTPUT_ROOT:-${PROJECT_DIR}/OUTPUT}"
OUT_ROOT="${OUTPUT_ROOT}/E3_sched_i2t"
CONFIG="configs/person/vit_clipreid.yml"

# 用 B5 的 Stage1 checkpoint（省略 Stage1 重训，已有 60ep 版本）
ANNOTATION="annotations/market1501_train.json"
# 使用 E1 的 Stage1 checkpoint（STRIDE_SIZE=[12,12]，与 E3 配置匹配）
# B1 Stage1 是 stride=16 [129,768]，会与 E3 的 stride=12 [211,768] 维度不匹配
S1_CKPT="${OUTPUT_ROOT}/E1_fullconfig/ViT-B-16_stage1_120.pth"

mkdir -p "$OUT_ROOT/s2a" "$OUT_ROOT/s2b"
cd "$PROJECT_DIR"

echo "============================================================"
echo "[E3] Scheduled I2T: 1.0→0.2 (两阶段 Stage2, 各30ep)"
echo "Stage1 ckpt: $S1_CKPT"
echo "============================================================"

# --- Stage 2a: 前 30ep，I2T=1.0，强文本对齐 ---
echo ""
echo "--- Stage 2a: I2T=1.0, 30 epochs ---"
$PYTHON train_text_reid.py \
    --config_file "$CONFIG" \
    --annotation_file "$ANNOTATION" \
    --skip_stage1 \
    --stage1_checkpoint "$S1_CKPT" \
    DATASETS.NAMES market1501 \
    DATASETS.ROOT_DIR "$DATA_ROOT" \
    MODEL.TEXT_ENCODER_TYPE clip_native \
    MODEL.SIE_CAMERA True \
    MODEL.SIE_COE 3.0 \
    MODEL.STRIDE_SIZE "[12, 12]" \
    MODEL.I2T_LOSS_WEIGHT 1.0 \
    SOLVER.STAGE1.IMS_PER_BATCH 32 \
    SOLVER.STAGE2.IMS_PER_BATCH 32 \
    SOLVER.STAGE2.MAX_EPOCHS 30 \
    SOLVER.STAGE2.BASE_LR 0.000035 \
    SOLVER.STAGE2.WARMUP_EPOCHS 5 \
    INPUT.SIZE_TRAIN "[256, 128]" \
    INPUT.SIZE_TEST "[256, 128]" \
    OUTPUT_DIR "${OUT_ROOT}/s2a" \
    TEST.EVAL False

S2A_CKPT="${OUT_ROOT}/s2a/ViT-B-16_stage2_30.pth"
echo "Stage 2a 完成，checkpoint: $S2A_CKPT"

# --- Stage 2b: 后 30ep，I2T=0.2，低权重精调 ---
echo ""
echo "--- Stage 2b: I2T=0.2, 30 epochs ---"
$PYTHON train_text_reid.py \
    --config_file "$CONFIG" \
    --annotation_file "$ANNOTATION" \
    --skip_stage1 \
    --stage1_checkpoint "$S2A_CKPT" \
    DATASETS.NAMES market1501 \
    DATASETS.ROOT_DIR "$DATA_ROOT" \
    MODEL.TEXT_ENCODER_TYPE clip_native \
    MODEL.SIE_CAMERA True \
    MODEL.SIE_COE 3.0 \
    MODEL.STRIDE_SIZE "[12, 12]" \
    MODEL.I2T_LOSS_WEIGHT 0.2 \
    SOLVER.STAGE1.IMS_PER_BATCH 32 \
    SOLVER.STAGE2.IMS_PER_BATCH 32 \
    SOLVER.STAGE2.MAX_EPOCHS 30 \
    SOLVER.STAGE2.BASE_LR 0.0000035 \
    SOLVER.STAGE2.WARMUP_EPOCHS 0 \
    INPUT.SIZE_TRAIN "[256, 128]" \
    INPUT.SIZE_TEST "[256, 128]" \
    OUTPUT_DIR "${OUT_ROOT}/s2b" \
    TEST.EVAL True

echo ""
echo "[E3] 全部完成！结果在 ${OUT_ROOT}/s2b/train_log.txt"
