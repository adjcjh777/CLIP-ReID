#!/usr/bin/env bash
# E5: Re-ranking 评估 —— 对 B5 best checkpoint 做无训练评估（quick win，无需 GPU 训练）
# 预期效果: mAP +0.5~1.5%，R1 +0.2~0.5%
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PROJECT_DIR}/.venv-5090/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python"
fi
DATA_ROOT="${CLIPREID_DATA_ROOT:-${PROJECT_DIR}/DATASETS}"
OUTPUT_ROOT="${CLIPREID_OUTPUT_ROOT:-${PROJECT_DIR}/OUTPUT}"
CKPT="${OUTPUT_ROOT}/phase2/B5_i2t_w0.2_sie_olp/ViT-B-16_stage2_60.pth"
CONFIG="configs/person/vit_clipreid.yml"

cd "$PROJECT_DIR"

echo "============================================================"
echo "[E5] Re-ranking 评估 on B5 checkpoint"
echo "CKPT: $CKPT"
echo "============================================================"

# --- 1. 不启用 re-ranking（baseline confirm）---
echo ""
echo "--- E5-baseline: 不开 re-ranking ---"
$PYTHON scripts/eval_checkpoint.py \
    --config_file "$CONFIG" \
    --checkpoint "$CKPT" \
    DATASETS.NAMES market1501 \
    DATASETS.ROOT_DIR "$DATA_ROOT" \
    TEST.FEAT_NORM yes \
    TEST.RE_RANKING False \
    MODEL.PRETRAIN_PATH "" \
    MODEL.TEXT_ENCODER_TYPE clip_native \
    MODEL.STRIDE_SIZE "[12, 12]" \
    MODEL.SIE_CAMERA True \
    MODEL.SIE_COE 3.0

echo ""
echo "--- E5-rr: 开启 re-ranking (k1=50, k2=15, λ=0.3) ---"
$PYTHON scripts/eval_checkpoint.py \
    --config_file "$CONFIG" \
    --checkpoint "$CKPT" \
    --reranking \
    DATASETS.NAMES market1501 \
    DATASETS.ROOT_DIR "$DATA_ROOT" \
    TEST.FEAT_NORM yes \
    MODEL.PRETRAIN_PATH "" \
    MODEL.TEXT_ENCODER_TYPE clip_native \
    MODEL.STRIDE_SIZE "[12, 12]" \
    MODEL.SIE_CAMERA True \
    MODEL.SIE_COE 3.0

echo ""
echo "[E5] Done. 对比两行 mAP 数字即可得出 re-ranking 收益。"
