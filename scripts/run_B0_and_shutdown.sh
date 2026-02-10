#!/usr/bin/env bash
# =======================================================================
# B0 修正版训练 + 训练完自动关机
# 修正: batch=32, Stage1=60ep（与成功的 text-guided 配置一致）
# 预计耗时: ~3 小时 (Stage1 60ep + Stage2 60ep, batch=32)
# =======================================================================

set -e

echo "============================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] B0 修正版训练开始"
echo "修正内容: batch 64→32, Stage1 epoch 120→60"
echo "============================================"

cd /root/CLIP-ReID

/root/miniconda3/bin/python train_text_reid.py \
    --config_file configs/person/vit_clipreid.yml \
    --annotation_file annotations/market1501_train.json \
    DATASETS.NAMES market1501 \
    DATASETS.ROOT_DIR /root/autodl-tmp/CLIP_REID/DATASETS \
    SOLVER.SEED 1234 \
    SOLVER.STAGE1.IMS_PER_BATCH 32 \
    SOLVER.STAGE1.MAX_EPOCHS 60 \
    SOLVER.STAGE1.CHECKPOINT_PERIOD 60 \
    SOLVER.STAGE2.IMS_PER_BATCH 32 \
    OUTPUT_DIR /root/autodl-tmp/CLIP_REID/OUTPUT/phase2/B0_baseline_clip_native \
    MODEL.TEXT_ENCODER_TYPE clip_native \
    MODEL.TEXT_LOSS_TYPE none \
    MODEL.I2T_LOSS_WEIGHT 1.0

EXIT_CODE=$?

echo ""
echo "============================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] B0 训练完成 (exit code: $EXIT_CODE)"
echo "结果目录: /root/autodl-tmp/CLIP_REID/OUTPUT/phase2/B0_baseline_clip_native"
echo "============================================"

# 训练完成后关机
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 即将关机..."
sleep 10
shutdown -h now
