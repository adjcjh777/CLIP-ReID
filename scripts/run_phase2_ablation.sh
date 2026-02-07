#!/usr/bin/env bash
# =======================================================================
# Phase 2 消融实验 — 自动化批量训练脚本
# Market-1501, seed=1234, ViT-B-16
#
# 设计思路:
#   - B0 = 对照基线 (clip_native, no cross-modal loss) → 需完整 Stage1+Stage2
#   - B2, B3a-B3d = 复用 B0 的 Stage1 checkpoint → 仅 Stage2
#   - B1 = query_encoder → 需完整 Stage1+Stage2
#   - B4 = 三者最优组合 → 需人工决定后运行
#
# 用法:
#   bash scripts/run_phase2_ablation.sh [experiment_id]
#   不指定 experiment_id 时，按顺序运行所有实验 (B0→B2→B3a→B3b→B3c→B3d→B1)
#   指定时只运行单个实验，如: bash scripts/run_phase2_ablation.sh B3a
# =======================================================================

set -e

PYTHON="/root/miniconda3/bin/python"
PROJECT_DIR="/root/CLIP-ReID"
CONFIG="configs/person/vit_clipreid.yml"
ANNOTATION="annotations/market1501_train.json"
DATASETS_ROOT="/root/autodl-tmp/CLIP_REID/DATASETS"
OUTPUT_BASE="/root/autodl-tmp/CLIP_REID/OUTPUT/phase2"
XLSX_PATH="${PROJECT_DIR}/docs/experiments/phase2_ablation_results.xlsx"

# Common overrides
COMMON_OPTS="DATASETS.NAMES market1501 DATASETS.ROOT_DIR ${DATASETS_ROOT} SOLVER.SEED 1234"

cd "$PROJECT_DIR"

log_msg() {
    echo ""
    echo "========================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "========================================"
    echo ""
}

update_xlsx() {
    local exp_id="$1"
    local status="$2"
    local output_dir="$3"
    $PYTHON scripts/update_experiment_results.py \
        --xlsx_path "$XLSX_PATH" \
        --exp_id "$exp_id" \
        --status "$status" \
        --output_dir "$output_dir" 2>/dev/null || echo "Warning: xlsx update failed for $exp_id"
}

# ---------- B0: 对照基线 (完整 Stage1+Stage2) ----------
run_B0() {
    local EXP_ID="B0"
    local OUT="${OUTPUT_BASE}/B0_baseline_clip_native"
    log_msg "Starting ${EXP_ID}: 对照基线 (clip_native, TEXT_LOSS_TYPE=none)"
    update_xlsx "$EXP_ID" "running" "$OUT"

    $PYTHON train_text_reid.py \
        --config_file $CONFIG \
        --annotation_file $ANNOTATION \
        $COMMON_OPTS \
        OUTPUT_DIR "$OUT" \
        MODEL.TEXT_ENCODER_TYPE clip_native \
        MODEL.TEXT_LOSS_TYPE none \
        MODEL.I2T_LOSS_WEIGHT 1.0

    log_msg "${EXP_ID} completed"
    update_xlsx "$EXP_ID" "completed" "$OUT"
}

# ---------- B2: +CrossModal contrastive loss ----------
run_B2() {
    local EXP_ID="B2"
    local OUT="${OUTPUT_BASE}/B2_crossmodal_contrastive"
    local S1_CKPT="${OUTPUT_BASE}/B0_baseline_clip_native/ViT-B-16_stage1_120.pth"

    if [ ! -f "$S1_CKPT" ]; then
        echo "ERROR: B0 Stage1 checkpoint not found at $S1_CKPT. Run B0 first."
        return 1
    fi

    log_msg "Starting ${EXP_ID}: +CrossModalContrastiveLoss (TEXT_LOSS_TYPE=contrastive, WEIGHT=0.5)"
    update_xlsx "$EXP_ID" "running" "$OUT"

    $PYTHON train_text_reid.py \
        --config_file $CONFIG \
        --annotation_file $ANNOTATION \
        --skip_stage1 \
        --stage1_checkpoint "$S1_CKPT" \
        $COMMON_OPTS \
        OUTPUT_DIR "$OUT" \
        MODEL.TEXT_ENCODER_TYPE clip_native \
        MODEL.TEXT_LOSS_TYPE contrastive \
        MODEL.TEXT_LOSS_WEIGHT 0.5 \
        MODEL.I2T_LOSS_WEIGHT 1.0

    log_msg "${EXP_ID} completed"
    update_xlsx "$EXP_ID" "completed" "$OUT"
}

# ---------- B3a: I2T_LOSS_WEIGHT=0.2 ----------
run_B3a() {
    local EXP_ID="B3a"
    local OUT="${OUTPUT_BASE}/B3a_i2t_w0.2"
    local S1_CKPT="${OUTPUT_BASE}/B0_baseline_clip_native/ViT-B-16_stage1_120.pth"

    if [ ! -f "$S1_CKPT" ]; then
        echo "ERROR: B0 Stage1 checkpoint not found. Run B0 first."
        return 1
    fi

    log_msg "Starting ${EXP_ID}: I2T_LOSS_WEIGHT=0.2"
    update_xlsx "$EXP_ID" "running" "$OUT"

    $PYTHON train_text_reid.py \
        --config_file $CONFIG \
        --annotation_file $ANNOTATION \
        --skip_stage1 \
        --stage1_checkpoint "$S1_CKPT" \
        $COMMON_OPTS \
        OUTPUT_DIR "$OUT" \
        MODEL.TEXT_ENCODER_TYPE clip_native \
        MODEL.TEXT_LOSS_TYPE none \
        MODEL.I2T_LOSS_WEIGHT 0.2

    log_msg "${EXP_ID} completed"
    update_xlsx "$EXP_ID" "completed" "$OUT"
}

# ---------- B3b: I2T_LOSS_WEIGHT=0.5 ----------
run_B3b() {
    local EXP_ID="B3b"
    local OUT="${OUTPUT_BASE}/B3b_i2t_w0.5"
    local S1_CKPT="${OUTPUT_BASE}/B0_baseline_clip_native/ViT-B-16_stage1_120.pth"

    if [ ! -f "$S1_CKPT" ]; then
        echo "ERROR: B0 Stage1 checkpoint not found. Run B0 first."
        return 1
    fi

    log_msg "Starting ${EXP_ID}: I2T_LOSS_WEIGHT=0.5"
    update_xlsx "$EXP_ID" "running" "$OUT"

    $PYTHON train_text_reid.py \
        --config_file $CONFIG \
        --annotation_file $ANNOTATION \
        --skip_stage1 \
        --stage1_checkpoint "$S1_CKPT" \
        $COMMON_OPTS \
        OUTPUT_DIR "$OUT" \
        MODEL.TEXT_ENCODER_TYPE clip_native \
        MODEL.TEXT_LOSS_TYPE none \
        MODEL.I2T_LOSS_WEIGHT 0.5

    log_msg "${EXP_ID} completed"
    update_xlsx "$EXP_ID" "completed" "$OUT"
}

# ---------- B3c: I2T_LOSS_WEIGHT=1.0 (same as B0 but Stage2 only, control) ----------
run_B3c() {
    local EXP_ID="B3c"
    local OUT="${OUTPUT_BASE}/B3c_i2t_w1.0"
    local S1_CKPT="${OUTPUT_BASE}/B0_baseline_clip_native/ViT-B-16_stage1_120.pth"

    if [ ! -f "$S1_CKPT" ]; then
        echo "ERROR: B0 Stage1 checkpoint not found. Run B0 first."
        return 1
    fi

    log_msg "Starting ${EXP_ID}: I2T_LOSS_WEIGHT=1.0 (Stage2重训对照)"
    update_xlsx "$EXP_ID" "running" "$OUT"

    $PYTHON train_text_reid.py \
        --config_file $CONFIG \
        --annotation_file $ANNOTATION \
        --skip_stage1 \
        --stage1_checkpoint "$S1_CKPT" \
        $COMMON_OPTS \
        OUTPUT_DIR "$OUT" \
        MODEL.TEXT_ENCODER_TYPE clip_native \
        MODEL.TEXT_LOSS_TYPE none \
        MODEL.I2T_LOSS_WEIGHT 1.0

    log_msg "${EXP_ID} completed"
    update_xlsx "$EXP_ID" "completed" "$OUT"
}

# ---------- B3d: I2T_LOSS_WEIGHT=2.0 ----------
run_B3d() {
    local EXP_ID="B3d"
    local OUT="${OUTPUT_BASE}/B3d_i2t_w2.0"
    local S1_CKPT="${OUTPUT_BASE}/B0_baseline_clip_native/ViT-B-16_stage1_120.pth"

    if [ ! -f "$S1_CKPT" ]; then
        echo "ERROR: B0 Stage1 checkpoint not found. Run B0 first."
        return 1
    fi

    log_msg "Starting ${EXP_ID}: I2T_LOSS_WEIGHT=2.0"
    update_xlsx "$EXP_ID" "running" "$OUT"

    $PYTHON train_text_reid.py \
        --config_file $CONFIG \
        --annotation_file $ANNOTATION \
        --skip_stage1 \
        --stage1_checkpoint "$S1_CKPT" \
        $COMMON_OPTS \
        OUTPUT_DIR "$OUT" \
        MODEL.TEXT_ENCODER_TYPE clip_native \
        MODEL.TEXT_LOSS_TYPE none \
        MODEL.I2T_LOSS_WEIGHT 2.0

    log_msg "${EXP_ID} completed"
    update_xlsx "$EXP_ID" "completed" "$OUT"
}

# ---------- B1: TextQueryEncoder (完整 Stage1+Stage2) ----------
run_B1() {
    local EXP_ID="B1"
    local OUT="${OUTPUT_BASE}/B1_query_encoder"

    log_msg "Starting ${EXP_ID}: TextQueryEncoder (query_encoder, 完整训练)"
    update_xlsx "$EXP_ID" "running" "$OUT"

    $PYTHON train_text_reid.py \
        --config_file $CONFIG \
        --annotation_file $ANNOTATION \
        $COMMON_OPTS \
        OUTPUT_DIR "$OUT" \
        MODEL.TEXT_ENCODER_TYPE query_encoder \
        MODEL.TEXT_LOSS_TYPE none \
        MODEL.I2T_LOSS_WEIGHT 1.0

    log_msg "${EXP_ID} completed"
    update_xlsx "$EXP_ID" "completed" "$OUT"
}

# ---------- Main ----------
EXPERIMENT="${1:-all}"

case "$EXPERIMENT" in
    B0)   run_B0 ;;
    B2)   run_B2 ;;
    B3a)  run_B3a ;;
    B3b)  run_B3b ;;
    B3c)  run_B3c ;;
    B3d)  run_B3d ;;
    B1)   run_B1 ;;
    all)
        log_msg "Running ALL Phase 2 experiments (B0→B2→B3a→B3b→B3c→B3d→B1)"
        run_B0
        run_B2
        run_B3a
        run_B3b
        run_B3c
        run_B3d
        run_B1
        log_msg "ALL Phase 2 experiments completed!"
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Usage: $0 [B0|B1|B2|B3a|B3b|B3c|B3d|all]"
        exit 1
        ;;
esac
