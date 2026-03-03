#!/bin/bash
# =============================================================================
# Phase 3: BLIP 标注生成 + MSMT17 & Market1501 训练
# 
# 使用方式：
#   bash scripts/run_phase3.sh [step]
#   step 可选: blip_msmt17, blip_market, train_c1, train_c2, all (默认)
# =============================================================================

set -e

# === 环境配置 ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PROJECT_DIR}/.venv-5090/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python"
fi
DATA_DIR="${CLIPREID_DATA_ROOT:-${PROJECT_DIR}/DATASETS}"
OUTPUT_ROOT="${CLIPREID_OUTPUT_ROOT:-${PROJECT_DIR}/OUTPUT}"
OUTPUT_BASE="${OUTPUT_ROOT}/phase3"
ANNO_DIR="${PROJECT_DIR}/annotations"
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.cache/huggingface}"
export HF_ENDPOINT="https://hf-mirror.com"

# BLIP 配置
BLIP_MODEL="Salesforce/blip2-opt-2.7b"
BLIP_BATCH_SIZE=8

# 训练配置（基于阶段 2 最优 B3a：I2T_LOSS_WEIGHT=0.2）
CONFIG_FILE="${PROJECT_DIR}/configs/person/vit_clipreid.yml"
SEED=1234
STAGE1_EPOCHS=60
STAGE2_EPOCHS=60
TRAIN_BATCH=32

STEP="${1:-all}"

cd "$PROJECT_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# =============================================================================
# Step 10a: BLIP 标注生成 - MSMT17
# =============================================================================
run_blip_msmt17() {
    log "=== Step 10a: 为 MSMT17 生成 BLIP-2 标注 ==="
    
    MSMT17_BLIP_OUTPUT="${ANNO_DIR}/msmt17_train_blip.json"
    
    if [ -f "$MSMT17_BLIP_OUTPUT" ]; then
        ENTRY_COUNT=$($PYTHON -c "import json; print(len(json.load(open('$MSMT17_BLIP_OUTPUT'))))")
        if [ "$ENTRY_COUNT" -gt 30000 ]; then
            log "MSMT17 BLIP 标注已存在 ($ENTRY_COUNT 条)，跳过"
            return 0
        fi
    fi
    
    $PYTHON scripts/generate_captions_blip.py \
        --image_dir "${DATA_DIR}/MSMT17/train" \
        --output_file "$MSMT17_BLIP_OUTPUT" \
        --model_name "$BLIP_MODEL" \
        --batch_size "$BLIP_BATCH_SIZE" \
        --dataset_type msmt17 \
        --prompt "Describe this person's appearance in detail, including their gender, age, clothing colors, clothing style, and any accessories they are carrying."
    
    # 验证标注数量
    ENTRY_COUNT=$($PYTHON -c "import json; print(len(json.load(open('$MSMT17_BLIP_OUTPUT'))))")
    log "MSMT17 BLIP 标注生成完毕：$ENTRY_COUNT 条"
    
    # 随机抽查 5 条
    log "=== 随机抽查标注质量 ==="
    $PYTHON -c "
import json, random
data = json.load(open('$MSMT17_BLIP_OUTPUT'))
keys = list(data.keys())
random.seed(42)
samples = random.sample(keys, min(5, len(keys)))
for k in samples:
    print(f'{k}: {data[k][\"text\"][:120]}')
    print()
"
}

# =============================================================================
# Step 10b: BLIP 标注生成 - Market1501（用于 C2 消融）
# =============================================================================
run_blip_market() {
    log "=== Step 10b: 为 Market1501 生成 BLIP-2 标注 ==="
    
    MARKET_BLIP_OUTPUT="${ANNO_DIR}/market1501_train_blip.json"
    
    if [ -f "$MARKET_BLIP_OUTPUT" ]; then
        ENTRY_COUNT=$($PYTHON -c "import json; print(len(json.load(open('$MARKET_BLIP_OUTPUT'))))")
        if [ "$ENTRY_COUNT" -gt 10000 ]; then
            log "Market1501 BLIP 标注已存在 ($ENTRY_COUNT 条)，跳过"
            return 0
        fi
    fi
    
    $PYTHON scripts/generate_captions_blip.py \
        --image_dir "${DATA_DIR}/Market-1501-v15.09.15/bounding_box_train" \
        --output_file "$MARKET_BLIP_OUTPUT" \
        --model_name "$BLIP_MODEL" \
        --batch_size "$BLIP_BATCH_SIZE" \
        --dataset_type market1501 \
        --prompt "Describe this person's appearance in detail, including their gender, age, clothing colors, clothing style, and any accessories they are carrying."
    
    ENTRY_COUNT=$($PYTHON -c "import json; print(len(json.load(open('$MARKET_BLIP_OUTPUT'))))")
    log "Market1501 BLIP 标注生成完毕：$ENTRY_COUNT 条"
    
    # 随机抽查 5 条
    log "=== 随机抽查标注质量 ==="
    $PYTHON -c "
import json, random
data = json.load(open('$MARKET_BLIP_OUTPUT'))
keys = list(data.keys())
random.seed(42)
samples = random.sample(keys, min(5, len(keys)))
for k in samples:
    print(f'{k}: {data[k][\"text\"][:120]}')
    print()
"
}

# =============================================================================
# Step 11 (C1): MSMT17 Text-Guided 训练（BLIP 标注）
# =============================================================================
run_train_c1() {
    log "=== Step 11 (C1): MSMT17 Text-Guided 训练 ==="
    
    MSMT17_BLIP_OUTPUT="${ANNO_DIR}/msmt17_train_blip.json"
    C1_OUTPUT="${OUTPUT_BASE}/C1_msmt17_blip"
    
    if [ ! -f "$MSMT17_BLIP_OUTPUT" ]; then
        log "ERROR: MSMT17 BLIP 标注不存在，请先运行 blip_msmt17"
        return 1
    fi
    
    mkdir -p "$C1_OUTPUT"
    
    $PYTHON train_text_reid.py \
        --config_file "$CONFIG_FILE" \
        --annotation_file "$MSMT17_BLIP_OUTPUT" \
        DATASETS.NAMES msmt17 \
        DATASETS.ROOT_DIR "$DATA_DIR" \
        SOLVER.SEED "$SEED" \
        SOLVER.STAGE1.IMS_PER_BATCH "$TRAIN_BATCH" \
        SOLVER.STAGE1.MAX_EPOCHS "$STAGE1_EPOCHS" \
        SOLVER.STAGE1.CHECKPOINT_PERIOD "$STAGE1_EPOCHS" \
        SOLVER.STAGE2.IMS_PER_BATCH "$TRAIN_BATCH" \
        SOLVER.STAGE2.MAX_EPOCHS "$STAGE2_EPOCHS" \
        OUTPUT_DIR "$C1_OUTPUT" \
        MODEL.TEXT_ENCODER_TYPE clip_native \
        MODEL.TEXT_LOSS_TYPE none \
        MODEL.I2T_LOSS_WEIGHT 0.2 \
        MODEL.SIE_CAMERA False \
        MODEL.STRIDE_SIZE "[16, 16]" \
        INPUT.SIZE_TRAIN "[256, 128]" \
        INPUT.SIZE_TEST "[256, 128]" \
        2>&1 | tee "${C1_OUTPUT}/run.log"
    
    log "C1 训练完成！结果保存在 $C1_OUTPUT"
}

# =============================================================================
# Step 12 (C2): Market1501 文本来源消融（属性模板 vs BLIP）
# =============================================================================
run_train_c2() {
    log "=== Step 12 (C2): Market1501 文本来源消融 ==="
    
    MARKET_BLIP_OUTPUT="${ANNO_DIR}/market1501_train_blip.json"
    C2_OUTPUT="${OUTPUT_BASE}/C2_market_blip"
    
    if [ ! -f "$MARKET_BLIP_OUTPUT" ]; then
        log "ERROR: Market1501 BLIP 标注不存在，请先运行 blip_market"
        return 1
    fi
    
    mkdir -p "$C2_OUTPUT"
    
    # 用 BLIP 标注训练 Market1501（对比 B3a 的属性模板标注）
    $PYTHON train_text_reid.py \
        --config_file "$CONFIG_FILE" \
        --annotation_file "$MARKET_BLIP_OUTPUT" \
        DATASETS.NAMES market1501 \
        DATASETS.ROOT_DIR "$DATA_DIR" \
        SOLVER.SEED "$SEED" \
        SOLVER.STAGE1.IMS_PER_BATCH "$TRAIN_BATCH" \
        SOLVER.STAGE1.MAX_EPOCHS "$STAGE1_EPOCHS" \
        SOLVER.STAGE1.CHECKPOINT_PERIOD "$STAGE1_EPOCHS" \
        SOLVER.STAGE2.IMS_PER_BATCH "$TRAIN_BATCH" \
        SOLVER.STAGE2.MAX_EPOCHS "$STAGE2_EPOCHS" \
        OUTPUT_DIR "$C2_OUTPUT" \
        MODEL.TEXT_ENCODER_TYPE clip_native \
        MODEL.TEXT_LOSS_TYPE none \
        MODEL.I2T_LOSS_WEIGHT 0.2 \
        MODEL.SIE_CAMERA False \
        MODEL.STRIDE_SIZE "[16, 16]" \
        INPUT.SIZE_TRAIN "[256, 128]" \
        INPUT.SIZE_TEST "[256, 128]" \
        2>&1 | tee "${C2_OUTPUT}/run.log"
    
    log "C2 训练完成！结果保存在 $C2_OUTPUT"
}

# =============================================================================
# 主流程
# =============================================================================
case "$STEP" in
    blip_msmt17)
        run_blip_msmt17
        ;;
    blip_market)
        run_blip_market
        ;;
    train_c1)
        run_train_c1
        ;;
    train_c2)
        run_train_c2
        ;;
    blip_all)
        run_blip_msmt17
        run_blip_market
        ;;
    all)
        log "=== Phase 3 完整流水线 ==="
        run_blip_msmt17
        run_blip_market
        log "=== BLIP 标注生成全部完成，开始训练 ==="
        run_train_c1
        run_train_c2
        log "=== Phase 3 完成 ==="
        ;;
    *)
        echo "Usage: $0 [blip_msmt17|blip_market|blip_all|train_c1|train_c2|all]"
        exit 1
        ;;
esac
