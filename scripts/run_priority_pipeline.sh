#!/usr/bin/env bash
# =======================================================================
# Priority Pipeline Runner
#
# 执行顺序：
#   1) P0: C2 (Market BLIP)
#   2) P1: B4 best combo
#   3) P2: D1 triplet + cmpm
#   4) P3: final multi-seed (1234, 777, 4321)
# =======================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${CLIPREID_OUTPUT_ROOT:-${PROJECT_DIR}/OUTPUT}"
LOG_DIR="${OUTPUT_ROOT}/pipeline"
mkdir -p "$LOG_DIR"

C2_OUT="${OUTPUT_ROOT}/phase3/C2_market_blip"
C2_CKPT="${C2_OUT}/ViT-B-16_stage2_60.pth"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

wait_c2_finish() {
  log "[P0] checking C2 status..."

  if [ -f "$C2_CKPT" ]; then
    log "[P0] C2 already completed"
    return 0
  fi

  if ! ps -ef | grep -E 'train_text_reid.py.*C2_market_blip|run_phase3.sh train_c2' | grep -v grep >/dev/null; then
    log "[P0] C2 not running, starting it"
    (cd "$PROJECT_DIR" && bash scripts/run_phase3.sh train_c2) >> "$LOG_DIR/p0_c2.log" 2>&1
  fi

  while true; do
    if [ -f "$C2_CKPT" ]; then
      log "[P0] C2 finished"
      break
    fi

    if ! ps -ef | grep -E 'train_text_reid.py.*C2_market_blip|run_phase3.sh train_c2' | grep -v grep >/dev/null; then
      log "[P0] C2 stopped unexpectedly, restarting"
      (cd "$PROJECT_DIR" && bash scripts/run_phase3.sh train_c2) >> "$LOG_DIR/p0_c2.log" 2>&1
    fi

    sleep 60
  done
}

run_step() {
  local step="$1"
  local cmd="$2"
  local mark="$3"
  if [ -f "$mark" ]; then
    log "[$step] already done: $mark"
    return 0
  fi

  log "[$step] starting"
  (cd "$PROJECT_DIR" && eval "$cmd") >> "$LOG_DIR/${step}.log" 2>&1
  touch "$mark"
  log "[$step] completed"
}

wait_c2_finish

run_step "p1_b4" "bash scripts/run_b4_best_combo.sh" "$LOG_DIR/.p1_b4.done"
run_step "p2_d1_triplet" "bash scripts/run_phase4_loss_ablation.sh D1_triplet" "$LOG_DIR/.p2_d1_triplet.done"
run_step "p2_d1_cmpm" "bash scripts/run_phase4_loss_ablation.sh D1_cmpm" "$LOG_DIR/.p2_d1_cmpm.done"
run_step "p3_seed_1234" "bash scripts/run_multiseed_final.sh 1234" "$LOG_DIR/.p3_seed_1234.done"
run_step "p3_seed_777" "bash scripts/run_multiseed_final.sh 777" "$LOG_DIR/.p3_seed_777.done"
run_step "p3_seed_4321" "bash scripts/run_multiseed_final.sh 4321" "$LOG_DIR/.p3_seed_4321.done"

log "Pipeline finished"
