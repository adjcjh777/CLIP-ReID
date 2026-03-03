#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${CLIPREID_OUTPUT_ROOT:-${PROJECT_DIR}/OUTPUT}"
LOG_DIR="${OUTPUT_ROOT}/pipeline"
MASTER_LOG="${LOG_DIR}/master.log"
WATCH_LOG="${LOG_DIR}/watchdog.log"
FINAL_MARK="${LOG_DIR}/.p3_seed_4321.done"
C2_LOG="${OUTPUT_ROOT}/phase3/C2_market_blip/train_log.txt"
INTERVAL=60

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$WATCH_LOG"
}

start_pipeline_if_needed() {
  if pgrep -f "bash scripts/run_priority_pipeline.sh" >/dev/null; then
    return 0
  fi

  log "pipeline not running, restarting"
  setsid bash -lc "cd ${PROJECT_DIR} && bash scripts/run_priority_pipeline.sh >> ${MASTER_LOG} 2>&1" >/dev/null 2>&1 < /dev/null &
  sleep 2
  if pgrep -f "bash scripts/run_priority_pipeline.sh" >/dev/null; then
    log "pipeline restarted successfully"
  else
    log "pipeline restart failed"
  fi
}

shutdown_host() {
  log "all priority tasks completed, shutting down"
  sync || true
  shutdown -h now || systemctl poweroff || /sbin/poweroff || halt -p
}

log "watchdog started"

while true; do
  if [[ -f "$FINAL_MARK" ]]; then
    shutdown_host
    exit 0
  fi

  start_pipeline_if_needed

  if [[ -f "$C2_LOG" ]]; then
    last_line=$(tail -n 1 "$C2_LOG" 2>/dev/null || true)
    if [[ -n "$last_line" ]]; then
      log "C2 progress: $last_line"
    fi
  fi

  sleep "$INTERVAL"
done
