#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${CLIPREID_OUTPUT_ROOT:-${PROJECT_DIR}/OUTPUT}"
LOG_FILE="${OUTPUT_ROOT}/text_guided/market1501/train.log"
OUT_FILE="${OUTPUT_ROOT}/text_guided/market1501/eta.log"
INTERVAL=${INTERVAL:-120}

while true; do
  python "${PROJECT_DIR}/scripts/eta_monitor.py" --log "$LOG_FILE" --out "$OUT_FILE"
  sleep "$INTERVAL"
done
