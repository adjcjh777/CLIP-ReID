#!/bin/bash
set -e
LOG_FILE="/root/autodl-tmp/CLIP_REID/OUTPUT/text_guided/market1501/train.log"
OUT_FILE="/root/autodl-tmp/CLIP_REID/OUTPUT/text_guided/market1501/eta.log"
INTERVAL=${INTERVAL:-120}

while true; do
  python /root/CLIP-ReID/scripts/eta_monitor.py --log "$LOG_FILE" --out "$OUT_FILE"
  sleep "$INTERVAL"
done
