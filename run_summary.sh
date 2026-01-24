#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="/root/autodl-tmp/CLIP_REID/OUTPUT"
OUT_FILE="${OUTPUT_ROOT}/summary_$(date +"%Y%m%d_%H%M%S").tsv"

cd /root/CLIP-ReID

# per-run + summary + baseline vs SIE+OLP comparison
./summarize_clipreid_results.py --output-root "${OUTPUT_ROOT}" --per-run --summary --compare | tee "${OUT_FILE}"

echo "[INFO] Summary saved to ${OUT_FILE}"
