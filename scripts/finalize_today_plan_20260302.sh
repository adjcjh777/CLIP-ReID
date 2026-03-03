#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${CLIPREID_OUTPUT_ROOT:-${PROJECT_DIR}/OUTPUT}"
SOURCE_DIR="${1:-${OUTPUT_ROOT}/today_plan_20260302_b5}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-30}"

raw_machine_tag="${CLIPREID_MACHINE_TAG:-$(hostname -s 2>/dev/null || echo local)}"
machine_tag="$(echo "${raw_machine_tag}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')"
if [[ -z "${machine_tag}" ]]; then
  machine_tag="local"
fi

target_base="${OUTPUT_ROOT}/phase2"
target_dir="${target_base}/B5_i2t_w0.2_sie_olp_${machine_tag}"

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "[FINALIZE] Source directory not found: ${SOURCE_DIR}"
  exit 1
fi

if [[ "${SOURCE_DIR}" == "${target_dir}" ]]; then
  echo "[FINALIZE] Source already uses final naming: ${SOURCE_DIR}"
  exit 0
fi

if [[ -f "${SOURCE_DIR}/launcher.pid" ]]; then
  launcher_pid="$(cat "${SOURCE_DIR}/launcher.pid" 2>/dev/null || true)"
  if [[ "${launcher_pid}" =~ ^[0-9]+$ ]]; then
    echo "[FINALIZE] Waiting for launcher PID ${launcher_pid} to exit..."
    while kill -0 "${launcher_pid}" 2>/dev/null; do
      sleep "${CHECK_INTERVAL_SEC}"
    done
  fi
fi

echo "[FINALIZE] Waiting for training outputs to be ready..."
while true; do
  if [[ -f "${SOURCE_DIR}/ViT-B-16_stage2_60.pth" && -f "${SOURCE_DIR}/eval_baseline.log" && -f "${SOURCE_DIR}/eval_rerank.log" ]]; then
    break
  fi
  sleep "${CHECK_INTERVAL_SEC}"
done

mkdir -p "${target_base}"
final_target="${target_dir}"
if [[ -e "${final_target}" ]]; then
  final_target="${target_dir}_$(date '+%Y%m%d_%H%M%S')"
fi

mv "${SOURCE_DIR}" "${final_target}"
ln -sfn "${final_target}" "${SOURCE_DIR}"

echo "[FINALIZE] Renamed experiment directory:"
echo "[FINALIZE]   ${final_target}"
echo "[FINALIZE] Legacy alias retained:"
echo "[FINALIZE]   ${SOURCE_DIR} -> ${final_target}"
