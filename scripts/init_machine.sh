#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv-5090"
REQUIREMENTS_FILE="${PROJECT_DIR}/requirements/base.txt"

WORK_ROOT="${CLIPREID_WORK_ROOT:-${PROJECT_DIR}}"
DATA_ROOT="${CLIPREID_DATA_ROOT:-${WORK_ROOT}/DATASETS}"
OUTPUT_ROOT="${CLIPREID_OUTPUT_ROOT:-${WORK_ROOT}/OUTPUT}"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "No Python interpreter found in PATH." >&2
  exit 1
fi

echo "[1/6] Project directory: ${PROJECT_DIR}"
echo "[2/6] Python interpreter: ${PYTHON_BIN} ($(${PYTHON_BIN} --version 2>&1))"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[3/6] Creating virtualenv at ${VENV_DIR}"
  # Reuse preinstalled site packages (e.g. torch/cu113) to avoid large reinstall.
  "${PYTHON_BIN}" -m venv "${VENV_DIR}" --system-site-packages
else
  echo "[3/6] Virtualenv already exists: ${VENV_DIR}"
fi

echo "[4/6] Installing Python dependencies from ${REQUIREMENTS_FILE}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
if ! "${VENV_DIR}/bin/python" -m pip install -r "${REQUIREMENTS_FILE}"; then
  echo "Primary package index failed, retrying from official PyPI."
  "${VENV_DIR}/bin/python" -m pip install -i https://pypi.org/simple -r "${REQUIREMENTS_FILE}"
fi

echo "[5/6] Preparing data/output paths"
mkdir -p "${DATA_ROOT}" "${OUTPUT_ROOT}"

echo "[6/6] Running environment smoke checks"
"${VENV_DIR}/bin/python" - <<'PY'
import importlib
import sys

required = [
    "torch",
    "torchvision",
    "timm",
    "yacs",
    "ftfy",
    "regex",
    "skimage",
    "transformers",
    "accelerate",
    "pptx",
]

missing = []
for mod in required:
    try:
        importlib.import_module(mod)
    except Exception as exc:
        missing.append((mod, str(exc)))

if missing:
    print("Missing dependencies:")
    for mod, err in missing:
        print(f"  - {mod}: {err}")
    sys.exit(1)

import torch
print(f"Python: {sys.version.split()[0]}")
print(f"Torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
PY

cat <<EOF

Initialization completed.

Use this Python interpreter:
  ${VENV_DIR}/bin/python

Default data/output roots:
  DATASETS: ${DATA_ROOT}
  OUTPUT:   ${OUTPUT_ROOT}

Quick check:
  ${VENV_DIR}/bin/python train_clipreid.py --help
EOF
