#!/usr/bin/env bash
# Thin wrapper around yolo/run.py
#
# Usage:
#   ./scripts/yolo.sh train --model yolov5 --epochs 1
#   ./scripts/yolo.sh models
#
# Environment:
#   PYTHON  Python executable (default: python3)

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
YOLO_RUN_PY="${REPO_ROOT}/yolo/run.py"
PYTHON="${PYTHON:-python3}"

if [[ ! -f "${YOLO_RUN_PY}" ]]; then
  echo "[ERROR] Not found: ${YOLO_RUN_PY}" >&2
  exit 1
fi

cd "${REPO_ROOT}/yolo"
exec "${PYTHON}" "${YOLO_RUN_PY}" "$@"
