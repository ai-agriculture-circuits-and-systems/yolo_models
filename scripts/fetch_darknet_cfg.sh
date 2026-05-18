#!/usr/bin/env bash
# Download Darknet .cfg files for YOLOv1/v2/v4 training (AlexeyAB + pjreddie).
#
# Usage:
#   ./scripts/fetch_darknet_cfg.sh

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
# shellcheck source=lib/dataset_common.sh
source "${_SCRIPT_DIR}/lib/dataset_common.sh"

CFG_DIR="${REPO_ROOT}/yolo/darknet/cfg"
mkdir -p "${CFG_DIR}"

fetch_cfg() {
  local url="$1"
  local dest="$2"
  if [[ -f "${dest}" && "${FORCE:-0}" != "1" ]]; then
    log_info "Skipping existing: ${dest}"
    return 0
  fi
  download_file "${url}" "${dest}"
}

log_info "Fetching Darknet cfg files into ${CFG_DIR}"

fetch_cfg \
  "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov1.cfg" \
  "${CFG_DIR}/yolov1.cfg"
fetch_cfg \
  "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2-voc.cfg" \
  "${CFG_DIR}/yolov2-voc.cfg"
fetch_cfg \
  "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov2-tiny.cfg" \
  "${CFG_DIR}/yolov2-tiny.cfg"
fetch_cfg \
  "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg" \
  "${CFG_DIR}/yolov4.cfg"
fetch_cfg \
  "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg" \
  "${CFG_DIR}/yolov4-tiny.cfg"
fetch_cfg \
  "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-csp.cfg" \
  "${CFG_DIR}/yolov4-csp.cfg"
fetch_cfg \
  "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4x-mish.cfg" \
  "${CFG_DIR}/yolov4x-mish.cfg"

log_info "Darknet cfg fetch complete."
