#!/usr/bin/env bash
# Build AlexeyAB/darknet (git submodule at tools/darknet-src) and install tools/darknet/darknet.
#
# Usage:
#   ./scripts/install_darknet.sh
#   ./scripts/install_darknet.sh --cpu
#   ./scripts/install_darknet.sh --init-submodule   # only init/update submodule, no build
#
# First-time repo clone:
#   git clone --recurse-submodules <repo-url>
#   # or: git submodule update --init tools/darknet-src
#
# Requires: git, make, g++, (optional) OpenCV dev headers for OPENCV=1

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
# shellcheck source=lib/dataset_common.sh
source "${_SCRIPT_DIR}/lib/dataset_common.sh"

SUBMODULE_PATH="tools/darknet-src"
SRC_DIR="${REPO_ROOT}/${SUBMODULE_PATH}"
INSTALL_DIR="${REPO_ROOT}/tools/darknet"
GPU_FLAG=1
OPENCV_FLAG=1
INIT_SUBMODULE_ONLY=false

while [[ $# -gt 0 ]]; do
  case "$1" in
  --cpu)
    GPU_FLAG=0
    shift
    ;;
  --no-opencv)
    OPENCV_FLAG=0
    shift
    ;;
  --init-submodule)
    INIT_SUBMODULE_ONLY=true
    shift
    ;;
  -h | --help)
    grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -18
    exit 0
    ;;
  *)
    log_error "Unknown option: $1"
    exit 1
    ;;
  esac
done

require_cmd git

ensure_darknet_submodule() {
  if [[ ! -f "${REPO_ROOT}/.gitmodules" ]]; then
    log_error "Missing .gitmodules. Add the submodule with:"
    log_error "  git submodule add https://github.com/AlexeyAB/darknet.git ${SUBMODULE_PATH}"
    exit 1
  fi

  if [[ ! -d "${SRC_DIR}" ]] || [[ -z "$(ls -A "${SRC_DIR}" 2>/dev/null || true)" ]]; then
    log_info "Initializing submodule ${SUBMODULE_PATH}..."
    git -C "${REPO_ROOT}" submodule update --init --depth 1 "${SUBMODULE_PATH}"
  else
    log_info "Updating submodule ${SUBMODULE_PATH}..."
    git -C "${REPO_ROOT}" submodule update --init --recursive --depth 1 "${SUBMODULE_PATH}" || true
  fi

  if [[ ! -f "${SRC_DIR}/Makefile" ]]; then
    log_error "Submodule checkout incomplete: ${SRC_DIR}/Makefile not found"
    exit 1
  fi
  log_info "Submodule ready: ${SRC_DIR}"
}

build_darknet() {
  require_cmd make
  require_cmd g++

  if [[ "${OPENCV_FLAG}" -eq 1 ]]; then
    if ! pkg-config --exists opencv4 2>/dev/null && ! pkg-config --exists opencv 2>/dev/null; then
      log_warn "OpenCV dev packages not found (opencv-devel / libopencv-dev)."
      log_warn "Building with OPENCV=0 (training works; install OpenCV for GUI/streaming)."
      OPENCV_FLAG=0
    fi
  fi

  log_info "Building darknet (GPU=${GPU_FLAG}, OPENCV=${OPENCV_FLAG})..."
  make -C "${SRC_DIR}" clean >/dev/null 2>&1 || true
  if [[ "${GPU_FLAG}" -eq 1 ]]; then
    make -C "${SRC_DIR}" -j"$(nproc 2>/dev/null || echo 2)" GPU=1 CUDNN=0 "OPENCV=${OPENCV_FLAG}"
  else
    make -C "${SRC_DIR}" -j"$(nproc 2>/dev/null || echo 2)" GPU=0 "OPENCV=${OPENCV_FLAG}"
  fi

  mkdir -p "${INSTALL_DIR}"
  install -m 755 "${SRC_DIR}/darknet" "${INSTALL_DIR}/darknet"
  log_info "Installed: ${INSTALL_DIR}/darknet"
  log_info "Optional: ./scripts/fetch_darknet_cfg.sh"
}

ensure_darknet_submodule

if [[ "${INIT_SUBMODULE_ONLY}" == "true" ]]; then
  exit 0
fi

build_darknet
