#!/usr/bin/env bash
# Download PASCAL VOC detection datasets into datasets/voc/.
#
# Layout after extraction:
#   datasets/voc/VOCdevkit/VOC2007/
#   datasets/voc/VOCdevkit/VOC2012/
#
# Usage:
#   ./scripts/download_voc.sh              # VOC 2007 + 2012 (default)
#   ./scripts/download_voc.sh --2007-only
#   ./scripts/download_voc.sh --2012-only
#   DATASETS_DIR=/path/to/data ./scripts/download_voc.sh
#
# Approximate download sizes:
#   VOC 2007 trainval + test: ~0.5 GB
#   VOC 2012 trainval:        ~1.9 GB

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/dataset_common.sh
source "${SCRIPT_DIR}/lib/dataset_common.sh"

require_cmd curl
require_cmd tar

download_2007=true
download_2012=true

for opt in "$@"; do
  case "${opt}" in
  --2007-only)
    download_2007=true
    download_2012=false
    ;;
  --2012-only)
    download_2007=false
    download_2012=true
    ;;
  --help | -h)
    grep '^#' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  *)
    log_error "Unknown option: ${opt}"
    exit 1
    ;;
  esac
done

VOC_BASE="https://thor.robots.ox.ac.uk/pascal/VOC"
VOC_DIR="${DATASETS_DIR}/voc"
ensure_datasets_dir
mkdir -p "${VOC_DIR}"
cd "${VOC_DIR}"

if [[ "${download_2007}" == "true" ]]; then
  log_info "Fetching PASCAL VOC 2007 (trainval + test) ..."
  archive="VOCtrainval_06-Nov-2007.tar"
  download_file "${VOC_BASE}/voc2007/${archive}" "${VOC_DIR}/${archive}"
  extract_tar "${VOC_DIR}/${archive}" "${VOC_DIR}"

  archive="VOCtest_06-Nov-2007.tar"
  download_file "${VOC_BASE}/voc2007/${archive}" "${VOC_DIR}/${archive}"
  extract_tar "${VOC_DIR}/${archive}" "${VOC_DIR}"
fi

if [[ "${download_2012}" == "true" ]]; then
  log_info "Fetching PASCAL VOC 2012 trainval ..."
  archive="VOCtrainval_11-May-2012.tar"
  download_file "${VOC_BASE}/voc2012/${archive}" "${VOC_DIR}/${archive}"
  extract_tar "${VOC_DIR}/${archive}" "${VOC_DIR}"
fi

log_info "VOC download complete: ${VOC_DIR}/VOCdevkit"
