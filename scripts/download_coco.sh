#!/usr/bin/env bash
# Download MS COCO 2017 (images + YOLO-format labels) into datasets/coco/.
#
# Layout (YOLO-compatible):
#   datasets/coco/images/train2017/
#   datasets/coco/images/val2017/
#   datasets/coco/labels/...
#
# Usage:
#   ./scripts/download_coco.sh
#   ./scripts/download_coco.sh --val
#   ./scripts/download_coco.sh --train --val --segments
#   ./scripts/download_coco.sh --train --val --test
#
# Approximate download sizes (defaults: train + val, no test):
#   labels zip:  ~46 MB
#   val2017:     ~1 GB
#   train2017:   ~19 GB

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/dataset_common.sh
source "${SCRIPT_DIR}/lib/dataset_common.sh"

require_cmd curl
require_cmd unzip

train=false
val=false
test=false
segments=false

if [[ "$#" -eq 0 ]]; then
  train=true
  val=true
else
  for opt in "$@"; do
    case "${opt}" in
    --train) train=true ;;
    --val) val=true ;;
    --test) test=true ;;
    --segments) segments=true ;;
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
fi

COCO_IMAGES_URL="http://images.cocodataset.org/zips/"
LABELS_URL="https://github.com/ultralytics/yolov5/releases/download/v1.0/"

ensure_datasets_dir

if [[ "${segments}" == "true" ]]; then
  labels_archive="coco2017labels-segments.zip"
else
  labels_archive="coco2017labels.zip"
fi

labels_path="${DATASETS_DIR}/${labels_archive}"
download_file "${LABELS_URL}${labels_archive}" "${labels_path}"
extract_zip "${labels_path}" "${DATASETS_DIR}"

images_dir="${DATASETS_DIR}/coco/images"
mkdir -p "${images_dir}"

download_coco_images() {
  local split="$1"
  local archive="${split}.zip"
  local dest="${images_dir}/${archive}"
  download_file "${COCO_IMAGES_URL}${archive}" "${dest}"
  log_info "Extracting ${archive} ..."
  unzip -q "${dest}" -d "${images_dir}"
  rm -f "${dest}"
}

if [[ "${train}" == "true" ]]; then
  download_coco_images "train2017"
fi

if [[ "${val}" == "true" ]]; then
  download_coco_images "val2017"
fi

if [[ "${test}" == "true" ]]; then
  download_coco_images "test2017"
fi

log_info "COCO download complete: ${DATASETS_DIR}/coco"
