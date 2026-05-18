#!/usr/bin/env bash
# Download ImageNet ILSVRC 2012 (or small Ultralytics subsets) into datasets/.
#
# Layout (full ILSVRC2012):
#   datasets/imagenet/train/<synset>/...
#   datasets/imagenet/val/<synset>/...
#
# Small subsets (no registration, good for smoke tests):
#   datasets/imagenet10/
#   datasets/imagenet100/
#   datasets/imagenet1000/
#
# Usage:
#   ./scripts/download_imagenet.sh --val
#   ./scripts/download_imagenet.sh --train --val
#   ./scripts/download_imagenet.sh --subset 100
#
# Approximate download sizes:
#   val only (full):     ~6.3 GB
#   train + val (full):  ~145 GB
#   imagenet100 zip:     ~several GB (see Ultralytics release)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/dataset_common.sh
source "${SCRIPT_DIR}/lib/dataset_common.sh"

require_cmd curl
require_cmd tar

train=false
val=false
subset="full"

if [[ "$#" -eq 0 ]]; then
  train=true
  val=true
fi

while [[ "$#" -gt 0 ]]; do
  case "$1" in
  --train)
    train=true
    shift
    ;;
  --val)
    val=true
    shift
    ;;
  --subset)
    if [[ "$#" -lt 2 ]]; then
      log_error "--subset requires a value: 10, 100, 1000, or full"
      exit 1
    fi
    subset="$2"
    shift 2
    ;;
  --subset=*)
    subset="${1#--subset=}"
    shift
    ;;
  --help | -h)
    grep '^#' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  *)
    log_error "Unknown option: $1"
    exit 1
    ;;
  esac
done

IMAGENET_URL="https://image-net.org/data/ILSVRC/2012"
ULTRALYTICS_URL="https://github.com/ultralytics/yolov5/releases/download/v1.0"
VALPREP_URL="https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh"

ensure_datasets_dir

download_imagenet_subset() {
  local name="$1"
  local dest="${DATASETS_DIR}/imagenet${name}"
  local archive="imagenet${name}.zip"
  mkdir -p "${dest}"
  download_file "${ULTRALYTICS_URL}/${archive}" "${dest}/${archive}"
  extract_zip "${dest}/${archive}" "${dest}"
  log_info "ImageNet subset ready: ${dest}"
}

if [[ "${subset}" != "full" ]]; then
  case "${subset}" in
  10 | 100 | 1000)
    download_imagenet_subset "${subset}"
    exit 0
    ;;
  *)
    log_error "Unsupported --subset value: ${subset} (use 10, 100, 1000, or full)"
    exit 1
    ;;
  esac
fi

IMAGENET_DIR="${DATASETS_DIR}/imagenet"
mkdir -p "${IMAGENET_DIR}"

if [[ "${train}" != "true" && "${val}" != "true" ]]; then
  log_error "Specify --train and/or --val for full ImageNet, or use --subset 10|100|1000."
  exit 1
fi

if [[ "${train}" == "true" ]]; then
  log_warn "Full ImageNet train is ~138 GB and takes a long time to extract."
  train_dir="${IMAGENET_DIR}/train"
  mkdir -p "${train_dir}"
  archive="${train_dir}/ILSVRC2012_img_train.tar"
  download_file "${IMAGENET_URL}/ILSVRC2012_img_train.tar" "${archive}"
  log_info "Extracting train tarball (nested archives) ..."
  (
    cd "${train_dir}"
    tar -xf ILSVRC2012_img_train.tar
    rm -f ILSVRC2012_img_train.tar
    find . -maxdepth 1 -name '*.tar' | while read -r nested; do
      class_dir="${nested%.tar}"
      mkdir -p "${class_dir}"
      tar -xf "${nested}" -C "${class_dir}"
      rm -f "${nested}"
    done
  )
fi

if [[ "${val}" == "true" ]]; then
  val_dir="${IMAGENET_DIR}/val"
  mkdir -p "${val_dir}"
  archive="${val_dir}/ILSVRC2012_img_val.tar"
  download_file "${IMAGENET_URL}/ILSVRC2012_img_val.tar" "${archive}"
  log_info "Extracting val tarball ..."
  (
    cd "${val_dir}"
    tar -xf ILSVRC2012_img_val.tar
    rm -f ILSVRC2012_img_val.tar
    log_info "Organizing val images into class subfolders (valprep.sh) ..."
    curl -fsSL "${VALPREP_URL}" -o valprep.sh
    bash valprep.sh
    rm -f valprep.sh
  )
fi

log_info "ImageNet download complete: ${IMAGENET_DIR}"
