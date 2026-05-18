#!/usr/bin/env bash
# Build a mini PASCAL VOC 2007 subset (~10 images per class) for fast dry-runs.
#
# Output layout (VOC-compatible):
#   datasets/voc-mini/VOCdevkit/VOC2007/
#     JPEGImages/  Annotations/  ImageSets/Main/{trainval,test}.txt
#
# Usage:
#   ./scripts/create_mini_voc.sh
#   ./scripts/create_mini_voc.sh --per-class 10 --year 2007
#   ./scripts/create_mini_voc.sh --output datasets/voc-mini
#
# Then run YOLO regression (1 epoch per family):
#   ./scripts/regression_test.sh
#
# Requires full VOC at datasets/voc/VOCdevkit (run ./scripts/download_voc.sh first).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/dataset_common.sh
source "${SCRIPT_DIR}/lib/dataset_common.sh"

VOC_YEAR="2007"
PER_CLASS=10
SAMPLE_SPLIT="trainval"
TEST_FRACTION="0.2"
SEED=42
OUTPUT_ROOT="${DATASETS_DIR}/voc-mini"
SOURCE_VOC_ROOT="${DATASETS_DIR}/voc/VOCdevkit"
LINK_MODE=false

VOC_CLASSES=(
  aeroplane bicycle bird boat bottle bus car cat chair cow
  diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor
)

usage() {
  grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -22
}

while [[ $# -gt 0 ]]; do
  case "$1" in
  --year)
    VOC_YEAR="$2"
    shift 2
    ;;
  --per-class)
    PER_CLASS="$2"
    shift 2
    ;;
  --sample-split)
    SAMPLE_SPLIT="$2"
    shift 2
    ;;
  --test-fraction)
    TEST_FRACTION="$2"
    shift 2
    ;;
  --seed)
    SEED="$2"
    shift 2
    ;;
  --output)
    OUTPUT_ROOT="$2"
    shift 2
    ;;
  --source)
    SOURCE_VOC_ROOT="$2"
    shift 2
    ;;
  --link)
    LINK_MODE=true
    shift
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    log_error "Unknown option: $1"
    usage
    exit 1
    ;;
  esac
done

SRC="${SOURCE_VOC_ROOT}/VOC${VOC_YEAR}"
DEST="${OUTPUT_ROOT}/VOCdevkit/VOC${VOC_YEAR}"

if [[ ! -d "${SRC}/ImageSets/Main" ]]; then
  log_error "Source VOC not found: ${SRC}"
  log_error "Run: ./scripts/download_voc.sh --2007-only"
  exit 1
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT
SELECTED="${WORKDIR}/selected_ids.txt"
UNIQUE="${WORKDIR}/unique_ids.txt"
: > "${SELECTED}"

log_info "Sampling up to ${PER_CLASS} positive images per class from ${SRC} (${SAMPLE_SPLIT})"

for class_name in "${VOC_CLASSES[@]}"; do
  class_file="${SRC}/ImageSets/Main/${class_name}_${SAMPLE_SPLIT}.txt"
  if [[ ! -f "${class_file}" ]]; then
    log_error "Missing class split file: ${class_file}"
    exit 1
  fi
  count=0
  while read -r image_id label; do
    [[ "${label}" == "1" ]] || continue
    echo "${image_id}" >> "${SELECTED}"
    count=$((count + 1))
    if [[ "${count}" -ge "${PER_CLASS}" ]]; then
      break
    fi
  done < "${class_file}"
done

sort -u "${SELECTED}" > "${UNIQUE}"
total="$(wc -l < "${UNIQUE}")"
if [[ "${total}" -eq 0 ]]; then
  log_error "No images selected; check source dataset and --sample-split"
  exit 1
fi

log_info "Selected ${total} unique images (union across classes)"

mkdir -p "${DEST}/JPEGImages" "${DEST}/Annotations" "${DEST}/ImageSets/Main"
if [[ -d "${SRC}/SegmentationObject" ]]; then
  mkdir -p "${DEST}/SegmentationObject"
fi

copy_file() {
  local src_file="$1"
  local dest_file="$2"
  if [[ "${LINK_MODE}" == "true" ]]; then
    ln -sf "$(realpath "${src_file}")" "${dest_file}"
  else
    cp -f "${src_file}" "${dest_file}"
  fi
}

while read -r image_id; do
  [[ -n "${image_id}" ]] || continue
  jpg="${SRC}/JPEGImages/${image_id}.jpg"
  xml="${SRC}/Annotations/${image_id}.xml"
  if [[ ! -f "${jpg}" ]] || [[ ! -f "${xml}" ]]; then
    log_warn "Skipping missing pair for id=${image_id}"
    continue
  fi
  copy_file "${jpg}" "${DEST}/JPEGImages/${image_id}.jpg"
  copy_file "${xml}" "${DEST}/Annotations/${image_id}.xml"
  seg="${SRC}/SegmentationObject/${image_id}.png"
  if [[ -f "${seg}" ]]; then
    copy_file "${seg}" "${DEST}/SegmentationObject/${image_id}.png"
  fi
done < "${UNIQUE}"

# Reproducible train/test split of the mini set
shuf --random-source=<(yes "${SEED}") "${UNIQUE}" > "${WORKDIR}/shuffled.txt"
test_count="$(python3 - <<PY
import math
total = int("${total}")
frac = float("${TEST_FRACTION}")
print(max(1, min(total - 1, int(round(total * frac)))))
PY
)"

head -n "${test_count}" "${WORKDIR}/shuffled.txt" | sort > "${DEST}/ImageSets/Main/test.txt"
tail -n "+$((test_count + 1))" "${WORKDIR}/shuffled.txt" | sort > "${DEST}/ImageSets/Main/trainval.txt"

# VOC also expects train.txt / val.txt for some tools; mirror trainval for train, test for val
cp "${DEST}/ImageSets/Main/trainval.txt" "${DEST}/ImageSets/Main/train.txt"
cp "${DEST}/ImageSets/Main/test.txt" "${DEST}/ImageSets/Main/val.txt"

train_n="$(wc -l < "${DEST}/ImageSets/Main/trainval.txt")"
test_n="$(wc -l < "${DEST}/ImageSets/Main/test.txt")"

log_info "Wrote mini VOC to ${DEST}"
log_info "  trainval: ${train_n} images"
log_info "  test:     ${test_n} images"
log_info "Dry-run example:"
log_info "  ./scripts/od.sh all"
