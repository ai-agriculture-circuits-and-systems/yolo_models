#!/usr/bin/env bash
# Build a mini PASCAL VOC subset for fast YOLO regression (det / seg / pose / cls).
#
# Output layout (VOC-compatible):
#   datasets/voc-mini/VOCdevkit/VOC2007/
#     JPEGImages/  Annotations/  ImageSets/Main/{trainval,test}.txt
#     SegmentationObject/   # copied when present in source VOC
#
# Usage:
#   ./scripts/create_mini_voc.sh --regression
#   ./scripts/create_mini_voc.sh --per-class 10 --year 2007
#   ./scripts/create_mini_voc.sh --output datasets/voc-mini --link
#
# Regression profile (--regression):
#   - ~5 images per class (up to ~100-150 unique images)
#   - Prefer images with SegmentationObject masks (VOC 2007 trainval)
#   - Extra person images for pose smoke tests
#   - Writes datasets/voc-mini/.regression-manifest
#
# Then convert to YOLO layouts:
#   ./scripts/prepare_mini_voc_yolo.py --force
#   ./scripts/regression_test.sh --all-trainable --gpus 0,1,2,3,4,5
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
REGRESSION_MODE=false
MIN_TOTAL=0
REQUIRE_SEGMENTATION=false
EXTRA_PERSON=0
PREFER_SEGMENTATION=true

VOC_CLASSES=(
  aeroplane bicycle bird boat bottle bus car cat chair cow
  diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor
)

usage() {
  grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -35
}

while [[ $# -gt 0 ]]; do
  case "$1" in
  --regression)
    REGRESSION_MODE=true
    PER_CLASS=5
    MIN_TOTAL=60
    TEST_FRACTION=0.2
    REQUIRE_SEGMENTATION=true
    EXTRA_PERSON=20
    PREFER_SEGMENTATION=true
    shift
    ;;
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
  --min-total)
    MIN_TOTAL="$2"
    shift 2
    ;;
  --require-segmentation)
    REQUIRE_SEGMENTATION=true
    shift
    ;;
  --extra-person)
    EXTRA_PERSON="$2"
    shift 2
    ;;
  --no-prefer-segmentation)
    PREFER_SEGMENTATION=false
    shift
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
MANIFEST="${OUTPUT_ROOT}/.regression-manifest"

if [[ ! -d "${SRC}/ImageSets/Main" ]]; then
  log_error "Source VOC not found: ${SRC}"
  log_error "Run: ./scripts/download_voc.sh --2007-only"
  exit 1
fi

HAS_SEG_DIR=false
if [[ -d "${SRC}/SegmentationObject" ]]; then
  HAS_SEG_DIR=true
elif [[ -d "${SOURCE_VOC_ROOT}/VOC2012/SegmentationObject" ]]; then
  log_warn "VOC${VOC_YEAR} has no SegmentationObject/; seg regression will use bbox-derived masks"
  log_warn "For real instance masks use VOC 2007 trainval source (download_voc.sh --2007-only)"
else
  log_warn "No SegmentationObject/ in source VOC; segmentation smoke tests use bbox quads only"
fi

if [[ "${REQUIRE_SEGMENTATION}" == "true" && "${HAS_SEG_DIR}" != "true" ]]; then
  log_error "--require-segmentation set but ${SRC}/SegmentationObject is missing"
  log_error "Use full PASCAL VOC 2007 trainval (includes segmentation masks)."
  exit 1
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT
SELECTED="${WORKDIR}/selected_ids.txt"
UNIQUE="${WORKDIR}/unique_ids.txt"
WITH_SEG="${WORKDIR}/with_seg.txt"
: > "${SELECTED}"

image_has_segmentation() {
  local image_id="$1"
  [[ -f "${SRC}/SegmentationObject/${image_id}.png" ]]
}

pick_from_class_file() {
  local class_file="$1"
  local want="$2"
  local picked="${WORKDIR}/picked_${RANDOM}.txt"
  : > "${picked}"
  local count=0
  local image_id label

  if [[ "${PREFER_SEGMENTATION}" == "true" && "${HAS_SEG_DIR}" == "true" ]]; then
    while read -r image_id label; do
      [[ "${label}" == "1" ]] || continue
      image_has_segmentation "${image_id}" || continue
      echo "${image_id}" >> "${picked}"
      count=$((count + 1))
      if [[ "${count}" -ge "${want}" ]]; then
        break
      fi
    done < "${class_file}"
  fi

  if [[ "${count}" -lt "${want}" ]]; then
    while read -r image_id label; do
      [[ "${label}" == "1" ]] || continue
      grep -qxF "${image_id}" "${picked}" 2>/dev/null && continue
      echo "${image_id}" >> "${picked}"
      count=$((count + 1))
      if [[ "${count}" -ge "${want}" ]]; then
        break
      fi
    done < "${class_file}"
  fi

  cat "${picked}" >> "${SELECTED}"
  rm -f "${picked}"
}

log_info "Sampling up to ${PER_CLASS} positive images per class from ${SRC} (${SAMPLE_SPLIT})"
if [[ "${REGRESSION_MODE}" == "true" ]]; then
  log_info "Regression profile: min_total=${MIN_TOTAL} extra_person=${EXTRA_PERSON} require_seg=${REQUIRE_SEGMENTATION}"
fi

for class_name in "${VOC_CLASSES[@]}"; do
  class_file="${SRC}/ImageSets/Main/${class_name}_${SAMPLE_SPLIT}.txt"
  if [[ ! -f "${class_file}" ]]; then
    log_error "Missing class split file: ${class_file}"
    exit 1
  fi
  pick_from_class_file "${class_file}" "${PER_CLASS}"
done

# Extra person images (pose trainers need several person instances).
if [[ "${EXTRA_PERSON}" -gt 0 ]]; then
  person_file="${SRC}/ImageSets/Main/person_${SAMPLE_SPLIT}.txt"
  if [[ -f "${person_file}" ]]; then
    log_info "Adding up to ${EXTRA_PERSON} extra person images"
    extra_picked=0
    while read -r image_id label; do
      [[ "${label}" == "1" ]] || continue
      grep -qxF "${image_id}" "${SELECTED}" 2>/dev/null && continue
      echo "${image_id}" >> "${SELECTED}"
      extra_picked=$((extra_picked + 1))
      if [[ "${extra_picked}" -ge "${EXTRA_PERSON}" ]]; then
        break
      fi
    done < "${person_file}"
  fi
fi

sort -u "${SELECTED}" > "${UNIQUE}"
total="$(wc -l < "${UNIQUE}")"
if [[ "${total}" -eq 0 ]]; then
  log_error "No images selected; check source dataset and --sample-split"
  exit 1
fi

if [[ "${MIN_TOTAL}" -gt 0 && "${total}" -lt "${MIN_TOTAL}" ]]; then
  log_error "Only ${total} images selected (need at least ${MIN_TOTAL})."
  log_error "Increase --per-class or ensure full VOC is installed under ${SOURCE_VOC_ROOT}"
  exit 1
fi

seg_count=0
if [[ "${HAS_SEG_DIR}" == "true" ]]; then
  while read -r image_id; do
    image_has_segmentation "${image_id}" && seg_count=$((seg_count + 1))
  done < "${UNIQUE}"
fi

log_info "Selected ${total} unique images (union across classes; ${seg_count} with SegmentationObject)"

mkdir -p "${DEST}/JPEGImages" "${DEST}/Annotations" "${DEST}/ImageSets/Main"
if [[ "${HAS_SEG_DIR}" == "true" ]]; then
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

copied_seg=0
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
    copied_seg=$((copied_seg + 1))
  fi
done < "${UNIQUE}"

if [[ "${REQUIRE_SEGMENTATION}" == "true" && "${copied_seg}" -eq 0 ]]; then
  log_error "No SegmentationObject masks copied; cannot satisfy --require-segmentation"
  exit 1
fi

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

cp "${DEST}/ImageSets/Main/trainval.txt" "${DEST}/ImageSets/Main/train.txt"
cp "${DEST}/ImageSets/Main/test.txt" "${DEST}/ImageSets/Main/val.txt"

train_n="$(wc -l < "${DEST}/ImageSets/Main/trainval.txt")"
test_n="$(wc -l < "${DEST}/ImageSets/Main/test.txt")"

cat > "${MANIFEST}" <<EOF
# Auto-generated by create_mini_voc.sh — do not edit
created=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
regression_mode=${REGRESSION_MODE}
per_class=${PER_CLASS}
total_images=${total}
train_images=${train_n}
test_images=${test_n}
segmentation_masks=${copied_seg}
source=${SRC}
EOF

log_info "Wrote mini VOC to ${DEST}"
log_info "  trainval: ${train_n} images"
log_info "  test:     ${test_n} images"
log_info "  SegmentationObject masks: ${copied_seg}/${total}"
log_info "  manifest: ${MANIFEST}"
log_info "Next:"
log_info "  ./scripts/prepare_mini_voc_yolo.py --force"
log_info "  ./scripts/regression_test.sh --all-trainable --gpus 0,1,2,3,4,5"
