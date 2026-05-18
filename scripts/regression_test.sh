#!/usr/bin/env bash
# Regression smoke test: train representative YOLO checkpoints for one epoch on mini VOC.
#
# Usage:
#   ./scripts/regression_test.sh
#   ./scripts/regression_test.sh --device cpu
#   ./scripts/regression_test.sh --gpus 0,1,2,3,4   # train models in parallel (one job per GPU)
#   ./scripts/regression_test.sh --all-trainable   # all 68 checkpoints on mini VOC (slow)
#
# Default run trains 8 representative models (one per backend), NOT all 68.
# Use --all-trainable to train every checkpoint under models/ on mini VOC.
#   ./scripts/regression_test.sh --with-darknet    # also train yolov4-tiny via AlexeyAB darknet
#   ./scripts/regression_test.sh --models yolov5n,yolov8n
#   ./scripts/regression_test.sh --epochs 2
#   ./scripts/regression_test.sh --check-only      # verify prerequisites, do not train
#   ./scripts/regression_test.sh --skip-setup      # do not auto-install/sync/download
#
# By default the script auto-fixes common failures:
#   - pip install -r requirements.txt (torch, ultralytics, tensorboard, …)
#   - ./scripts/download_yolo_models.sh --sync-sources (yolov3/yolov5 models/)
#   - ./scripts/download_yolo_models.sh --regression-only (missing default weights)
#
# Environment:
#   PYTHON              Python executable (default: venv in repo, else $VIRTUAL_ENV, else python3)
#   REGRESSION_DEVICE   cuda | cpu | auto (default: auto); ignored when --gpus is set
#   REGRESSION_GPUS     Same as --gpus (comma-separated physical GPU ids)
#   REGRESSION_MODELS   Comma-separated model ids (default: all from registry)
#   REGRESSION_EPOCHS   Epoch count (default: 1)

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
YOLO_SH="${_SCRIPT_DIR}/yolo.sh"
PREPARE_PY="${_SCRIPT_DIR}/prepare_mini_voc_yolo.py"
DOWNLOAD_SH="${_SCRIPT_DIR}/download_yolo_models.sh"
FETCH_DARKNET_CFG_SH="${_SCRIPT_DIR}/fetch_darknet_cfg.sh"
INSTALL_DARKNET_SH="${_SCRIPT_DIR}/install_darknet.sh"
REQUIREMENTS="${REPO_ROOT}/requirements.txt"

EPOCHS="${REGRESSION_EPOCHS:-1}"
DEVICE="${REGRESSION_DEVICE:-auto}"
GPUS_LIST="${REGRESSION_GPUS:-}"
MODELS_FILTER="${REGRESSION_MODELS:-}"
SKIP_DATA_SETUP=false
SKIP_SETUP=false
FAIL_FAST=false
ALL_TRAINABLE=false
WITH_DARKNET=false
CHECK_ONLY=false
BATCH_SIZE=4

MINI_VOC_MARKER="${REPO_ROOT}/datasets/voc-mini/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt"
YOLO_DATA_MARKER="${REPO_ROOT}/datasets/voc-mini-yolo/.ready"
REGRESSION_CONFIG_DIR="${REPO_ROOT}/.regression_configs"
REGRESSION_LOG_DIR="${REPO_ROOT}/outputs/regression-test/logs"
REGRESSION_STATUS_DIR="${REPO_ROOT}/outputs/regression-test/status"
YOLOV3_VENDOR_MARKER="${REPO_ROOT}/yolo/yolov3/utils/dataloaders.py"
YOLOV5_VENDOR_MARKER="${REPO_ROOT}/yolo/yolov5/utils/dataloaders.py"
DARKNET_BIN="${REPO_ROOT}/tools/darknet/darknet"

usage() {
  sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
  --device)
    DEVICE="$2"
    shift 2
    ;;
  --gpus)
    GPUS_LIST="$2"
    shift 2
    ;;
  --epochs)
    EPOCHS="$2"
    shift 2
    ;;
  --batch-size)
    BATCH_SIZE="$2"
    shift 2
    ;;
  --models)
    MODELS_FILTER="$2"
    shift 2
    ;;
  --skip-data-setup)
    SKIP_DATA_SETUP=true
    shift
    ;;
  --skip-setup)
    SKIP_SETUP=true
    shift
    ;;
  --fail-fast)
    FAIL_FAST=true
    shift
    ;;
  --all-trainable)
    ALL_TRAINABLE=true
    shift
    ;;
  --with-darknet)
    WITH_DARKNET=true
    shift
    ;;
  --check-only)
    CHECK_ONLY=true
    shift
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    echo "[ERROR] Unknown option: $1" >&2
    usage
    exit 1
    ;;
  esac
done

# shellcheck source=lib/dataset_common.sh
source "${_SCRIPT_DIR}/lib/dataset_common.sh"

if [[ ! -f "${YOLO_SH}" ]]; then
  log_error "Not found: ${YOLO_SH}"
  exit 1
fi

resolve_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    return 0
  fi
  if [[ -x "${REPO_ROOT}/venv/bin/python" ]]; then
    PYTHON="${REPO_ROOT}/venv/bin/python"
  elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON="${VIRTUAL_ENV}/bin/python"
  else
    PYTHON="python3"
  fi
  export PYTHON
}

resolve_python
log_info "Python: ${PYTHON} ($("${PYTHON}" --version 2>&1))"

parse_gpu_list() {
  # Fill array variable name passed as $1 from comma-separated GPUS_LIST.
  local -n _out_arr="$1"
  _out_arr=()
  local item
  IFS=',' read -r -a _raw <<< "${GPUS_LIST}"
  for item in "${_raw[@]}"; do
    item="$(echo "${item}" | xargs)"
    [[ -n "${item}" ]] && _out_arr+=("${item}")
  done
}

count_visible_gpus() {
  "${PYTHON}" -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)"
}

train_one_model() {
  local model_id="$1"
  local train_device="$2"
  local log_file="${REGRESSION_LOG_DIR}/${model_id}.log"
  local status_file="${REGRESSION_STATUS_DIR}/${model_id}.exit"

  log_info "Training ${model_id} on device=${train_device} (log: ${log_file})"
  set +e
  "${YOLO_SH}" train \
    --model "${model_id}" \
    --epochs "${EPOCHS}" \
    --device "${train_device}" \
    --batch-size "${BATCH_SIZE}" >"${log_file}" 2>&1
  local status=$?
  set -e
  echo "${status}" >"${status_file}"
  return "${status}"
}

collect_results() {
  PASSED=()
  FAILED=()
  local model_id status
  for model_id in "${MODEL_IDS[@]}"; do
    status_file="${REGRESSION_STATUS_DIR}/${model_id}.exit"
    if [[ ! -f "${status_file}" ]]; then
      FAILED+=("${model_id}")
      log_error "FAIL: ${model_id} (no status file — job did not finish?)"
      continue
    fi
    status="$(<"${status_file}")"
    if [[ "${status}" -eq 0 ]]; then
      PASSED+=("${model_id}")
      log_info "PASS: ${model_id}"
    else
      FAILED+=("${model_id}")
      log_error "FAIL: ${model_id} (exit ${status}, see ${REGRESSION_LOG_DIR}/${model_id}.log)"
    fi
  done
}

run_training_sequential() {
  local model_id
  PASSED=()
  FAILED=()
  for model_id in "${MODEL_IDS[@]}"; do
    log_info "========== ${model_id} =========="
    set +e
    "${YOLO_SH}" train \
      --model "${model_id}" \
      --epochs "${EPOCHS}" \
      --device "${RUN_DEVICE}" \
      --batch-size "${BATCH_SIZE}"
    status=$?
    set -e
    if [[ ${status} -eq 0 ]]; then
      PASSED+=("${model_id}")
      log_info "PASS: ${model_id}"
    elif [[ ${status} -eq 130 ]] || [[ ${status} -eq 143 ]]; then
      log_error "Interrupted during ${model_id}"
      exit "${status}"
    else
      FAILED+=("${model_id}")
      log_error "FAIL: ${model_id} (exit ${status})"
      if [[ "${FAIL_FAST}" == "true" ]]; then
        break
      fi
    fi
  done
}

run_training_parallel_gpus() {
  local -a gpus=()
  parse_gpu_list gpus
  if [[ ${#gpus[@]} -eq 0 ]]; then
    log_error "--gpus requires at least one GPU id, e.g. --gpus 0,1,2,3,4"
    exit 1
  fi

  local visible
  visible="$(count_visible_gpus)"
  if [[ "${visible}" -eq 0 ]]; then
    log_error "CUDA not available; cannot use --gpus"
    exit 1
  fi

  local gpu max_gpu=-1
  for gpu in "${gpus[@]}"; do
    if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
      log_error "Invalid GPU id '${gpu}' (use comma-separated integers, e.g. 0,1,2,3,4)"
      exit 1
    fi
    if [[ "${gpu}" -gt "${max_gpu}" ]]; then
      max_gpu="${gpu}"
    fi
  done
  if [[ "${max_gpu}" -ge "${visible}" ]]; then
    log_error "Requested GPU ${max_gpu} but only ${visible} visible (CUDA_VISIBLE_DEVICES may be set)"
    exit 1
  fi

  mkdir -p "${REGRESSION_LOG_DIR}" "${REGRESSION_STATUS_DIR}"
  rm -f "${REGRESSION_STATUS_DIR}"/*.exit 2>/dev/null || true

  log_info "Parallel training: ${#MODEL_IDS[@]} model(s) across ${#gpus[@]} GPU(s): ${gpus[*]}"
  log_info "Each job uses CUDA_VISIBLE_DEVICES=<id> and --device 0 inside the trainer"

  local -a pids=()
  local -a pid_models=()
  local -a pid_gpus=()
  local gpu_idx=0 ngpus=${#gpus[@]}
  local model_id assigned_gpu pid i

  launch_job() {
    local mid="$1"
    local g="$2"
    (
      export CUDA_VISIBLE_DEVICES="${g}"
      train_one_model "${mid}" "0"
    ) &
    pids+=("$!")
    pid_models+=("${mid}")
    pid_gpus+=("${g}")
  }

  reap_finished_jobs() {
    local new_pids=() new_models=() new_gpus=()
    for i in "${!pids[@]}"; do
      if kill -0 "${pids[$i]}" 2>/dev/null; then
        new_pids+=("${pids[$i]}")
        new_models+=("${pid_models[$i]}")
        new_gpus+=("${pid_gpus[$i]}")
      else
        if wait "${pids[$i]}"; then
          :
        else
          if [[ "${FAIL_FAST}" == "true" ]]; then
            log_error "FAIL_FAST: stopping after ${pid_models[$i]} on GPU ${pid_gpus[$i]}"
            for pid in "${pids[@]}"; do
              kill "${pid}" 2>/dev/null || true
            done
            exit 1
          fi
        fi
      fi
    done
    pids=("${new_pids[@]}")
    pid_models=("${new_models[@]}")
    pid_gpus=("${new_gpus[@]}")
  }

  for model_id in "${MODEL_IDS[@]}"; do
    while [[ ${#pids[@]} -ge ${ngpus} ]]; do
      sleep 2
      reap_finished_jobs
    done
    assigned_gpu="${gpus[$((gpu_idx % ngpus))]}"
    gpu_idx=$((gpu_idx + 1))
    log_info "========== ${model_id} -> GPU ${assigned_gpu} =========="
    launch_job "${model_id}" "${assigned_gpu}"
  done

  while [[ ${#pids[@]} -gt 0 ]]; do
    sleep 2
    reap_finished_jobs
  done

  collect_results
}

resolve_device() {
  local requested="$1"
  case "${requested}" in
  auto)
    if "${PYTHON}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
      echo "0"
    else
      echo "cpu"
    fi
    ;;
  cuda)
    echo "0"
    ;;
  cpu)
    echo "cpu"
    ;;
  *)
    echo "${requested}"
    ;;
  esac
}

python_has_module() {
  local module="$1"
  "${PYTHON}" -c "import ${module}" 2>/dev/null
}

ensure_python_deps() {
  local -a missing=()
  python_has_module torch || missing+=("torch")
  python_has_module ultralytics || missing+=("ultralytics")
  python_has_module tensorboard || missing+=("tensorboard")
  python_has_module cv2 || missing+=("opencv-python")
  python_has_module pandas || missing+=("pandas")
  python_has_module yaml || missing+=("PyYAML")
  python_has_module tqdm || missing+=("tqdm")
  python_has_module scipy || missing+=("scipy")
  python_has_module thop || missing+=("thop")
  python_has_module addict || missing+=("addict")
  python_has_module pkg_resources || missing+=("setuptools")
  python_has_module seaborn || missing+=("seaborn")
  python_has_module IPython || missing+=("ipython")

  if [[ ${#missing[@]} -eq 0 ]]; then
    log_info "Python dependencies OK"
    return 0
  fi

  if [[ "${SKIP_SETUP}" == "true" ]]; then
    log_error "Missing Python modules: ${missing[*]}"
    log_error "Install with: ${PYTHON} -m pip install -r ${REQUIREMENTS}"
    return 1
  fi

  if [[ ! -f "${REQUIREMENTS}" ]]; then
    log_error "requirements.txt not found: ${REQUIREMENTS}"
    return 1
  fi

  log_info "Installing missing Python deps (${missing[*]}) from ${REQUIREMENTS}..."
  if ! "${PYTHON}" -m pip install -r "${REQUIREMENTS}"; then
    log_error "pip install failed"
    return 1
  fi

  local -a still_missing=()
  python_has_module torch || still_missing+=("torch")
  python_has_module ultralytics || still_missing+=("ultralytics")
  python_has_module tensorboard || still_missing+=("tensorboard")
  python_has_module cv2 || still_missing+=("opencv-python")
  python_has_module pandas || still_missing+=("pandas")
  python_has_module yaml || still_missing+=("PyYAML")
  python_has_module tqdm || still_missing+=("tqdm")
  python_has_module scipy || still_missing+=("scipy")
  python_has_module thop || still_missing+=("thop")
  python_has_module addict || still_missing+=("addict")
  python_has_module seaborn || still_missing+=("seaborn")
  python_has_module IPython || still_missing+=("ipython")
  if [[ ${#still_missing[@]} -gt 0 ]]; then
    log_error "Still missing after pip install: ${still_missing[*]}"
    return 1
  fi
  log_info "Python dependencies installed"
}

ensure_source_trees() {
  local need_sync=false
  [[ ! -f "${YOLOV3_VENDOR_MARKER}" ]] && need_sync=true
  [[ ! -f "${YOLOV5_VENDOR_MARKER}" ]] && need_sync=true

  if [[ "${need_sync}" == "false" ]]; then
    log_info "yolov3/yolov5 vendor trees present (models + utils)"
    return 0
  fi

  if [[ "${SKIP_SETUP}" == "true" ]]; then
    log_error "Incomplete vendored yolov3/yolov5 trees (need models/ + utils/ in sync):"
    [[ ! -f "${YOLOV3_VENDOR_MARKER}" ]] && log_error "  missing ${YOLOV3_VENDOR_MARKER}"
    [[ ! -f "${YOLOV5_VENDOR_MARKER}" ]] && log_error "  missing ${YOLOV5_VENDOR_MARKER}"
    log_error "Run: ./scripts/download_yolo_models.sh --sync-sources-only"
    return 1
  fi

  if [[ ! -x "${DOWNLOAD_SH}" ]]; then
    log_error "Not found: ${DOWNLOAD_SH}"
    return 1
  fi

  log_info "Syncing yolov3/yolov5 vendor trees (models + utils + hyps)..."
  PYTHON="${PYTHON}" "${DOWNLOAD_SH}" --sync-sources-only
}

ensure_regression_weights() {
  local -a model_ids=("$@")
  if [[ ${#model_ids[@]} -eq 0 ]]; then
    return 0
  fi

  local missing_list
  missing_list="$(
    REPO_ROOT="${REPO_ROOT}" MODEL_IDS="${model_ids[*]}" "${PYTHON}" - <<'PY'
import os
from pathlib import Path

yolo_root = Path(os.environ["REPO_ROOT"]) / "yolo"
import sys
sys.path.insert(0, str(yolo_root))
from registry import MODELS_DIR, TrainBackend, resolve_model

for model_id in os.environ.get("MODEL_IDS", "").split():
    if not model_id:
        continue
    spec = resolve_model(model_id)
    path = MODELS_DIR / spec.weights_file
    if not path.is_file():
        print(model_id)
PY
  )"

  if [[ -z "${missing_list}" ]]; then
    log_info "Regression weights present under ${REPO_ROOT}/models/"
    return 0
  fi

  if [[ "${SKIP_SETUP}" == "true" ]]; then
    log_error "Missing weights for: $(echo "${missing_list}" | tr '\n' ' ')"
    log_error "Run: ./scripts/download_yolo_models.sh --regression-only"
    return 1
  fi

  log_info "Downloading missing regression weights..."
  PYTHON="${PYTHON}" "${DOWNLOAD_SH}" --regression-only
}

ensure_darknet_prereqs() {
  if [[ "${WITH_DARKNET}" != "true" ]]; then
    return 0
  fi

  local cfg_dir="${REPO_ROOT}/yolo/darknet/cfg"
  if [[ ! -f "${cfg_dir}/yolov4-tiny.cfg" ]]; then
    if [[ "${SKIP_SETUP}" == "true" ]]; then
      log_error "Darknet cfg missing. Run: ./scripts/fetch_darknet_cfg.sh"
      return 1
    fi
    log_info "Fetching Darknet cfg files..."
    "${FETCH_DARKNET_CFG_SH}"
  fi

  if [[ -x "${DARKNET_BIN}" ]]; then
    log_info "Darknet binary: ${DARKNET_BIN}"
    return 0
  fi

  if [[ "${SKIP_SETUP}" == "true" ]]; then
    log_error "Darknet binary not found: ${DARKNET_BIN}"
    log_error "Build with: ./scripts/install_darknet.sh --cpu"
    return 1
  fi

  log_info "Building Darknet (CPU, OPENCV auto)..."
  "${INSTALL_DARKNET_SH}" --cpu
}

ensure_mini_voc() {
  if [[ -f "${MINI_VOC_MARKER}" ]]; then
    log_info "Mini VOC present: ${MINI_VOC_MARKER}"
    return 0
  fi
  if [[ "${SKIP_DATA_SETUP}" == "true" ]]; then
    log_error "Mini VOC missing and --skip-data-setup was set"
    exit 1
  fi
  local full_voc="${REPO_ROOT}/datasets/voc/VOCdevkit/VOC2007/ImageSets/Main"
  if [[ ! -d "${full_voc}" ]]; then
    log_error "Mini VOC not found. Create it with:"
    log_error "  ./scripts/download_voc.sh --2007-only"
    log_error "  ./scripts/create_mini_voc.sh"
    exit 1
  fi
  log_info "Building mini VOC (first-time setup)..."
  "${_SCRIPT_DIR}/create_mini_voc.sh"
}

ensure_yolo_dataset() {
  if [[ -f "${YOLO_DATA_MARKER}" && -f "${REGRESSION_CONFIG_DIR}/voc-mini.yaml" \
    && -f "${REGRESSION_CONFIG_DIR}/voc-mini-seg.yaml" \
    && -f "${REGRESSION_CONFIG_DIR}/voc-mini-pose.yaml" \
    && -f "${REGRESSION_CONFIG_DIR}/voc-mini-cls.yaml" ]]; then
    log_info "Mini VOC regression datasets ready: ${YOLO_DATA_MARKER}"
    return 0
  fi
  if [[ "${SKIP_DATA_SETUP}" == "true" ]]; then
    log_error "YOLO-format mini VOC missing and --skip-data-setup was set"
    exit 1
  fi
  ensure_mini_voc
  log_info "Preparing mini VOC (det/seg/pose/cls) and dataset YAMLs..."
  "${PYTHON}" "${PREPARE_PY}" --config-dir "${REGRESSION_CONFIG_DIR}"
}

list_models() {
  if [[ -n "${MODELS_FILTER}" ]]; then
    IFS=',' read -r -a _models <<< "${MODELS_FILTER}"
    for model_id in "${_models[@]}"; do
      model_id="$(echo "${model_id}" | xargs)"
      [[ -n "${model_id}" ]] && printf '%s\n' "${model_id}"
    done
    return 0
  fi
  REPO_ROOT="${REPO_ROOT}" ALL_TRAINABLE="${ALL_TRAINABLE}" WITH_DARKNET="${WITH_DARKNET}" "${PYTHON}" - <<'PY'
import os
import sys
from pathlib import Path

yolo_root = Path(os.environ["REPO_ROOT"]) / "yolo"
sys.path.insert(0, str(yolo_root))
from registry import list_all_regression_model_ids, list_regression_models

if os.environ.get("ALL_TRAINABLE", "false").lower() == "true":
    ids = list_all_regression_model_ids()
else:
    ids = list_regression_models()
    if os.environ.get("WITH_DARKNET", "false").lower() == "true":
        for extra in ("yolov4-tiny",):
            if extra not in ids:
                ids.append(extra)
for model_id in ids:
    print(model_id)
PY
}

preflight_models() {
  local -a model_ids=("$@")
  REPO_ROOT="${REPO_ROOT}" MODEL_IDS="${model_ids[*]}" "${PYTHON}" - <<'PY'
import os
import sys
from pathlib import Path

yolo_root = Path(os.environ["REPO_ROOT"]) / "yolo"
sys.path.insert(0, str(yolo_root))
from backends import preflight
from registry import resolve_model

ok = True
for model_id in os.environ.get("MODEL_IDS", "").split():
    if not model_id:
        continue
    try:
        preflight(resolve_model(model_id))
        print(f"[OK]   {model_id}")
    except Exception as exc:
        print(f"[FAIL] {model_id}: {exc}", file=sys.stderr)
        ok = False
if not ok:
    sys.exit(1)
PY
}

RUN_DEVICE="$(resolve_device "${DEVICE}")"

mapfile -t MODEL_IDS < <(list_models)
if [[ ${#MODEL_IDS[@]} -eq 0 ]]; then
  log_error "No models to run"
  exit 1
fi

if [[ "${ALL_TRAINABLE}" == "true" ]]; then
  log_info "Regression plan: ALL trainable checkpoints (${#MODEL_IDS[@]}) on mini VOC"
else
  log_info "Regression plan: default smoke set (${#MODEL_IDS[@]} models, one per backend) on mini VOC"
  log_info "  Use --all-trainable to train all checkpoints under models/ (68 with full weights)"
fi
if [[ -n "${GPUS_LIST}" ]]; then
  log_info "  epochs=${EPOCHS} gpus=${GPUS_LIST} batch=${BATCH_SIZE} (parallel)"
else
  log_info "  epochs=${EPOCHS} device=${RUN_DEVICE} batch=${BATCH_SIZE} (sequential)"
fi
log_info "  ${MODEL_IDS[*]}"

log_info "========== Prerequisites =========="
ensure_python_deps
ensure_source_trees
ensure_regression_weights "${MODEL_IDS[@]}"
ensure_darknet_prereqs
ensure_yolo_dataset
mkdir -p "${REGRESSION_CONFIG_DIR}"

if ! preflight_models "${MODEL_IDS[@]}"; then
  log_error "Per-model preflight failed (see messages above)"
  if [[ "${SKIP_SETUP}" == "true" ]]; then
    log_error "Re-run without --skip-setup to auto-fix, or run:"
    log_error "  ./scripts/download_yolo_models.sh --regression-only --sync-sources --install-deps"
  fi
  exit 1
fi

if [[ "${CHECK_ONLY}" == "true" ]]; then
  log_info "Prerequisites OK (--check-only, skipping training)"
  exit 0
fi

if [[ -z "${GPUS_LIST}" ]]; then
  if [[ "${RUN_DEVICE}" != "cpu" ]]; then
    if ! "${PYTHON}" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
      log_warn "CUDA not available; falling back to cpu"
      RUN_DEVICE="cpu"
    fi
  fi
else
  if [[ "${RUN_DEVICE}" != "cpu" && "${DEVICE}" != "auto" ]]; then
    log_warn "--device is ignored when --gpus is set (each job uses its own GPU)"
  fi
fi

log_info "========== Training =========="
if [[ -n "${GPUS_LIST}" ]]; then
  run_training_parallel_gpus
else
  run_training_sequential
fi

echo ""
log_info "Regression summary: ${#PASSED[@]} passed, ${#FAILED[@]} failed (${#MODEL_IDS[@]} total)"
if [[ ${#PASSED[@]} -gt 0 ]]; then
  log_info "  passed: ${PASSED[*]}"
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
  log_error "  failed: ${FAILED[*]}"
  exit 1
fi
