#!/usr/bin/env bash
# Download YOLO pretrained weights (and optionally restore incomplete vendored source trees).
#
# Weights are parsed from README.md (wget -P models ...), with fixes in
# scripts/yolo_model_url_overrides.tsv for links that have moved (HTTP 404).
#
# Usage:
#   ./scripts/download_yolo_models.sh
#   ./scripts/download_yolo_models.sh --regression-only
#   ./scripts/download_yolo_models.sh --retry-failed   # only overrides for known 404s
#
# Existing files in models/ are skipped by default. Use --force to re-download.
# Tiny/corrupt files from old 404 responses are re-fetched automatically.
#   ./scripts/download_yolo_models.sh --filter yolov8
#   ./scripts/download_yolo_models.sh --sync-sources
#   ./scripts/download_yolo_models.sh --sync-sources-only   # no weight downloads
#   ./scripts/download_yolo_models.sh --install-deps
#   ./scripts/download_yolo_models.sh --dry-run --list
#
# Layout:
#   models/yolov5n.pt, models/yolov8n.pt, ...
#   yolo/YOLOv6/weights/yolov6n.pt   (linked/copied when --regression-only or --link-yolov6)

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
# shellcheck source=lib/dataset_common.sh
source "${_SCRIPT_DIR}/lib/dataset_common.sh"

README_FILE="${REPO_ROOT}/README.md"
OVERRIDES_FILE="${_SCRIPT_DIR}/yolo_model_url_overrides.tsv"
UNAVAILABLE_FILE="${_SCRIPT_DIR}/yolo_model_unavailable.txt"
MODELS_DIR="${REPO_ROOT}/models"
YOLO_ROOT="${REPO_ROOT}/yolo"
YOLOV6_WEIGHTS_DIR="${YOLO_ROOT}/YOLOv6/weights"

declare -A URL_OVERRIDE_PRIMARY=()
declare -A URL_OVERRIDE_FALLBACK=()
declare -A UNAVAILABLE_NAMES=()

REGRESSION_ONLY=false
RETRY_FAILED=false
SYNC_SOURCES=false
SYNC_SOURCES_ONLY=false
INSTALL_DEPS=false
LINK_YOLOV6=true
DRY_RUN=false
FORCE=false
LIST_ONLY=false
FILTER=""

# Files smaller than this are treated as failed/partial downloads (e.g. GitHub 404 HTML).
MIN_WEIGHT_BYTES=10240

# Minimal weights used by scripts/regression_test.sh (yolo/registry.py).
REGRESSION_WEIGHTS=(
  "https://github.com/ultralytics/yolov3/releases/download/v9.0/yolov3-tiny.pt"
  "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt"
  "https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6n.pt"
  "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
)

usage() {
  grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -28
}

while [[ $# -gt 0 ]]; do
  case "$1" in
  --regression-only)
    REGRESSION_ONLY=true
    shift
    ;;
  --retry-failed)
    RETRY_FAILED=true
    shift
    ;;
  --sync-sources)
    SYNC_SOURCES=true
    shift
    ;;
  --sync-sources-only)
    SYNC_SOURCES=true
    SYNC_SOURCES_ONLY=true
    shift
    ;;
  --install-deps)
    INSTALL_DEPS=true
    shift
    ;;
  --no-link-yolov6)
    LINK_YOLOV6=false
    shift
    ;;
  --output-dir)
    MODELS_DIR="$2"
    shift 2
    ;;
  --filter)
    FILTER="$2"
    shift 2
    ;;
  --force | -f)
    FORCE=true
    shift
    ;;
  --dry-run | -n)
    DRY_RUN=true
    shift
    ;;
  --list | -l)
    LIST_ONLY=true
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

url_basename() {
  basename "${1%%\?*}"
}

matches_filter() {
  local url="$1"
  local filename
  filename="$(url_basename "${url}")"
  if [[ -z "${FILTER}" ]]; then
    return 0
  fi
  [[ "${url}" == *"${FILTER}"* || "${filename}" == *"${FILTER}"* ]]
}

strip_crlf() {
  printf '%s' "$1" | tr -d '\r'
}

weight_file_ok() {
  local path="$1"
  [[ -f "${path}" ]] || return 1
  local size
  size="$(wc -c < "${path}")"
  [[ "${size}" -ge "${MIN_WEIGHT_BYTES}" ]]
}

should_skip_download() {
  local dest="$1"
  [[ "${FORCE}" == "true" ]] && return 1
  weight_file_ok "${dest}"
}

load_url_overrides() {
  if [[ ! -f "${OVERRIDES_FILE}" ]]; then
    return 0
  fi
  while IFS=$'\t' read -r name primary fallback _rest || [[ -n "${name:-}" ]]; do
    name="$(strip_crlf "${name:-}")"
    primary="$(strip_crlf "${primary:-}")"
    fallback="$(strip_crlf "${fallback:-}")"
    [[ "${name}" =~ ^# ]] && continue
    [[ -z "${name}" ]] && continue
    URL_OVERRIDE_PRIMARY["${name}"]="${primary}"
    if [[ -n "${fallback}" ]]; then
      URL_OVERRIDE_FALLBACK["${name}"]="${fallback}"
    fi
  done < "${OVERRIDES_FILE}"
}

load_unavailable_names() {
  if [[ ! -f "${UNAVAILABLE_FILE}" ]]; then
    return 0
  fi
  while read -r name; do
    name="$(strip_crlf "${name}")"
    [[ "${name}" =~ ^# ]] && continue
    [[ -z "${name}" ]] && continue
    UNAVAILABLE_NAMES["${name}"]=1
  done < "${UNAVAILABLE_FILE}"
}

is_unavailable() {
  [[ -n "${UNAVAILABLE_NAMES[$1]:-}" ]]
}

resolve_urls_for_file() {
  local filename="$1"
  local readme_url="$2"
  local -n _out="$3"
  _out=()
  if [[ -n "${URL_OVERRIDE_PRIMARY[${filename}]:-}" ]]; then
    _out+=("${URL_OVERRIDE_PRIMARY[${filename}]}")
    if [[ -n "${URL_OVERRIDE_FALLBACK[${filename}]:-}" ]]; then
      _out+=("${URL_OVERRIDE_FALLBACK[${filename}]}")
    fi
    return 0
  fi
  _out+=("${readme_url}")
}

collect_readme_urls() {
  if [[ ! -f "${README_FILE}" ]]; then
    log_error "README not found: ${README_FILE}"
    exit 1
  fi
  mapfile -t MODEL_URLS < <(
    grep -oE 'wget -P models https?://[^[:space:]]+' "${README_FILE}" \
      | sed -E 's/^wget -P models //' \
      | sort -u
  )
  if [[ ${#MODEL_URLS[@]} -eq 0 ]]; then
    log_error "No wget URLs found in ${README_FILE}"
    exit 1
  fi
}

download_one() {
  local url="$1"
  local dest="$2"
  if [[ "${DRY_RUN}" == "true" ]]; then
    log_info "[dry-run] ${url} -> ${dest}"
    return 0
  fi
  if should_skip_download "${dest}"; then
    return 0
  fi
  if [[ -f "${dest}" ]]; then
    log_warn "Re-downloading corrupt or tiny file: ${dest}"
    rm -f "${dest}"
  fi
  if download_file "${url}" "${dest}"; then
    return 0
  fi
  return 1
}

download_with_fallbacks() {
  local dest="$1"
  shift
  local -a try_urls=("$@")
  local url
  for url in "${try_urls[@]}"; do
    if download_one "${url}" "${dest}"; then
      return 0
    fi
    log_warn "Download failed, trying next mirror (if any): ${url}"
  done
  return 1
}

collect_retry_failed_files() {
  mapfile -t RETRY_FILES < <(
    while IFS=$'\t' read -r name _rest || [[ -n "${name:-}" ]]; do
      [[ "${name:-}" =~ ^# ]] && continue
      [[ -z "${name:-}" ]] && continue
      echo "${name}"
    done < "${OVERRIDES_FILE}"
  )
}

link_yolov6_weight() {
  local name="$1"
  local src="${MODELS_DIR}/${name}"
  local dest="${YOLOV6_WEIGHTS_DIR}/${name}"
  if ! weight_file_ok "${src}"; then
    return 0
  fi
  mkdir -p "${YOLOV6_WEIGHTS_DIR}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    log_info "[dry-run] link ${src} -> ${dest}"
    return 0
  fi
  if [[ -L "${dest}" || -f "${dest}" ]] && [[ "${FORCE}" == "false" ]]; then
    log_info "YOLOv6 link present, skipping: ${dest}"
    return 0
  fi
  ln -sf "$(realpath "${src}")" "${dest}"
  log_info "YOLOv6 weights link: ${dest}"
}

require_git() {
  if ! command -v git >/dev/null 2>&1; then
    log_error "git is required for --sync-sources"
    exit 1
  fi
}

rsync_tree() {
  local src="$1"
  local dest="$2"
  if [[ ! -d "${src}" ]]; then
    log_error "Missing source tree: ${src}"
    return 1
  fi
  if [[ "${DRY_RUN}" == "true" ]]; then
    log_info "[dry-run] rsync ${src}/ -> ${dest}/"
    return 0
  fi
  mkdir -p "${dest}"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "${src}/" "${dest}/"
  else
    cp -a "${src}/." "${dest}/"
  fi
}

clone_shallow() {
  local url="$1"
  local branch="$2"
  local dest="$3"
  if [[ "${DRY_RUN}" == "true" ]]; then
    log_info "[dry-run] git clone --depth 1 --branch ${branch} ${url} ${dest}"
    return 0
  fi
  rm -rf "${dest}"
  git clone --depth 1 --branch "${branch}" "${url}" "${dest}"
}

# Sync full upstream fork trees so train.py, utils/, models/, and data/ stay compatible.
sync_fork_vendor() {
  local name="$1"
  local url="$2"
  local branch="$3"
  local dest_root="$4"
  local marker_file="$5"
  shift 5
  local -a required_paths=("$@")

  local path ok=true
  for path in "${required_paths[@]}"; do
    if [[ ! -e "${path}" ]]; then
      ok=false
      break
    fi
  done
  if [[ "${ok}" == "true" && -f "${marker_file}" ]] && grep -qF "@${branch}" "${marker_file}"; then
    log_info "${name} vendor tree up to date (${branch})"
    return 0
  fi

  log_info "Restoring ${name} vendor tree from ${url} (${branch})..."
  local tmp="${REPO_ROOT}/.cache/${name}-src"
  clone_shallow "${url}" "${branch}" "${tmp}"
  mkdir -p "${dest_root}"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete --exclude .git "${tmp}/" "${dest_root}/"
  else
    rm -rf "${dest_root:?}"/*
    cp -a "${tmp}/." "${dest_root}/"
  fi
  echo "${url}@${branch}" > "${marker_file}"
  if [[ "${DRY_RUN}" != "true" ]]; then
    "${PYTHON:-python3}" "${REPO_ROOT}/scripts/patch_vendor_torch_load.py" || true
    "${PYTHON:-python3}" "${REPO_ROOT}/scripts/patch_vendor_regression.py" || true
  fi
  log_info "${name} vendor tree synced"
}

sync_yolov3_sources() {
  sync_fork_vendor \
    "yolov3" \
    "https://github.com/ultralytics/yolov3.git" \
    "v9.0" \
    "${YOLO_ROOT}/yolov3" \
    "${YOLO_ROOT}/yolov3/.vendor-sync" \
    "${YOLO_ROOT}/yolov3/train.py"
}

sync_yolov5_sources() {
  sync_fork_vendor \
    "yolov5" \
    "https://github.com/ultralytics/yolov5.git" \
    "v7.0" \
    "${YOLO_ROOT}/yolov5" \
    "${YOLO_ROOT}/yolov5/.vendor-sync" \
    "${YOLO_ROOT}/yolov5/train.py"
}

sync_ultralytics_sources() {
  if [[ -d "${YOLO_ROOT}/ultralytics/ultralytics/models" ]]; then
    log_info "vendored ultralytics models/ already present"
    return 0
  fi
  log_info "Restoring ultralytics package from github.com/ultralytics/ultralytics ..."
  local tmp="${REPO_ROOT}/.cache/ultralytics-src"
  clone_shallow "https://github.com/ultralytics/ultralytics.git" "main" "${tmp}"
  rsync_tree "${tmp}/ultralytics" "${YOLO_ROOT}/ultralytics/ultralytics"
}

# YOLOv7/v9 checkpoints unpickle modules from WongKinYiu repos (not YOLOv5 or pip ultralytics).
sync_yolov7_sources() {
  sync_fork_vendor \
    "yolov7" \
    "https://github.com/WongKinYiu/yolov7.git" \
    "main" \
    "${YOLO_ROOT}/yolov7" \
    "${YOLO_ROOT}/yolov7/.vendor-sync" \
    "${YOLO_ROOT}/yolov7/train.py"
}

sync_yolov9_sources() {
  sync_fork_vendor \
    "yolov9" \
    "https://github.com/WongKinYiu/yolov9.git" \
    "main" \
    "${YOLO_ROOT}/yolov9" \
    "${YOLO_ROOT}/yolov9/.vendor-sync" \
    "${YOLO_ROOT}/yolov9/train.py"
}

sync_yolov6_sources() {
  local models_dir="${YOLO_ROOT}/YOLOv6/yolov6/models"
  if [[ -d "${models_dir}" && -f "${YOLO_ROOT}/YOLOv6/tools/train.py" ]]; then
    log_info "YOLOv6 package present (${models_dir})"
    return 0
  fi
  log_info "Restoring YOLOv6 models/ from https://github.com/meituan/YOLOv6 ..."
  local tmp="${REPO_ROOT}/.cache/yolov6-src"
  clone_shallow "https://github.com/meituan/YOLOv6.git" "main" "${tmp}"
  if [[ "${DRY_RUN}" != "true" ]]; then
    mkdir -p "${YOLO_ROOT}/YOLOv6"
    rsync_tree "${tmp}/yolov6/models" "${YOLO_ROOT}/YOLOv6/yolov6/models"
    rsync_tree "${tmp}/tools" "${YOLO_ROOT}/YOLOv6/tools"
    rsync_tree "${tmp}/configs" "${YOLO_ROOT}/YOLOv6/configs"
  fi
  log_info "YOLOv6 sources synced"
}

sync_sources() {
  require_git
  sync_yolov3_sources
  sync_yolov5_sources
  sync_yolov6_sources
  sync_yolov7_sources
  sync_yolov9_sources
  sync_ultralytics_sources
}

install_deps() {
  local python="${PYTHON:-python3}"
  local req="${REPO_ROOT}/requirements.txt"
  if [[ "${DRY_RUN}" == "true" ]]; then
    log_info "[dry-run] ${python} -m pip install -r ${req} tensorboard"
    return 0
  fi
  log_info "Installing Python dependencies (requirements.txt + tensorboard for YOLOv6)..."
  "${python}" -m pip install -r "${req}" tensorboard
}

download_weights() {
  load_url_overrides
  load_unavailable_names

  local -a urls=()
  local -a file_jobs=()   # entries: filename|readme_url
  if [[ "${RETRY_FAILED}" == "true" ]]; then
    collect_retry_failed_files
    for name in "${RETRY_FILES[@]}"; do
      file_jobs+=("${name}|${URL_OVERRIDE_PRIMARY[${name}]:-}")
    done
    log_info "Retry-failed mode: ${#file_jobs[@]} file(s) from overrides"
  elif [[ "${REGRESSION_ONLY}" == "true" ]]; then
    for url in "${REGRESSION_WEIGHTS[@]}"; do
      file_jobs+=("$(url_basename "${url}")|${url}")
    done
    log_info "Regression-only mode: ${#file_jobs[@]} weight file(s)"
  else
    collect_readme_urls
    declare -A seen_files=()
    for url in "${MODEL_URLS[@]}"; do
      local fn
      fn="$(url_basename "${url}")"
      if [[ -n "${seen_files[${fn}]:-}" ]]; then
        continue
      fi
      seen_files["${fn}"]=1
      file_jobs+=("${fn}|${url}")
    done
    log_info "README mode: ${#file_jobs[@]} unique file(s) (overrides applied when listed)"
  fi

  mkdir -p "${MODELS_DIR}"
  local downloaded=0 skipped=0 failed=0 unavailable=0

  for job in "${file_jobs[@]}"; do
    local filename="${job%%|*}"
    local url="${job#*|}"
    if ! matches_filter "${url}"; then
      continue
    fi
    local dest
    dest="${MODELS_DIR}/${filename}"

    if [[ "${LIST_ONLY}" == "true" ]]; then
      local -a list_urls=()
      resolve_urls_for_file "${filename}" "${url}" list_urls
      for list_url in "${list_urls[@]}"; do
        echo "${filename}  ${list_url}"
      done
      continue
    fi

    if is_unavailable "${filename}"; then
      log_warn "Skipping unavailable weight (no public mirror): ${filename}"
      unavailable=$((unavailable + 1))
      continue
    fi

    if should_skip_download "${dest}"; then
      log_info "Skipping existing: ${filename}"
      skipped=$((skipped + 1))
      continue
    fi
    if [[ -f "${dest}" ]]; then
      log_warn "Replacing corrupt or tiny file: ${filename}"
      rm -f "${dest}"
    fi

    local -a try_urls=()
    resolve_urls_for_file "${filename}" "${url}" try_urls
    if [[ ${#try_urls[@]} -gt 1 ]]; then
      log_info "Using URL override for ${filename}"
    fi

    if download_with_fallbacks "${dest}" "${try_urls[@]}"; then
      downloaded=$((downloaded + 1))
      log_info "Saved: ${dest}"
    else
      log_error "Failed all URLs for ${filename}:"
      for try_url in "${try_urls[@]}"; do
        log_error "  ${try_url}"
      done
      failed=$((failed + 1))
    fi
  done

  if [[ "${LIST_ONLY}" == "true" ]]; then
    return 0
  fi

  if [[ "${LINK_YOLOV6}" == "true" ]]; then
    link_yolov6_weight "yolov6n.pt"
  fi

  log_info "Weights summary: downloaded=${downloaded} skipped=${skipped} unavailable=${unavailable} failed=${failed}"
  if [[ "${unavailable}" -gt 0 ]]; then
    log_warn "Unavailable weights: see ${UNAVAILABLE_FILE}"
    log_warn "  (yolov1-tiny.weights — use yolov1.weights or YOLOv3+ instead)"
  fi
  if [[ "${failed}" -gt 0 ]]; then
    log_error "Some downloads failed (stale README URL or offline host). Try: ./scripts/download_yolo_models.sh --retry-failed --force"
    exit 1
  fi
}

main() {
  log_info "YOLO models directory: ${MODELS_DIR}"
  if [[ "${FORCE}" == "true" ]]; then
    log_info "Force mode: re-download even when files already exist"
  else
    log_info "Skip mode: existing valid weights are not re-downloaded (use --force to override)"
  fi

  if [[ "${INSTALL_DEPS}" == "true" ]]; then
    install_deps
  fi

  if [[ "${SYNC_SOURCES_ONLY}" == "true" ]]; then
    if [[ "${SYNC_SOURCES}" != "true" ]]; then
      log_error "--sync-sources-only requires source sync"
      exit 1
    fi
    sync_sources
    log_info "Done (sources only)."
    return 0
  fi

  download_weights

  if [[ "${SYNC_SOURCES}" == "true" ]]; then
    sync_sources
  fi

  log_info "Done. For regression tests run:"
  log_info "  ./scripts/download_yolo_models.sh --regression-only --sync-sources --install-deps"
  log_info "  ./scripts/regression_test.sh"
}

main "$@"
