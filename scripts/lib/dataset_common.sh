#!/usr/bin/env bash
# Shared helpers for dataset download scripts.

set -euo pipefail

# shellcheck disable=SC2034
if [[ -z "${DATASET_COMMON_LOADED:-}" ]]; then
  DATASET_COMMON_LOADED=1

  _dataset_common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_dataset_common_dir}/../.." && pwd)"
  DATASETS_DIR="${DATASETS_DIR:-${REPO_ROOT}/datasets}"

  log_info() {
    printf '[INFO] %s\n' "$*"
  }

  log_warn() {
    printf '[WARN] %s\n' "$*" >&2
  }

  log_error() {
    printf '[ERROR] %s\n' "$*" >&2
  }

  require_cmd() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
      log_error "Required command not found: ${cmd}"
      exit 1
    fi
  }

  ensure_datasets_dir() {
    mkdir -p "${DATASETS_DIR}"
  }

  # download_file URL OUTPUT_PATH
  # Uses curl with resume (-C -). Falls back to wget if curl is missing.
  download_file() {
    local url="$1"
    local output="$2"
    local output_dir
    output_dir="$(dirname "${output}")"
    mkdir -p "${output_dir}"

    if [[ -f "${output}" ]]; then
      log_info "Already present, skipping download: ${output}"
      return 0
    fi

    log_info "Downloading ${url}"
    log_info "  -> ${output}"

    if command -v curl >/dev/null 2>&1; then
      curl -L -C - --fail --retry 3 --retry-delay 5 -o "${output}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
      wget -c -O "${output}" "${url}"
    else
      log_error "Install curl or wget to download datasets."
      exit 1
    fi
  }

  # extract_tar ARCHIVE DEST_DIR
  extract_tar() {
    local archive="$1"
    local dest="$2"
    mkdir -p "${dest}"
    log_info "Extracting ${archive} -> ${dest}"
    tar -xf "${archive}" -C "${dest}"
    rm -f "${archive}"
  }

  # extract_zip ARCHIVE DEST_DIR
  extract_zip() {
    local archive="$1"
    local dest="$2"
    mkdir -p "${dest}"
    log_info "Extracting ${archive} -> ${dest}"
    if command -v unzip >/dev/null 2>&1; then
      unzip -q "${archive}" -d "${dest}"
    else
      log_error "unzip is required to extract ${archive}"
      exit 1
    fi
    rm -f "${archive}"
  }
fi
