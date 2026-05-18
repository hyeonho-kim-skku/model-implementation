#!/usr/bin/env bash
# Run Taylor layer-wise sensitivity sweeps for all downstream datasets.

set -euo pipefail

GPU_ID="${GPU_ID:-7}"
TIMEZONE="${TIMEZONE:-Asia/Seoul}"
LOG_DIR="${LOG_DIR:-logs/taylor_sensitivity_$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"

run_experiment() {
  local name="$1"
  local config_path="$2"
  local log_path="${LOG_DIR}/${name}.log"

  echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Starting ${name}"
  echo "Config: ${config_path}"
  echo "Log: ${log_path}"
  if [ "$#" -gt 2 ]; then
    echo "Extra args: ${*:3}"
  fi

  CUDA_VISIBLE_DEVICES="$GPU_ID" bash scripts/sensitivity_taylor.sh "$config_path" "${@:3}" \
    2>&1 | tee "$log_path"

  echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Finished ${name}"
}

run_experiment "taylor_sensitivity_cifar100" \
  "configs/timm_vit_taylor_sensitivity_cifar100.yaml" \
  "$@"

run_experiment "taylor_sensitivity_flowers102" \
  "configs/timm_vit_taylor_sensitivity_flowers102.yaml" \
  "$@"

run_experiment "taylor_sensitivity_cub200" \
  "configs/timm_vit_taylor_sensitivity_cub200.yaml" \
  "$@"
