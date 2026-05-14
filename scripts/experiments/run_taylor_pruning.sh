#!/usr/bin/env bash
# Prune LoRA fine-tuned baseline checkpoints for each downstream dataset using
# Taylor importance. The Taylor configs run supervised calibration batches
# before pruning so channel importance is ranked from weight * gradient.

set -euo pipefail

GPU_ID="${GPU_ID:-7}"
TIMEZONE="${TIMEZONE:-Asia/Seoul}"
LOG_DIR="${LOG_DIR:-logs/taylor_pruning_$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"

run_experiment() {
  local name="$1"
  local config_path="$2"
  local log_path="${LOG_DIR}/${name}.log"

  echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Starting ${name}"
  echo "Config: ${config_path}"
  echo "Log: ${log_path}"

  CUDA_VISIBLE_DEVICES="$GPU_ID" bash scripts/prune.sh "$config_path" \
    2>&1 | tee "$log_path"

  echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Finished ${name}"
}

run_experiment "taylor_pruning_cifar100" "configs/timm_vit_taylor_pruning_cifar100.yaml"
run_experiment "taylor_pruning_flowers102" "configs/timm_vit_taylor_pruning_flowers102.yaml"
run_experiment "taylor_pruning_cub200" "configs/timm_vit_taylor_pruning_cub200.yaml"
