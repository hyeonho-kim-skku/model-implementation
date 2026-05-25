#!/usr/bin/env bash
# Prune LoRA fine-tuned baseline checkpoints for each downstream dataset.
# The pruning configs should point to the fixed LoRA baseline checkpoints that
# define the source models for post-fine-tuning pruning experiments.

set -euo pipefail

GPU_ID="${GPU_ID:-7}"
TIMEZONE="${TIMEZONE:-Asia/Seoul}"
LOG_DIR="${LOG_DIR:-logs/lora_pruning_$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)}"
# Set INSPECT_GROUPS=1 to print target shape changes. This is useful when
# validating a new pruning target, but it makes logs longer, so the default
# experiment run keeps it disabled.
INSPECT_GROUPS="${INSPECT_GROUPS:-0}"

mkdir -p "$LOG_DIR"

run_experiment() {
  local name="$1"
  local config_path="$2"
  local log_path="${LOG_DIR}/${name}.log"

  echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Starting ${name}"
  echo "Config: ${config_path}"
  echo "Log: ${log_path}"

  local extra_args=()
  if [ "$INSPECT_GROUPS" = "1" ]; then
    extra_args+=(--inspect-groups)
  fi

  CUDA_VISIBLE_DEVICES="$GPU_ID" bash scripts/prune.sh "$config_path" "${extra_args[@]}" \
    2>&1 | tee "$log_path"

  echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Finished ${name}"
}

run_experiment "lora_pruning_cifar100" "configs/timm_vit_pruning_cifar100.yaml"
run_experiment "lora_pruning_flowers102" "configs/timm_vit_pruning_flowers102.yaml"
run_experiment "lora_pruning_cub200" "configs/timm_vit_pruning_cub200.yaml"
