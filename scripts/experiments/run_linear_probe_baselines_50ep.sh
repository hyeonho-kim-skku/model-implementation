#!/usr/bin/env bash
# Run frozen-backbone linear probing baselines for the downstream datasets used
# in pruning and LoRA recovery experiments. These runs provide fixed reference
# accuracies for the unpruned pretrained backbone.

set -euo pipefail

GPU_ID="${GPU_ID:-7}"
EPOCHS="${EPOCHS:-50}"
LOG_DIR="${LOG_DIR:-logs/linear_probe_baselines_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"

run_experiment() {
  local name="$1"
  local config_path="$2"
  local log_path="${LOG_DIR}/${name}.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${name}"
  echo "Config: ${config_path}"
  echo "Epochs: ${EPOCHS}"
  echo "Log: ${log_path}"

  CUDA_VISIBLE_DEVICES="$GPU_ID" bash scripts/run.sh "$config_path" \
    --num-epochs "$EPOCHS" \
    --disable-progress \
    2>&1 | tee "$log_path"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${name}"
}

run_experiment "linear_probe_cifar100" "configs/timm_vit_linear_probe_cifar100.yaml"
run_experiment "linear_probe_flowers102" "configs/timm_vit_linear_probe_flowers102.yaml"
run_experiment "linear_probe_cub200" "configs/timm_vit_linear_probe_cub200.yaml"
