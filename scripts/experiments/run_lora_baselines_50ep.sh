#!/usr/bin/env bash
# Run LoRA fine-tuning baselines for the downstream datasets used in pruning
# experiments. These runs provide fixed reference accuracies for task-adapted
# unpruned backbones before any structured pruning is applied.

set -euo pipefail

GPU_ID="${GPU_ID:-7}"
EPOCHS="${EPOCHS:-50}"
TIMEZONE="${TIMEZONE:-Asia/Seoul}"
LOG_DIR="${LOG_DIR:-logs/lora_baselines_$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"

run_experiment() {
  local name="$1"
  local config_path="$2"
  local log_path="${LOG_DIR}/${name}.log"

  echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Starting ${name}"
  echo "Config: ${config_path}"
  echo "Epochs: ${EPOCHS}"
  echo "Log: ${log_path}"

  CUDA_VISIBLE_DEVICES="$GPU_ID" bash scripts/run.sh "$config_path" \
    --num-epochs "$EPOCHS" \
    --disable-progress \
    2>&1 | tee "$log_path"

  echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Finished ${name}"
}

run_experiment "lora_cifar100" "configs/timm_vit_lora_cifar100.yaml"
run_experiment "lora_flowers102" "configs/timm_vit_lora_flowers102.yaml"
run_experiment "lora_cub200" "configs/timm_vit_lora_cub200.yaml"
