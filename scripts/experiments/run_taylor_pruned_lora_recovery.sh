#!/usr/bin/env bash
# Run LoRA recovery from artifacts produced by Taylor pruning LoRA fine-tuned
# baselines. MLP_TAG selects which pruning artifact family to recover, e.g.
# mlp030 or mlp040.

set -euo pipefail

GPU_ID="${GPU_ID:-7}"
EPOCHS="${EPOCHS:-20}"
MLP_TAG="${MLP_TAG:-mlp030}"
TIMEZONE="${TIMEZONE:-Asia/Seoul}"
LOG_DIR="${LOG_DIR:-logs/taylor_pruned_lora_recovery_${MLP_TAG}_$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"

run_experiment() {
  local name="$1"
  local config_path="$2"
  local dataset="$3"
  local artifact_path="./pruned/vit_base_${dataset}_lora50_${MLP_TAG}_taylor/pruned_timm_classifier.pth"
  local log_path="${LOG_DIR}/${name}.log"

  echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Starting ${name}"
  echo "Config: ${config_path}"
  echo "Artifact: ${artifact_path}"
  echo "Epochs: ${EPOCHS}"
  echo "Log: ${log_path}"

  CUDA_VISIBLE_DEVICES="$GPU_ID" bash scripts/run.sh "$config_path" \
    --artifact-path "$artifact_path" \
    --num-epochs "$EPOCHS" \
    --disable-progress \
    2>&1 | tee "$log_path"

  echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Finished ${name}"
}

run_experiment "taylor_pruned_lora_recovery_cifar100" "configs/timm_vit_taylor_pruned_lora_recovery_cifar100.yaml" "cifar100"
run_experiment "taylor_pruned_lora_recovery_flowers102" "configs/timm_vit_taylor_pruned_lora_recovery_flowers102.yaml" "flowers102"
run_experiment "taylor_pruned_lora_recovery_cub200" "configs/timm_vit_taylor_pruned_lora_recovery_cub200.yaml" "cub200"
