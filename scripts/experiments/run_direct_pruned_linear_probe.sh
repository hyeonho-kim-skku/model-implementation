#!/usr/bin/env bash
# Directly prune pretrained TIMM ViT backbones for each downstream dataset, then
# run frozen-backbone linear probing on the pruned artifacts.

set -euo pipefail

GPU_ID="${GPU_ID:-7}"
EPOCHS="${EPOCHS:-50}"
TIMEZONE="${TIMEZONE:-Asia/Seoul}"
LOG_DIR="${LOG_DIR:-logs/direct_pruned_linear_probe_$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)}"
# Set INSPECT_GROUPS=1 to print a few Torch-Pruning dependency groups during
# the pruning phase. The default keeps logs shorter for full experiment runs.
INSPECT_GROUPS="${INSPECT_GROUPS:-0}"

mkdir -p "$LOG_DIR"

run_prune() {
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

run_probe() {
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

run_prune "direct_prune_pretrained_cifar100" "configs/timm_vit_prune_from_pretrained_cifar100.yaml"
run_probe "direct_pruned_linear_probe_cifar100" "configs/timm_vit_pruned_linear_probe_cifar100.yaml"

run_prune "direct_prune_pretrained_flowers102" "configs/timm_vit_prune_from_pretrained_flowers102.yaml"
run_probe "direct_pruned_linear_probe_flowers102" "configs/timm_vit_pruned_linear_probe_flowers102.yaml"

run_prune "direct_prune_pretrained_cub200" "configs/timm_vit_prune_from_pretrained_cub200.yaml"
run_probe "direct_pruned_linear_probe_cub200" "configs/timm_vit_pruned_linear_probe_cub200.yaml"
