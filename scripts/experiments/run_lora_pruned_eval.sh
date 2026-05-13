#!/usr/bin/env bash
# Evaluate artifacts produced by pruning LoRA fine-tuned baselines. This records
# the immediate post-pruning accuracy before any LoRA recovery is trained.

set -euo pipefail

GPU_ID="${GPU_ID:-7}"
TIMEZONE="${TIMEZONE:-Asia/Seoul}"
LOG_DIR="${LOG_DIR:-logs/lora_pruned_eval_$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)}"
SAVE_JSON="${SAVE_JSON:-1}"

mkdir -p "$LOG_DIR"

run_experiment() {
  local name="$1"
  local config_path="$2"
  local metrics_path="$3"
  local log_path="${LOG_DIR}/${name}.log"

  echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Starting ${name}"
  echo "Config: ${config_path}"
  echo "Log: ${log_path}"

  local extra_args=()
  if [ "$SAVE_JSON" = "1" ]; then
    extra_args+=(--output-json "$metrics_path")
    echo "Metrics JSON: ${metrics_path}"
  fi

  CUDA_VISIBLE_DEVICES="$GPU_ID" bash scripts/eval_pruned.sh "$config_path" "${extra_args[@]}" \
    2>&1 | tee "$log_path"

  echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Finished ${name}"
}

run_experiment "lora_pruned_eval_cifar100" \
  "configs/pruned_eval_cifar100.yaml" \
  "pruned/vit_base_cifar100_lora50_mlp020/eval_metrics.json"

run_experiment "lora_pruned_eval_flowers102" \
  "configs/pruned_eval_flowers102.yaml" \
  "pruned/vit_base_flowers102_lora50_mlp020/eval_metrics.json"

run_experiment "lora_pruned_eval_cub200" \
  "configs/pruned_eval_cub200.yaml" \
  "pruned/vit_base_cub200_lora50_mlp020/eval_metrics.json"
