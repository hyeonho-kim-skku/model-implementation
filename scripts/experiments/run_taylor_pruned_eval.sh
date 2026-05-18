#!/usr/bin/env bash
# Evaluate artifacts produced by Taylor pruning LoRA fine-tuned baselines. This
# records the immediate post-pruning accuracy before any LoRA recovery is run.

set -euo pipefail

GPU_ID="${GPU_ID:-7}"
MLP_TAG="${MLP_TAG:-mlp030}"
TIMEZONE="${TIMEZONE:-Asia/Seoul}"
LOG_DIR="${LOG_DIR:-logs/taylor_pruned_eval_${MLP_TAG}_$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)}"
SAVE_JSON="${SAVE_JSON:-1}"

mkdir -p "$LOG_DIR"

run_experiment() {
  local name="$1"
  local config_path="$2"
  local dataset="$3"
  local artifact_path="./pruned/vit_base_${dataset}_lora50_${MLP_TAG}_taylor/pruned_timm_classifier.pth"
  local metrics_path="./pruned/vit_base_${dataset}_lora50_${MLP_TAG}_taylor/eval_metrics.json"
  local log_path="${LOG_DIR}/${name}.log"

  echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Starting ${name}"
  echo "Config: ${config_path}"
  echo "Artifact: ${artifact_path}"
  echo "Log: ${log_path}"

  local extra_args=()
  if [ "$SAVE_JSON" = "1" ]; then
    extra_args+=(--output-json "$metrics_path")
    echo "Metrics JSON: ${metrics_path}"
  fi

  CUDA_VISIBLE_DEVICES="$GPU_ID" bash scripts/eval_pruned.sh "$config_path" \
    --artifact-path "$artifact_path" \
    "${extra_args[@]}" \
    2>&1 | tee "$log_path"

  echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Finished ${name}"
}

run_experiment "taylor_pruned_eval_cifar100" \
  "configs/taylor_pruned_eval_cifar100.yaml" \
  "cifar100"

run_experiment "taylor_pruned_eval_flowers102" \
  "configs/taylor_pruned_eval_flowers102.yaml" \
  "flowers102"

run_experiment "taylor_pruned_eval_cub200" \
  "configs/taylor_pruned_eval_cub200.yaml" \
  "cub200"
