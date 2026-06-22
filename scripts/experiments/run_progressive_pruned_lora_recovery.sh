#!/usr/bin/env bash
# Run LoRA recovery from progressive pruning artifacts.

set -euo pipefail

GPU_ID="${GPU_ID:-7}"
EPOCHS="${EPOCHS:-20}"
DATASETS="${DATASETS:-cifar100,cub200,fgvc_aircraft,stanford_cars}"
TARGETS="${TARGETS:-010,020,030,040,050,060}"
TIMEZONE="${TIMEZONE:-Asia/Seoul}"
LOG_DIR="${LOG_DIR:-logs/progressive_pruned_lora_recovery_$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"

declare -A CONFIGS=(
  [cifar100]="progressive_pruning/configs/recovery_cifar100.yaml"
  [cub200]="progressive_pruning/configs/recovery_cub200.yaml"
  [fgvc_aircraft]="progressive_pruning/configs/recovery_fgvc_aircraft.yaml"
  [stanford_cars]="progressive_pruning/configs/recovery_stanford_cars.yaml"
)

run_recovery() {
  local dataset="$1"
  local target="$2"
  local config_path="${CONFIGS[$dataset]}"
  local artifact_path="./pruned/progressive_baseline_${dataset}/target${target}/pruned_timm_classifier.pth"
  local name="progressive_pruned_lora_recovery_${dataset}_target${target}"
  local log_path="${LOG_DIR}/${name}.log"

  if [ ! -f "$artifact_path" ]; then
    echo "Missing artifact: ${artifact_path}" >&2
    exit 1
  fi

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

echo "Datasets: ${DATASETS}"
echo "Targets: ${TARGETS}"
echo "Log dir: ${LOG_DIR}"

IFS=',' read -ra selected_datasets <<< "$DATASETS"
IFS=',' read -ra selected_targets <<< "$TARGETS"

for dataset in "${selected_datasets[@]}"; do
  if [ -z "${CONFIGS[$dataset]+x}" ]; then
    echo "Unknown dataset: ${dataset}" >&2
    exit 1
  fi
  for target in "${selected_targets[@]}"; do
    run_recovery "$dataset" "$target"
  done
done
