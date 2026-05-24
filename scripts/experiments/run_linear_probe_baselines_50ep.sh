#!/usr/bin/env bash
# Run frozen-backbone linear probing baselines for the downstream datasets used
# in pruning and LoRA recovery experiments. These runs provide fixed reference
# accuracies for the unpruned pretrained backbone.

set -euo pipefail

GPU_ID="${GPU_ID:-7}"
EPOCHS="${EPOCHS:-50}"
DATASETS="${DATASETS:-cifar100,flowers102,cub200,fgvc_aircraft,stanford_cars}"
TIMEZONE="${TIMEZONE:-Asia/Seoul}"
LOG_DIR="${LOG_DIR:-logs/linear_probe_baselines_$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"

declare -A CONFIGS=(
  [cifar100]="configs/timm_vit_linear_probe_cifar100.yaml"
  [flowers102]="configs/timm_vit_linear_probe_flowers102.yaml"
  [cub200]="configs/timm_vit_linear_probe_cub200.yaml"
  [fgvc_aircraft]="configs/timm_vit_linear_probe_fgvc_aircraft.yaml"
  [stanford_cars]="configs/timm_vit_linear_probe_stanford_cars.yaml"
)

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

echo "Datasets: ${DATASETS}"

IFS=',' read -ra selected_datasets <<< "$DATASETS"
for dataset in "${selected_datasets[@]}"; do
  run_experiment "linear_probe_${dataset}" "${CONFIGS[$dataset]}"
done
