#!/usr/bin/env bash
# Prune LoRA fine-tuned baseline checkpoints for each downstream dataset using
# Taylor importance. The Taylor configs run supervised calibration batches
# before pruning so channel importance is ranked from weight * gradient.

set -euo pipefail

GPU_ID="${GPU_ID:-7}"
PRUNING_RATIO="${PRUNING_RATIO:-0.30}"
MLP_TAG="${MLP_TAG:-mlp030}"
DATASETS="${DATASETS:-cifar100,cub200,fgvc_aircraft,stanford_cars}"
TIMEZONE="${TIMEZONE:-Asia/Seoul}"
LOG_DIR="${LOG_DIR:-logs/taylor_pruning_${MLP_TAG}_$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"

declare -A CONFIGS=(
  [cifar100]="configs/timm_vit_taylor_pruning_cifar100.yaml"
  [cub200]="configs/timm_vit_taylor_pruning_cub200.yaml"
  [fgvc_aircraft]="configs/timm_vit_taylor_pruning_fgvc_aircraft.yaml"
  [stanford_cars]="configs/timm_vit_taylor_pruning_stanford_cars.yaml"
)

run_experiment() {
  local name="$1"
  local config_path="$2"
  local dataset="$3"
  local output_dir="./pruned/vit_base_${dataset}_lora50_${MLP_TAG}_taylor"
  local log_path="${LOG_DIR}/${name}.log"

  echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Starting ${name}"
  echo "Config: ${config_path}"
  echo "Pruning ratio: ${PRUNING_RATIO}"
  echo "Output dir: ${output_dir}"
  echo "Log: ${log_path}"
  if [ "$#" -gt 3 ]; then
    echo "Extra args: ${*:4}"
  fi

  CUDA_VISIBLE_DEVICES="$GPU_ID" bash scripts/prune.sh "$config_path" \
    --pruning-ratio "$PRUNING_RATIO" \
    --output-dir "$output_dir" \
    "${@:4}" \
    2>&1 | tee "$log_path"

  echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Finished ${name}"
}

echo "Datasets: ${DATASETS}"

IFS=',' read -ra selected_datasets <<< "$DATASETS"
for dataset in "${selected_datasets[@]}"; do
  run_experiment "taylor_pruning_${dataset}" "${CONFIGS[$dataset]}" "$dataset" "$@"
done
