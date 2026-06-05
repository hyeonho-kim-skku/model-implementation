#!/usr/bin/env bash
# Run cached gate-Taylor global MLP pruning sweeps.

set -euo pipefail

GPU_ID="${GPU_ID:-7}"
DATASETS="${DATASETS:-cifar100,cub200,fgvc_aircraft,stanford_cars}"
TIMEZONE="${TIMEZONE:-Asia/Seoul}"
LOG_DIR="${LOG_DIR:-logs/gate_taylor_global_$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"

declare -A CONFIGS=(
  [cifar100]="configs/timm_vit_gate_taylor_global_cifar100.yaml"
  [cub200]="configs/timm_vit_gate_taylor_global_cub200.yaml"
  [fgvc_aircraft]="configs/timm_vit_gate_taylor_global_fgvc_aircraft.yaml"
  [stanford_cars]="configs/timm_vit_gate_taylor_global_stanford_cars.yaml"
)

run_experiment() {
  local dataset="$1"
  local config_path="$2"
  local name="gate_taylor_global_${dataset}"
  local log_path="${LOG_DIR}/${name}.log"

  echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Starting ${name}"
  echo "Config: ${config_path}"
  echo "Log: ${log_path}"
  if [ "$#" -gt 2 ]; then
    echo "Extra args: ${*:3}"
  fi

  CUDA_VISIBLE_DEVICES="$GPU_ID" python global_gate_taylor_pruning.py \
    --config "$config_path" \
    "${@:3}" \
    2>&1 | tee "$log_path"

  echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Finished ${name}"
}

echo "Datasets: ${DATASETS}"

IFS=',' read -ra selected_datasets <<< "$DATASETS"
for dataset in "${selected_datasets[@]}"; do
  if [ -z "${CONFIGS[$dataset]+x}" ]; then
    echo "Unknown dataset: ${dataset}" >&2
    exit 1
  fi
  run_experiment "$dataset" "${CONFIGS[$dataset]}" "$@"
done
