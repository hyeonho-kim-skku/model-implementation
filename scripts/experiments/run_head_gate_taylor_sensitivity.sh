#!/usr/bin/env bash
# Run Head Gate Taylor attention-head sensitivity sweeps.
#
# Typical usage:
#   GPU_ID=7 DATASETS=cifar100 bash scripts/experiments/run_head_gate_taylor_sensitivity.sh
#
# This script is only orchestration: it selects one YAML config per dataset and
# forwards all extra CLI args to sensitivity_taylor.py through
# scripts/sensitivity_taylor.sh. The actual pruning flow is documented in
# sensitivity_taylor.py and pruning/structured.py.

set -euo pipefail

GPU_ID="${GPU_ID:-7}"
DATASETS="${DATASETS:-cifar100,cub200,fgvc_aircraft,stanford_cars}"
TIMEZONE="${TIMEZONE:-Asia/Seoul}"
LOG_DIR="${LOG_DIR:-logs/head_gate_taylor_sensitivity_$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"

declare -A CONFIGS=(
  [cifar100]="configs/timm_vit_head_gate_taylor_sensitivity/cifar100.yaml"
  [cub200]="configs/timm_vit_head_gate_taylor_sensitivity/cub200.yaml"
  [fgvc_aircraft]="configs/timm_vit_head_gate_taylor_sensitivity/fgvc_aircraft.yaml"
  [stanford_cars]="configs/timm_vit_head_gate_taylor_sensitivity/stanford_cars.yaml"
)

run_experiment() {
  local name="$1"
  local config_path="$2"
  local log_path="${LOG_DIR}/${name}.log"

  echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Starting ${name}"
  echo "Config: ${config_path}"
  echo "Log: ${log_path}"
  if [ "$#" -gt 2 ]; then
    echo "Extra args: ${*:3}"
  fi

  CUDA_VISIBLE_DEVICES="$GPU_ID" bash scripts/sensitivity_taylor.sh "$config_path" "${@:3}" \
    2>&1 | tee "$log_path"

  echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Finished ${name}"
}

echo "Datasets: ${DATASETS}"

IFS=',' read -ra selected_datasets <<< "$DATASETS"
for dataset in "${selected_datasets[@]}"; do
  if [[ -z "${CONFIGS[$dataset]+x}" ]]; then
    echo "Unknown dataset: ${dataset}" >&2
    echo "Available datasets: ${!CONFIGS[*]}" >&2
    exit 1
  fi
  run_experiment "head_gate_taylor_sensitivity_${dataset}" "${CONFIGS[$dataset]}" "$@"
done
