#!/usr/bin/env bash
# Run LoRA recovery from artifacts produced by Taylor pruning LoRA fine-tuned
# baselines. MLP_TAG selects which pruning artifact family to recover, e.g.
# mlp030 or mlp040.

set -euo pipefail

GPU_ID="${GPU_ID:-7}"
EPOCHS="${EPOCHS:-20}"
MLP_TAG="${MLP_TAG:-mlp030}"
ARTIFACT_TEMPLATE="${ARTIFACT_TEMPLATE:-}"
DATASETS="${DATASETS:-cifar100,cub200,fgvc_aircraft,stanford_cars}"
TIMEZONE="${TIMEZONE:-Asia/Seoul}"
LOG_DIR="${LOG_DIR:-logs/taylor_pruned_lora_recovery_${MLP_TAG}_$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"

declare -A CONFIGS=(
  [cifar100]="configs/timm_vit_taylor_pruned_lora_recovery_cifar100.yaml"
  [cub200]="configs/timm_vit_taylor_pruned_lora_recovery_cub200.yaml"
  [fgvc_aircraft]="configs/timm_vit_taylor_pruned_lora_recovery_fgvc_aircraft.yaml"
  [stanford_cars]="configs/timm_vit_taylor_pruned_lora_recovery_stanford_cars.yaml"
)

run_experiment() {
  local name="$1"
  local config_path="$2"
  local dataset="$3"
  local artifact_path="./pruned/vit_base_${dataset}_lora50_${MLP_TAG}_taylor/pruned_timm_classifier.pth"
  if [ -n "$ARTIFACT_TEMPLATE" ]; then
    artifact_path="${ARTIFACT_TEMPLATE//\{dataset\}/$dataset}"
  fi
  local log_path="${LOG_DIR}/${name}.log"

  echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Starting ${name}"
  echo "Config: ${config_path}"
  echo "Artifact: ${artifact_path}"
  if [ -n "$ARTIFACT_TEMPLATE" ]; then
    echo "Artifact template: ${ARTIFACT_TEMPLATE}"
  fi
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

IFS=',' read -ra selected_datasets <<< "$DATASETS"
for dataset in "${selected_datasets[@]}"; do
  run_experiment "taylor_pruned_lora_recovery_${dataset}" "${CONFIGS[$dataset]}" "$dataset"
done
