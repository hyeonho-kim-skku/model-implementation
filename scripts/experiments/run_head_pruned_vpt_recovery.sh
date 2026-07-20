#!/usr/bin/env bash
# Run shallow/deep visual-prompt recovery from 40% head-pruned artifacts.

set -euo pipefail

GPU_ID="${GPU_ID:-7}"
PYTHON_BIN="${PYTHON_BIN:-/home/hyeonho/miniconda3/envs/resnet/bin/python}"
EPOCHS="${EPOCHS:-20}"
PROMPT_MODES="${PROMPT_MODES:-shallow,deep}"
NUM_PROMPT_TOKENS="${NUM_PROMPT_TOKENS:-1}"
DATASETS="${DATASETS:-cifar100,cub200,fgvc_aircraft,stanford_cars}"
TIMEZONE="${TIMEZONE:-Asia/Seoul}"
LOG_DIR="${LOG_DIR:-logs/head_pruned_vpt_recovery_$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"

declare -A CONFIGS=(
  [cifar100]="configs/timm_vit_pruned_vpt_cifar100.yaml"
  [cub200]="configs/timm_vit_pruned_vpt_cub200.yaml"
  [fgvc_aircraft]="configs/timm_vit_pruned_vpt_fgvc_aircraft.yaml"
  [stanford_cars]="configs/timm_vit_pruned_vpt_stanford_cars.yaml"
)

IFS=',' read -ra selected_modes <<< "$PROMPT_MODES"
IFS=',' read -ra selected_datasets <<< "$DATASETS"

for prompt_mode in "${selected_modes[@]}"; do
  if [[ "$prompt_mode" != "shallow" && "$prompt_mode" != "deep" ]]; then
    echo "Unknown prompt mode: $prompt_mode" >&2
    exit 1
  fi
  for dataset in "${selected_datasets[@]}"; do
    if [[ -z "${CONFIGS[$dataset]+x}" ]]; then
      echo "Unknown dataset: $dataset" >&2
      exit 1
    fi
    name="head_pruned_vpt_${prompt_mode}${NUM_PROMPT_TOKENS}_${dataset}"
    log_path="${LOG_DIR}/${name}.log"
    echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Starting $name"
    CUDA_VISIBLE_DEVICES="$GPU_ID" bash scripts/run.sh "${CONFIGS[$dataset]}" \
      --prompt-mode "$prompt_mode" \
      --num-prompt-tokens "$NUM_PROMPT_TOKENS" \
      --num-epochs "$EPOCHS" \
      --reset-classifier \
      --disable-progress \
      2>&1 | tee "$log_path"
    echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Finished $name"
  done
done

"$PYTHON_BIN" analysis/summarize_head_pruned_vpt_recovery.py \
  --log-dir "$LOG_DIR" \
  --prompt-modes "$PROMPT_MODES" \
  --output "$LOG_DIR/summary.csv" \
  --num-prompt-tokens "$NUM_PROMPT_TOKENS"
