#!/usr/bin/env bash
# Run uniform-5 and head-proportional VPT for selected datasets.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/hyeonho/miniconda3/envs/resnet/bin/python}"
EPOCHS="${EPOCHS:-20}"
DATASETS="${DATASETS:-cifar100,cub200,fgvc_aircraft,stanford_cars}"
LOG_DIR="${LOG_DIR:-logs/layerwise_vpt_recovery_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"
IFS=',' read -ra selected_datasets <<< "$DATASETS"

for dataset in "${selected_datasets[@]}"; do
  config="configs/timm_vit_pruned_vpt_${dataset}.yaml"
  if [[ ! -f "$config" ]]; then
    echo "Missing config: $config" >&2
    exit 1
  fi
  head_proportional_schedule="$($PYTHON_BIN -c \
    'import sys, yaml; c=yaml.safe_load(open(sys.argv[1])); print(",".join(map(str, c["head_proportional_prompt_tokens_per_layer"])))' \
    "$config")"

  for allocation in uniform5 head_proportional; do
    log_path="$LOG_DIR/${dataset}__${allocation}.log"
    common_args=(
      --config "$config"
      --prompt-mode deep
      --num-epochs "$EPOCHS"
      --reset-classifier
      --profile-macs
      --disable-progress
    )
    if [[ "$allocation" == "uniform5" ]]; then
      allocation_args=(
        --num-prompt-tokens 5
        --prompt-allocation-label uniform-5
      )
    else
      allocation_args=(
        --prompt-tokens-per-layer "$head_proportional_schedule"
        --prompt-allocation-label head-proportional-1to1
      )
    fi

    echo "Starting ${dataset}/${allocation}"
    "$PYTHON_BIN" main.py "${common_args[@]}" "${allocation_args[@]}" \
      2>&1 | tee "$log_path"
    echo "Finished ${dataset}/${allocation}"
  done
done

"$PYTHON_BIN" analysis/summarize_layerwise_vpt_recovery.py \
  --log-dir "$LOG_DIR" \
  --datasets "$DATASETS" \
  --output "$LOG_DIR/summary.csv"
