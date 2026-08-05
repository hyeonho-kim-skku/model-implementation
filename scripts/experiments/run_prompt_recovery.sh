#!/usr/bin/env bash
# Run selected VPT/KV prompt recovery variants across dataset configs.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/hyeonho/miniconda3/envs/resnet/bin/python}"
EPOCHS="${EPOCHS:-20}"
DATASETS="${DATASETS:-cifar100}"
EXPERIMENTS="${EXPERIMENTS:-kv5,kv8}"
LOG_DIR="${LOG_DIR:-logs/prompt_recovery_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$LOG_DIR"
IFS=',' read -ra selected_datasets <<< "$DATASETS"
IFS=',' read -ra selected_experiments <<< "$EXPERIMENTS"

for dataset in "${selected_datasets[@]}"; do
  # The existing VPT config is the shared dataset/artifact/training base;
  # prompt components and token counts are selected below via CLI overrides.
  base_config="configs/timm_vit_pruned_vpt_${dataset}.yaml"
  if [[ ! -f "$base_config" ]]; then
    echo "Missing config: $base_config" >&2
    exit 1
  fi
  for experiment in "${selected_experiments[@]}"; do
    common_args=(
      --config "$base_config"
      --model timm_pruned_prompt
      --num-epochs "$EPOCHS"
      --profile-macs
      --disable-progress
    )
    case "$experiment" in
      vpt5)
        experiment_args=(
          --prompt-components vpt
          --prompt-mode deep
          --num-prompt-tokens 5
          --prompt-allocation-label vpt-uniform-5
        )
        ;;
      lora4_vpt5)
        experiment_args=(
          --prompt-components vpt
          --prompt-mode deep
          --num-prompt-tokens 5
          --lora-rank 4
          --lora-modules qkv,proj,mlp
          --qkv-lora-components q,k,v
          --prompt-allocation-label lora4-vpt-uniform-5
        )
        ;;
      lora_then_vpt5)
        initial_recovery_checkpoint="$($PYTHON_BIN -c \
          'import sys, yaml; c=yaml.safe_load(open(sys.argv[1])); p=c.get("lora_recovery_checkpoint"); assert p, "lora_recovery_checkpoint is required"; print(p)' \
          "$base_config")"
        experiment_args=(
          --no-reset-classifier
          --initial-recovery-checkpoint "$initial_recovery_checkpoint"
          --prompt-components vpt
          --prompt-mode deep
          --num-prompt-tokens 5
          --prompt-allocation-label lora-then-vpt-uniform-5
        )
        ;;
      vpt_head_proportional)
        head_proportional_schedule="$($PYTHON_BIN -c \
          'import sys, yaml; c=yaml.safe_load(open(sys.argv[1])); print(",".join(map(str, c["head_proportional_prompt_tokens_per_layer"])))' \
          "$base_config")"
        experiment_args=(
          --prompt-components vpt
          --prompt-mode deep
          --prompt-tokens-per-layer "$head_proportional_schedule"
          --prompt-allocation-label vpt-head-proportional-1to1
        )
        ;;
      kv5)
        experiment_args=(
          --prompt-components kv
          --num-kv-prompt-tokens 5
          --prompt-allocation-label kv-uniform-5
        )
        ;;
      kv8)
        experiment_args=(
          --prompt-components kv
          --num-kv-prompt-tokens 8
          --prompt-allocation-label kv-uniform-8
        )
        ;;
      kv_separate4)
        experiment_args=(
          --prompt-components kv
          --num-kv-prompt-tokens 4
          --no-share-kv-prompt
          --prompt-allocation-label kv-separate-uniform-4
        )
        ;;
      vpt5_kv1)
        experiment_args=(
          --prompt-components vpt,kv
          --prompt-mode deep
          --num-prompt-tokens 5
          --num-kv-prompt-tokens 1
          --prompt-allocation-label vpt5-kv1
        )
        ;;
      vpt5_kv5)
        experiment_args=(
          --prompt-components vpt,kv
          --prompt-mode deep
          --num-prompt-tokens 5
          --num-kv-prompt-tokens 5
          --prompt-allocation-label vpt5-kv5
        )
        ;;
      vpt5_kv_separate1)
        experiment_args=(
          --prompt-components vpt,kv
          --prompt-mode deep
          --num-prompt-tokens 5
          --num-kv-prompt-tokens 1
          --no-share-kv-prompt
          --prompt-allocation-label vpt5-kv-separate1
        )
        ;;
      vpt5_kv_separate5)
        experiment_args=(
          --prompt-components vpt,kv
          --prompt-mode deep
          --num-prompt-tokens 5
          --num-kv-prompt-tokens 5
          --no-share-kv-prompt
          --prompt-allocation-label vpt5-kv-separate5
        )
        ;;
      *)
        echo "Unknown experiment: $experiment" >&2
        exit 1
        ;;
    esac

    if [[ "$experiment" == "lora_then_vpt5" ]]; then
      recovery_classifier_args=()
    else
      recovery_classifier_args=(--reset-classifier)
    fi

    log_path="$LOG_DIR/${dataset}__${experiment}.log"
    echo "Starting ${dataset}/${experiment}"
    if [[ "$DRY_RUN" == "1" ]]; then
      printf 'Command:'
      printf ' %q' "$PYTHON_BIN" main.py "${common_args[@]}" "${recovery_classifier_args[@]}" "${experiment_args[@]}"
      printf '\n'
      continue
    fi
    "$PYTHON_BIN" main.py "${common_args[@]}" "${recovery_classifier_args[@]}" "${experiment_args[@]}" \
      2>&1 | tee "$log_path"
    echo "Finished ${dataset}/${experiment}"
  done
done

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

"$PYTHON_BIN" analysis/summarize_prompt_recovery.py \
  --log-dir "$LOG_DIR" \
  --datasets "$DATASETS" \
  --output "$LOG_DIR/summary.csv"
