#!/usr/bin/env bash
# Run uniform and pruning-aware deep VPT recovery sequentially on idle GPU 7.

set -euo pipefail

GPU_ID=7
PYTHON_BIN="${PYTHON_BIN:-/home/hyeonho/miniconda3/envs/resnet/bin/python}"
EPOCHS="${EPOCHS:-20}"
TIMEZONE="${TIMEZONE:-Asia/Seoul}"
LOG_DIR="${LOG_DIR:-logs/cifar100_layerwise_vpt_$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to verify that GPU 7 is idle." >&2
  exit 1
fi

check_gpu_idle() {
  local active_pids
  active_pids="$(nvidia-smi --id="$GPU_ID" --query-compute-apps=pid \
    --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
  if [[ -n "$active_pids" ]]; then
    echo "GPU $GPU_ID is in use by compute process(es): $active_pids" >&2
    exit 1
  fi
}

mkdir -p "$LOG_DIR"

declare -a RUNS=(
  "uniform5:configs/timm_vit_pruned_vpt_deep5_cifar100.yaml"
  "pruning_aware:configs/timm_vit_pruned_vpt_pruning_aware_cifar100.yaml"
)

for run_spec in "${RUNS[@]}"; do
  check_gpu_idle
  name="${run_spec%%:*}"
  config="${run_spec#*:}"
  log_path="$LOG_DIR/${name}.log"
  echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Starting $name"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" main.py \
    --config "$config" \
    --num-epochs "$EPOCHS" \
    --reset-classifier \
    --profile-macs \
    --disable-progress \
    2>&1 | tee "$log_path"
  echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Finished $name"
done

"$PYTHON_BIN" analysis/summarize_layerwise_vpt_recovery.py \
  --log-dir "$LOG_DIR" \
  --dataset cifar100 \
  --output "$LOG_DIR/summary.csv"
