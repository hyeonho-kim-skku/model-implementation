#!/usr/bin/env bash
# Run progressive pruning from a YAML config.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "사용법: GPU_ID=<GPU_ID> bash scripts/progressive_pruning.sh <CONFIG_PATH> [추가 인자]"
    echo "예시: GPU_ID=5 bash scripts/progressive_pruning.sh progressive_pruning/configs/baseline_cifar100.yaml"
    echo "Smoke 예시: GPU_ID=5 bash scripts/progressive_pruning.sh progressive_pruning/configs/baseline_cifar100.yaml --target-ratios 0.1,0.2 --calibration-batches 1 --no-eval-each-step --no-save-artifacts --output-dir /tmp/progressive_pruning_smoke --no-verbose"
    exit 1
fi

CONFIG_PATH="$1"
shift

TIMEZONE="${TIMEZONE:-Asia/Seoul}"
TIMESTAMP="$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)"
CONFIG_NAME="$(basename "$CONFIG_PATH")"
CONFIG_NAME="${CONFIG_NAME%.*}"
LOG_DIR="${LOG_DIR:-logs/progressive_pruning}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${CONFIG_NAME}_${TIMESTAMP}.log}"
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-}}"

mkdir -p "$LOG_DIR"

echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Starting progressive_pruning_${CONFIG_NAME}"
echo "Config: ${CONFIG_PATH}"
echo "Log: ${LOG_PATH}"
echo "Python: $(command -v python)"
echo "GPU_ID: ${GPU_ID:-unset}"
if [ "$#" -gt 0 ]; then
    echo "Extra args: $*"
fi

if [ -n "$GPU_ID" ]; then
    CUDA_VISIBLE_DEVICES="$GPU_ID" python progressive_pruning/run.py \
        --config "$CONFIG_PATH" \
        "$@" \
        2>&1 | tee "$LOG_PATH"
else
    python progressive_pruning/run.py \
        --config "$CONFIG_PATH" \
        "$@" \
        2>&1 | tee "$LOG_PATH"
fi

echo "[$(TZ="$TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z')] Finished progressive_pruning_${CONFIG_NAME}"
