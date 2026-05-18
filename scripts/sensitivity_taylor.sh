#!/usr/bin/env bash
# Run Taylor layer-wise pruning sensitivity from a YAML config.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "사용법: CUDA_VISIBLE_DEVICES=<GPU_ID> bash scripts/sensitivity_taylor.sh <CONFIG_PATH> [추가 인자]"
    echo "예시: CUDA_VISIBLE_DEVICES=7 bash scripts/sensitivity_taylor.sh configs/timm_vit_taylor_sensitivity_cifar100.yaml --max-batches 2"
    exit 1
fi

CONFIG_PATH="$1"
shift

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" sensitivity_taylor.py --config "$CONFIG_PATH" "$@"
