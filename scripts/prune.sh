#!/bin/bash
# scripts/prune.sh

if [ -z "$1" ]; then
    echo "사용법: CUDA_VISIBLE_DEVICES=<GPU_ID> bash scripts/prune.sh <CONFIG_PATH> [추가 인자]"
    echo "예시: CUDA_VISIBLE_DEVICES=7 bash scripts/prune.sh configs/timm_dinov2_pruning.yaml --pruning-ratio 0.3"
    exit 1
fi

CONFIG_PATH=$1
shift 1

python prune.py --config "$CONFIG_PATH" "$@"
