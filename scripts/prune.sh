#!/bin/bash
# scripts/prune.sh

if [ -z "$1" ]; then
    echo "사용법: bash scripts/prune.sh <CONFIG_PATH> [추가 인자]"
    echo "예시: bash scripts/prune.sh configs/timm_dinov2_pruning.yaml --pruning-ratio 0.3"
    exit 1
fi

GPU_ID=7

CONFIG_PATH=$1
shift 1

CUDA_VISIBLE_DEVICES=$GPU_ID python prune.py --config "$CONFIG_PATH" "$@"
