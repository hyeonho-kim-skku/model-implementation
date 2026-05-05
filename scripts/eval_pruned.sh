#!/bin/bash
# scripts/eval_pruned.sh

if [ -z "$1" ]; then
    echo "사용법: CUDA_VISIBLE_DEVICES=<GPU_ID> bash scripts/eval_pruned.sh <CONFIG_PATH> [추가 인자]"
    echo "예시: CUDA_VISIBLE_DEVICES=7 bash scripts/eval_pruned.sh configs/timm_dinov2_pruned_eval.yaml"
    echo "JSON 저장 예시: CUDA_VISIBLE_DEVICES=7 bash scripts/eval_pruned.sh configs/pruned_eval_flowers102.yaml --output-json pruned/<EXPERIMENT>/eval_metrics.json"
    exit 1
fi

CONFIG_PATH=$1
shift 1

python eval_pruned.py --config "$CONFIG_PATH" "$@"
