#!/bin/bash
# scripts/run.sh

# 1. 사용법 안내 (설정파일이 없으면 종료)
if [ -z "$1" ]; then
    echo "사용법: bash scripts/run.sh <CONFIG_PATH> [추가 인자]"
    echo "예시: bash scripts/run.sh configs/pretrained_vit_pruning.yaml --lr 0.001"
    exit 1
fi

# 2. GPU 번호 고정
GPU_ID=7

# 3. 인자 할당
CONFIG_PATH=$1   # 첫 번째 인자: 읽어올 YAML 파일 경로
shift 1          # 앞의 한 개를 치우고 나머지($@)에 추가 인자들을 남김

# 4. 실행
# CUDA_VISIBLE_DEVICES로 사용할 GPU를 강제 지정합니다.
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config "$CONFIG_PATH" "$@"
