#!/bin/bash
# scripts/run.sh

# 1. 사용법 안내 (GPU 번호와 설정파일이 없으면 종료)
if [ -z "$2" ]; then
    echo "사용법: bash scripts/run.sh <GPU_ID> <CONFIG_PATH> [추가 인자]"
    echo "예시: bash scripts/run.sh 0 configs/dinov2_lora.yaml --lr 0.001"
    exit 1
fi

# 2. 인자 할당
GPU_ID=$1       # 첫 번째 인자: 사용할 GPU 번호
CONFIG_PATH=$2   # 두 번째 인자: 읽어올 YAML 파일 경로
shift 2          # 앞의 두 개를 치우고 나머지($@)에 추가 인자들을 남김

# 3. 실행
# CUDA_VISIBLE_DEVICES로 사용할 GPU를 강제 지정합니다.
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config "$CONFIG_PATH" "$@"