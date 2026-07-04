#!/usr/bin/env bash
# -------------------------------------------------------
# SOL-Nav: 本地单卡调试（无需 sbatch，直接 ./scripts/debug_train_1gpu.sh）
# 用法:
#   ./scripts/debug_train_1gpu.sh              # 默认 GPU 0
#   ./scripts/debug_train_1gpu.sh 2            # 指定 GPU 2
# -------------------------------------------------------

set -euo pipefail
cd /mnt/slurmfs-4090node1/homes/dpeng108/sol-nav

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate env_transformer_eval

GPU_ID="${1:-0}"

# 校验 GPU_ID 合法性（4090node1/2 各有 4 块 GPU，index 0-3）
if ! [[ "$GPU_ID" =~ ^[0-3]$ ]]; then
    echo "ERROR: GPU_ID=$GPU_ID 不合法，4090node 可用 GPU index 为 0-3"
    echo "用法: ./scripts/debug_train_1gpu.sh [0|1|2|3]"
    exit 1
fi

# ── NCCL: RTX 4090 不支持 P2P / IB ──
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

mkdir -p logs

echo "=== DEBUG: GPU=$GPU_ID  NODE=$(hostname) ==="

PYTHON="/mnt/slurmfs-4090node1/homes/dpeng108/miniforge3/envs/env_transformer_eval/bin/python"

# 单卡直接 python，无需 torchrun
$PYTHON train.py --config configs/default.yaml
