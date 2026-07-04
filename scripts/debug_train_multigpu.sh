#!/usr/bin/env bash
# -------------------------------------------------------
# SOL-Nav: 本地多卡调试（无需 sbatch，验证多 GPU DDP 是否正常）
# 用法:
#   ./scripts/debug_train_multigpu.sh                # 默认用全部 4 块 GPU
#   ./scripts/debug_train_multigpu.sh 0,1            # 指定 GPU 0 和 1
#   ./scripts/debug_train_multigpu.sh 0,1,2          # 指定 GPU 0, 1, 2
# -------------------------------------------------------

set -euo pipefail
cd /mnt/slurmfs-4090node1/homes/dpeng108/sol-nav

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate env_transformer_eval

# ── 参数解析 ──
GPU_IDS="${1:-0,1,2,3}"

# 校验 GPU_ID 合法性：每个 ID 必须是 0-3 的数字，逗号分隔
IFS=',' read -ra GPU_ARR <<< "$GPU_IDS"
for gid in "${GPU_ARR[@]}"; do
    if ! [[ "$gid" =~ ^[0-3]$ ]]; then
        echo "ERROR: GPU_ID=$gid 不合法，4090node 可用 GPU index 为 0-3"
        echo "用法: ./scripts/debug_train_multigpu.sh [0,1 | 0,1,2 | 0,1,2,3]"
        exit 1
    fi
done

NUM_GPUS=${#GPU_ARR[@]}

# ── NCCL: RTX 4090 不支持 P2P / IB ──
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

echo "=== DEBUG: GPUs=$GPU_IDS  NUM_GPUS=$NUM_GPUS  NODE=$(hostname) ==="

PYTHON="/mnt/slurmfs-4090node1/homes/dpeng108/miniforge3/envs/env_transformer_eval/bin/python"

if [ "$NUM_GPUS" -eq 1 ]; then
    # 单卡直接 python，无需 torchrun
    echo "=== 单卡模式，跳过 torchrun ==="
    $PYTHON train.py --config configs/default.yaml
else
    # 多卡：torchrun 启动 DDP
    # --nproc_per_node = 每个 GPU 一个进程
    # --master_port    随机选 29500-29599 避免端口冲突
    MASTER_PORT=$((29500 + RANDOM % 100))
    echo "=== 多卡 DDP 模式: torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT ==="
    $PYTHON -m torch.distributed.run \
        --nproc_per_node="$NUM_GPUS" \
        --master_port="$MASTER_PORT" \
        train.py --config configs/default.yaml
fi
