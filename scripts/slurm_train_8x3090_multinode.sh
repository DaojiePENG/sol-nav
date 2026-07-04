#!/bin/bash
#SBATCH --job-name=solnav_3090
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:3090:4
#SBATCH --output=logs/train_8x3090_%j.out
#SBATCH --error=logs/train_8x3090_%j.err
#SBATCH --nodelist=3090node[1,3]

# -------------------------------------------------------
# SOL-Nav: 跨节点 8×RTX 3090 DDP 训练（2 节点 × 4 卡）
# 提交: sbatch scripts/slurm_train_8x3090_multinode.sh
# -------------------------------------------------------
# check GPU 状态: nvidia-smi -L
# scontrol show node 3090node2 2>/dev/null | grep -E "NodeName|Gres|State|CPUTot"
# -------------------------------------------------------

set -euo pipefail
cd /mnt/slurmfs-4090node1/homes/dpeng108/sol-nav

# ── Conda ──
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate env_transformer_eval

# ── NCCL: RTX 3090 同样不支持 P2P / IB ──
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

# ── 多节点环境变量（SLURM 自动设置）──
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=29500
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_NODEID

echo "=== SLURM_JOB_ID=$SLURM_JOB_ID  NODE=$(hostname) NODE_RANK=$NODE_RANK ==="
echo "=== MASTER=$MASTER_ADDR:$MASTER_PORT  NNODES=$NNODES ==="
echo "=== GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -4) ==="

PYTHON="/mnt/slurmfs-4090node1/homes/dpeng108/miniforge3/envs/env_transformer_eval/bin/python"

$PYTHON -m torch.distributed.run \
    --nnodes=$NNODES \
    --nproc_per_node=4 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py --config configs/default.yaml
