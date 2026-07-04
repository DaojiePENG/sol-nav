#!/bin/bash
#SBATCH --job-name=solnav_8gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4090:4
#SBATCH --output=logs/train_8x4090_%j.out
#SBATCH --error=logs/train_8x4090_%j.err
#SBATCH --nodelist=4090node[2-3]

# -------------------------------------------------------
# SOL-Nav: 跨节点 8×RTX 4090 DDP 训练（2 节点 × 4 卡）
# 提交: sbatch scripts/slurm_train_8x4090_multinode.sh
# -------------------------------------------------------

set -euo pipefail
cd /mnt/slurmfs-4090node1/homes/dpeng108/sol-nav

# ── Conda ──
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate env_transformer_eval

# ── NCCL: RTX 4090 不支持 P2P / IB ──
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

mkdir -p logs

# ── 多节点环境变量（SLURM 自动设置）──
# SLURM_NODELIST   → 所有节点列表
# SLURM_NODEID     → 当前节点编号
# SLURM_NNODES     → 节点总数
# SLURM_PROCID     → 全局进程编号

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=29500
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_NODEID  # 当前节点在集群中的 rank（0 或 1）

echo "=== SLURM_JOB_ID=$SLURM_JOB_ID  NODE=$(hostname) NODE_RANK=$NODE_RANK ==="
echo "=== MASTER=$MASTER_ADDR:$MASTER_PORT  NNODES=$NNODES ==="
echo "=== GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -4) ==="

PYTHON="/mnt/slurmfs-4090node1/homes/dpeng108/miniforge3/envs/env_transformer_eval/bin/python"

# --nnodes=2        → 总共 2 个节点
# --nproc_per_node=4 → 每节点 4 个进程（各绑一张 GPU）
# --node_rank       → 当前节点编号（SLURM 自动分配）
# --master_addr     → rank 0 节点的地址
$PYTHON -m torch.distributed.run \
    --nnodes=$NNODES \
    --nproc_per_node=4 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py --config configs/default.yaml
