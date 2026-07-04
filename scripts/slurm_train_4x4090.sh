#!/bin/bash
#SBATCH --job-name=solnav_ddp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4090:4
#SBATCH --output=logs/train_4x4090_%j.out
#SBATCH --error=logs/train_4x4090_%j.err
#SBATCH --nodelist=4090node2

# -------------------------------------------------------
# SOL-Nav: 单节点 4×RTX 4090 DDP 训练
# 提交: sbatch scripts/slurm_train_4x4090.sh
# -------------------------------------------------------

set -euo pipefail
cd /mnt/slurmfs-4090node1/homes/dpeng108/sol-nav

# ── Conda ──
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate env_transformer_eval

# ── NCCL: RTX 4090 不支持 P2P / IB，必须禁用 ──
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

mkdir -p logs

echo "=== SLURM_JOB_ID=$SLURM_JOB_ID  NODE=$(hostname) ==="
echo "=== GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -4) ==="

PYTHON="/mnt/slurmfs-4090node1/homes/dpeng108/miniforge3/envs/env_transformer_eval/bin/python"

# --nproc_per_node=4 → 在本节点启动 4 个进程，各绑一张卡，走 DDP
$PYTHON -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29500 \
    train.py --config configs/default.yaml
