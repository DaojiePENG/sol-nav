#!/bin/bash
#SBATCH --job-name=solnav_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4090:1
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodelist=4090node2

# -------------------------------------------------------
# SOL-Nav: 单卡 RTX 4090 训练（调试 / 快速验证）
# 提交: sbatch scripts/slurm_train_1x4090.sh
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
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

mkdir -p logs

echo "=== SLURM_JOB_ID=$SLURM_JOB_ID  NODE=$(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="

PYTHON="/mnt/slurmfs-4090node1/homes/dpeng108/miniforge3/envs/env_transformer_eval/bin/python"

# 单卡不需要 torchrun，直接 python 即可
$PYTHON train.py --config configs/default.yaml
