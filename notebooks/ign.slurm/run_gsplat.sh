#!/bin/bash
#SBATCH --job-name=gsplat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=jean-zellou

#SBATCH -o /mnt/common/hdd/slurm/logs/gsplat-%j.out
#SBATCH -e /mnt/common/hdd/slurm/logs/gsplat-%j.err

set -euo pipefail
set -x

# =========================
# LOAD USER CONFIG
# =========================
source config.sh

# =========================
# SRUN / SLURM SETTINGS
# =========================
export SRUN_CPUS_PER_TASK=12

# Optional distributed debug
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# =========================
# PROXY / ENV
# =========================
export HTTP_PROXY
export HTTPS_PROXY
export http_proxy
export https_proxy
export NO_PROXY

export MAX_JOBS

# =========================
# MATPLOTLIB CACHE
# =========================
export MPLCONFIGDIR="$HOME_SLURM/.config/matplotlib"
mkdir -p "$MPLCONFIGDIR"

# =========================
# PROJECT
# =========================
cd "$HOME_SLURM/video_to_ply"

# =========================
# CONDA ENV
# =========================
eval "$(conda shell.bash hook)"
conda activate gsplat

# =========================
# DEBUG INFO
# =========================
echo "========================"
echo "🚀 GSPLAT RUN ENV"
echo "========================"

echo "python: $(which python)"

python -c "
import torch
print(torch.__version__)
print(torch.cuda.is_available())
"

nvidia-smi

# =========================
# RUN TRAINING
# =========================
srun -vv bash run.sh \
  --root "$ROOTDIR" \
  --name "$BASENAME" \
  --video "$VIDEOSOURCE" \
  --skip-conda \
  --preprocess-profile "$PREPROCESS_PROFILE" \
  --gsplat-profile "$GSPLAT_PROFILE" \
  --num-frames "$NUM_FRAMES"

echo "✅ Job complete"
