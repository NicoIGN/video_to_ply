#!/bin/bash
set -e

source config.sh


export MPLCONFIGDIR=$HOME_SLURM/.config/matplotlib
mkdir -p $MPLCONFIGDIR

cd $HOME_SLURM/video_to_ply

export HTTP_PROXY
export HTTPS_PROXY
export http_proxy
export https_proxy
export NO_PROXY
export MAX_JOBS

# RUN SLURM
# -----------------------------
srun \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=16G \
  --time=04:00:00 \
  bash -lc '

set -e

export HTTP_PROXY=$HTTP_PROXY
export HTTPS_PROXY=$HTTPS_PROXY
export http_proxy=$http_proxy
export https_proxy=$https_proxy
export NO_PROXY=$NO_PROXY
export MAX_JOBS=$MAX_JOBS

# -----------------------------
# CONDA AUTO INIT (GSPLAT ENV)
# -----------------------------
CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
if [ -z "$CONDA_BASE" ]; then
    echo "❌ conda not found"
    exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate gsplat

echo "========================"
echo "🚀 GSPLAT RUN ENV"
echo "========================"

echo "python: $(which python)"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

echo MAX_JOBS - 1: $MAX_JOBS
bash run.sh \
  --root "'"$ROOTDIR"'" \
  --name "'"$BASENAME"'" \
  --video "'"$VIDEOSOURCE"'" \
  --skip-conda \
  --preprocess-profile "'"$PREPROCESS_PROFILE"'" \
  --gsplat-profile "'"$GSPLAT_PROFILE"'" \
  --num-frames "'"$NUM_FRAMES"'"

'
