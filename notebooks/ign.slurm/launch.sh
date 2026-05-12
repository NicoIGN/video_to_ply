#!/bin/bash
set -e

source config.sh


export MPLCONFIGDIR=$HOME_SLURM/.config/matplotlib
mkdir -p $MPLCONFIGDIR

cd $HOME_SLURM/video_to_ply

echo "===================================="
echo "🌐 SHELL PROXY ENV"
echo "HTTP_PROXY  = $HTTP_PROXY"
echo "HTTPS_PROXY = $HTTPS_PROXY"
echo "http_proxy  = $http_proxy"
echo "https_proxy = $https_proxy"
echo "NO_PROXY    = $NO_PROXY"
echo "===================================="

python - << 'EOF'
import os

print("====================================")
print("🐍 PYTHON ENV PROXY VIEW")
print("HTTP_PROXY  =", os.getenv("HTTP_PROXY"))
print("HTTPS_PROXY =", os.getenv("HTTPS_PROXY"))
print("http_proxy  =", os.getenv("http_proxy"))
print("https_proxy =", os.getenv("https_proxy"))
print("NO_PROXY    =", os.getenv("NO_PROXY"))
print("====================================")
EOF


export HTTP_PROXY
export HTTPS_PROXY
export http_proxy
export https_proxy
export NO_PROXY

srun \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=16G \
  --time=04:00:00 \
  bash -lc "
    export HTTP_PROXY=$HTTP_PROXY
    export HTTPS_PROXY=$HTTPS_PROXY
    export http_proxy=$http_proxy
    export https_proxy=$https_proxy
    export NO_PROXY=$NO_PROXY

    bash run.sh \
      --root \"$ROOTDIR\" \
      --name \"$BASENAME\" \
      --video \"$VIDEOSOURCE\" \
      --skip-conda \
      --preprocess-profile \"$PREPROCESS_PROFILE\" \
      --gsplat-profile \"$GSPLAT_PROFILE\" \
      --num-frames \"$NUM_FRAMES\"
  "
