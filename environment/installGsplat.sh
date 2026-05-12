#!/usr/bin/env bash
set -e

########################################
# CONFIG
########################################
ENV_NAME="gsplat"
GSPLAT_VERSION="v1.4.0"
CUDA_ARCH="7.5"
REPO="https://github.com/nerfstudio-project/gsplat.git"

########################################
# CHECK CONDA ENV
########################################
echo "🔍 Checking conda environment..."

if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    echo "❌ Error: you are not in conda env '$ENV_NAME'"
    echo "👉 Current env: $CONDA_DEFAULT_ENV"
    echo "👉 Run: conda activate $ENV_NAME"
    exit 1
fi

echo "✅ Correct environment: $ENV_NAME"

########################################
# CHECK TORCH
########################################
echo "🔍 Checking torch..."

python -c "import torch" 2>/dev/null || {
    echo "❌ PyTorch not found in env"
    exit 1
}

echo "✅ PyTorch found: $(python -c 'import torch; print(torch.__version__)')"

########################################
# CHECK CUDA
########################################
echo "🔍 Checking CUDA..."

if ! command -v nvcc &> /dev/null; then
    echo "❌ nvcc not found"
    exit 1
fi

echo "✅ nvcc version:"
nvcc --version | tail -n 1

########################################
# SET CUDA ARCH
########################################
export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH"
export MAX_JOBS=10

echo "🚀 TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

########################################
# INSTALL GSPLAT
########################################
echo "📦 Installing gsplat $GSPLAT_VERSION..."

pip install --no-build-isolation --no-cache-dir \
git+$REPO@$GSPLAT_VERSION

########################################
# DONE
########################################
echo "🎉 gsplat installation complete!"
