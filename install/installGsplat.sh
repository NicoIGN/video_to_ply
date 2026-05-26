#!/usr/bin/env bash
set -euo pipefail

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

if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
    echo "❌ Error: not in conda env '$ENV_NAME'"
    echo "👉 Current env: ${CONDA_DEFAULT_ENV:-none}"
    echo "👉 Run: conda activate $ENV_NAME"
    exit 1
fi

echo "✅ Environment OK: $ENV_NAME"

########################################
# CHECK TORCH
########################################
echo "🔍 Checking PyTorch..."

if ! python -c "import torch" &>/dev/null; then
    echo "❌ PyTorch not found"
    exit 1
fi

TORCH_VER=$(python -c "import torch; print(torch.__version__)")
CUDA_VER=$(python -c "import torch; print(torch.version.cuda)")

echo "✅ Torch: $TORCH_VER"
echo "✅ Torch CUDA: $CUDA_VER"

########################################
# CHECK NVCC
########################################
echo "🔍 Checking nvcc..."

if ! command -v nvcc &>/dev/null; then
    echo "❌ nvcc not found"
    exit 1
fi

NVCC_VER=$(nvcc --version | tail -n 1)
echo "✅ $NVCC_VER"

########################################
# CUDA SANITY CHECK
########################################
if [[ "$CUDA_VER" != "12.1" ]]; then
    echo "⚠️ Warning: PyTorch CUDA version is not 12.1 (got $CUDA_VER)"
    echo "👉 This may still work but is risky for gsplat"
fi

########################################
# CUDA ENV FIX (CRITICAL)
########################################
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"

########################################
# CUDA ARCH CONFIG
########################################
export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH"
export MAX_JOBS=10

echo "🚀 CUDA_HOME=$CUDA_HOME"
echo "🚀 TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

########################################
# CLEAN OLD BUILD CACHE (IMPORTANT)
########################################
echo "🧹 Cleaning torch extension cache..."
rm -rf ~/.cache/torch_extensions || true

########################################
# INSTALL GSPLAT
########################################
echo "📦 Installing gsplat $GSPLAT_VERSION..."

TORCH_CUDA_ARCH_LIST="$CUDA_ARCH" \
pip install --no-build-isolation --no-cache-dir \
git+$REPO@$GSPLAT_VERSION

########################################
# DONE
########################################
echo "🎉 gsplat installation complete!"
