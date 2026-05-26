#!/bin/bash

set -e

srun \
  --gres=gpu:1 \
  --cpus-per-task=1 \
  --mem=2G \
  --time=00:15:00 \
  bash -lc '

set -e

echo "========================"
echo "🧪 SLURM ENV CHECK"
echo "========================"

# -------------------------
# CONDA AUTO DETECT
# -------------------------
CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
if [ -z "$CONDA_BASE" ]; then
    echo "❌ conda not found"
    exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate gsplat

echo ""
echo "📍 Host:"
hostname

echo ""
echo "📍 GPU (nvidia-smi):"
nvidia-smi || echo "❌ nvidia-smi failed"

echo ""
echo "📍 CUDA nvcc:"
which nvcc || echo "❌ nvcc not found"
nvcc --version || echo "❌ nvcc version failed"

echo ""
echo "📍 CUDA_VISIBLE_DEVICES:"
echo "$CUDA_VISIBLE_DEVICES"

echo ""
echo "📍 Python:"
which python
python --version

echo ""
echo "📍 PyTorch CUDA:"
python -c "import torch; print(\"torch version:\", torch.__version__); print(\"cuda available:\", torch.cuda.is_available())"

echo ""
echo "📍 Torch GPU info:"
python - <<EOF
import torch
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
EOF

echo ""
echo "📍 gsplat import:"
python -c "import gsplat; print(\"gsplat OK\")" || echo "❌ gsplat failed"

echo ""
echo "📍 COLMAP:"
which colmap || echo "❌ colmap not found"
colmap --version || echo "❌ colmap version failed"

echo ""
echo "📍 OpenGL check:"
python -c "from OpenGL import GL" 2>/dev/null && echo "OpenGL OK" || echo "❌ OpenGL missing"

echo ""
echo "📍 Environment summary:"
env | grep -E "CUDA|NVIDIA|SLURM|CONDA|PATH" | head -50

echo ""
echo "========================"
echo "DONE"
echo "========================"
'
