#!/bin/bash

set -e

srun \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=8G \
  --time=00:15:00 \
  bash -lc "

echo '========================'
echo '🧪 SLURM ENV CHECK'
echo '========================'

echo ''
echo '📍 Host:'
hostname

echo ''
echo '📍 GPU (nvidia-smi):'
nvidia-smi || echo '❌ nvidia-smi failed'

echo ''
echo '📍 CUDA nvcc:'
which nvcc || echo '❌ nvcc not found'
nvcc --version || echo '❌ nvcc version failed'

echo ''
echo '📍 CUDA_VISIBLE_DEVICES:'
echo \$CUDA_VISIBLE_DEVICES

echo ''
echo '📍 Python:'
which python
python --version

echo ''
echo '📍 PyTorch CUDA:'
python -c 'import torch; print(\"torch version:\", torch.__version__); print(\"cuda available:\", torch.cuda.is_available())' || echo '❌ torch failed'

echo ''
echo '📍 Torch CUDA device:'
python -c 'import torch; 
print(torch.cuda.device_count());
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))' || true

echo ''
echo '📍 gsplat import:'
python -c 'import gsplat' && echo '✅ gsplat OK' || echo '❌ gsplat failed (CUDA toolkit missing probable)'

echo ''
echo '📍 COLMAP:'
which colmap || echo '❌ colmap not found'
colmap --version || echo '❌ colmap version failed'

echo ''
echo '📍 OpenGL (headless check):'
python -c 'from OpenGL import GL' 2>/dev/null && echo 'OpenGL import OK' || echo '❌ OpenGL missing'

echo ''
echo '📍 Environment summary:'
env | grep -E 'CUDA|NVIDIA|SLURM|CONDA|PATH' | head -50

echo ''
echo '========================'
echo 'DONE'
echo '========================'
"
