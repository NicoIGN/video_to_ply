#!/bin/bash
set -e

# =========================
# CONFIG
# =========================
WORK_DIR=/content/work
VENV_DIR=$WORK_DIR/venv
BIN_DIR=$WORK_DIR/bin

rm -rf $WORK_DIR
mkdir -p $BIN_DIR

# =========================
# SYSTEM
# =========================
apt-get update -y

apt-get install -y \
  python3.12 python3.12-venv python3.12-dev \
  colmap ffmpeg cmake ninja-build \
  libgl1-mesa-glx xvfb \
  libeigen3-dev libsuitesparse-dev \
  libglew-dev qtbase5-dev libqt5opengl5-dev

echo "SYSTEM PYTHON:"
python3.12 --version

# =========================
# VENV CLEAN (NO ENSUREPIP BUG)
# =========================
python3.12 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# FIX pip (ensurepip broken in colab sometimes)
curl -sS https://bootstrap.pypa.io/get-pip.py | python

pip install --upgrade pip setuptools wheel

# =========================
# CORE STACK (PYTHON 3.12 SAFE)
# =========================
pip install numpy==2.1.2 scipy

# =========================
# VISION
# =========================
pip install imageio imageio-ffmpeg opencv-python
# ⚠️ openimageio retiré (force numpy 2.x instable avec torch)

# =========================
# BUILD
# =========================
pip install pybind11 ninja

# =========================
# PYTORCH (CUDA COLAB)
# =========================
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

# =========================
# NERFSTUDIO (LATEST COMPAT 3.12)
# =========================
pip install nerfstudio

# pycolmap optionnel (pas critique)
pip install pycolmap || true

# =========================
# ENV
# =========================
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
export OPENCV_LOG_LEVEL=ERROR
export XDG_RUNTIME_DIR=/tmp/runtime-root
export LIBGL_ALWAYS_SOFTWARE=1

# =========================
# WRAPPER
# =========================
cat > $BIN_DIR/ns-train << 'EOF'
#!/usr/bin/env python
from nerfstudio.scripts.train import entrypoint
import sys
sys.exit(entrypoint())
EOF

chmod +x $BIN_DIR/ns-train
export PATH="$BIN_DIR:$PATH"

# =========================
# 🔍 HARD VALIDATION (CRASH PREVENTION)
# =========================
echo "==== VALIDATION ===="

python - << 'EOF'
import sys

errors = []

try:
    import torch
    if not torch.cuda.is_available():
        errors.append("CUDA NOT AVAILABLE")
except:
    errors.append("TORCH IMPORT FAIL")

try:
    import numpy
    if int(numpy.__version__.split('.')[0]) < 2:
        errors.append("NUMPY < 2 (INCOMPATIBLE PYTHON 3.12 STACK)")
except:
    errors.append("NUMPY IMPORT FAIL")

try:
    import nerfstudio
except:
    errors.append("NERFSTUDIO IMPORT FAIL")

if errors:
    print("\n❌ ENVIRONMENT INVALID:")
    for e in errors:
        print(" -", e)
    sys.exit(1)

print("✅ ENVIRONMENT OK")
EOF

# =========================
# FINAL CHECK
# =========================
echo "PYTHON:"
python --version

echo "TORCH:"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

echo "NERFSTUDIO:"
python -c "import nerfstudio; print('OK')"

echo "NS-TRAIN:"
ns-train --help || (echo "❌ ns-train broken" && exit 1)
