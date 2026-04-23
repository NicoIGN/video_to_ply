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
  libglew-dev qtbase5-dev libqt5opengl5-dev \
  curl

echo "SYSTEM PYTHON:"
python3.12 --version

# =========================
# VENV CLEAN
# =========================
python3.12 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# FIX pip
curl -sS https://bootstrap.pypa.io/get-pip.py | python

pip install --upgrade pip setuptools wheel

# =========================
# 🔒 FORCE NUMPY 2 (LOCK)
# =========================
pip install numpy==2.1.2
pip install "numpy>=2,<3" --no-deps --force-reinstall

# =========================
# PYTORCH FIRST (IMPORTANT)
# =========================
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

# =========================
# CORE SCIENTIFIC
# =========================
pip install scipy

# =========================
# VISION (aligned numpy 2)
# =========================
pip install opencv-python imageio imageio-ffmpeg

# =========================
# BUILD
# =========================
pip install pybind11 ninja

# =========================
# NERFSTUDIO
# =========================
pip install nerfstudio

# OPTIONAL
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
# 🔍 HARD VALIDATION (ANTI-FREEZE)
# =========================
echo "==== VALIDATION ===="

python - << 'EOF'
import sys

errors = []

# NUMPY ABI
try:
    import numpy
    if int(numpy.__version__.split('.')[0]) < 2:
        errors.append("NUMPY < 2")
except Exception as e:
    errors.append(f"NUMPY FAIL: {e}")

# TORCH
try:
    import torch
    if not torch.cuda.is_available():
        print("⚠️ WARNING: CUDA not available (will be slow)")
except Exception as e:
    errors.append(f"TORCH FAIL: {e}")

# OPENCV ABI
try:
    import cv2
except Exception as e:
    errors.append(f"OPENCV FAIL (likely numpy ABI): {e}")

# NERFSTUDIO
try:
    import nerfstudio
except Exception as e:
    errors.append(f"NERFSTUDIO FAIL: {e}")

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

echo "NUMPY:"
python -c "import numpy; print(numpy.__version__)"

echo "TORCH:"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

echo "OPENCV:"
python -c "import cv2; print(cv2.__version__)"

echo "NERFSTUDIO:"
python -c "import nerfstudio; print('OK')"

echo "NS-TRAIN:"
ns-train --help || (echo "❌ ns-train broken" && exit 1)
