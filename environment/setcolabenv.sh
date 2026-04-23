set -e

# =========================
# CONFIG
# =========================
WORK_DIR=/content/work
BIN_DIR=$WORK_DIR/bin
VENV_DIR=/content/venv

rm -rf $WORK_DIR $VENV_DIR
mkdir -p $BIN_DIR

# =========================
# SYSTEM DEPENDENCIES
# =========================
apt-get update -y

apt-get install -y \
  colmap ffmpeg cmake ninja-build \
  libgl1-mesa-glx xvfb \
  libeigen3-dev \
  libsuitesparse-dev \
  libglew-dev \
  qtbase5-dev \
  libqt5opengl5-dev

# =========================
# PYTHON INFO
# =========================
echo "SYSTEM PYTHON:"
python3 --version

# =========================
# VENV (CRITICAL FIX)
# =========================
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# =========================
# PIP BASE
# =========================
pip install --upgrade pip setuptools wheel

# =========================
# CORE STACK (LOCKED)
# =========================
pip install numpy==1.26.4 scipy

# ⚠️ PAS openimageio (casse numpy)

pip install imageio imageio-ffmpeg opencv-python-headless

# =========================
# BUILD TOOLS
# =========================
pip install pybind11

# =========================
# PYTORCH
# =========================
pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu121

# =========================
# NERFSTUDIO
# =========================
pip install nerfstudio pycolmap

# =========================
# ENV FLAGS
# =========================
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
export OPENCV_LOG_LEVEL=ERROR
export XDG_RUNTIME_DIR=/tmp/runtime-root
export LIBGL_ALWAYS_SOFTWARE=1
export CUDA_VISIBLE_DEVICES=0

# =========================
# WRAPPER
# =========================
cat > $BIN_DIR/ns-train << 'EOF'
#!/usr/bin/env python3
from nerfstudio.scripts.train import entrypoint
import sys

sys.exit(entrypoint())
EOF

chmod +x $BIN_DIR/ns-train
export PATH="$BIN_DIR:$PATH"

# =========================
# VALIDATION (FAIL FAST)
# =========================
echo "=== VALIDATION ==="

python3 << 'EOF'
import sys

def fail(msg):
    print(f"\n❌ ENV ERROR: {msg}\n")
    sys.exit(1)

# --- Python ---
import sys as _sys
print("Python:", _sys.version)

# --- Torch ---
try:
    import torch
    print("Torch:", torch.__version__)
    if not torch.cuda.is_available():
        fail("CUDA not available (ns-train will be extremely slow or crash)")
except Exception as e:
    fail(f"Torch import failed: {e}")

# --- Numpy ---
try:
    import numpy as np
    print("Numpy:", np.__version__)
    major = int(np.__version__.split('.')[0])
    if major >= 2:
        fail("Numpy >=2 detected (incompatible with nerfstudio stack)")
except Exception as e:
    fail(f"Numpy import failed: {e}")

# --- Nerfstudio ---
try:
    import nerfstudio
    print("Nerfstudio OK")
except Exception as e:
    fail(f"Nerfstudio import failed: {e}")

# --- Entrypoint ---
try:
    from nerfstudio.scripts.train import entrypoint
    print("Entrypoint OK")
except Exception as e:
    fail(f"ns-train entrypoint broken: {e}")

print("\n✅ ENVIRONMENT OK\n")
EOF

# =========================
# FINAL CHECK
# =========================
echo "NS-TRAIN CHECK:"
ns-train --help > /dev/null 2>&1 || { echo "❌ ns-train broken"; exit 1; }

echo "🎉 READY"
