#!/bin/bash
set -e

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
# VENV CLEAN
# =========================
python3.12 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# FIX pip (robuste colab)
curl -sS https://bootstrap.pypa.io/get-pip.py | python

pip install --upgrade pip setuptools wheel

# =========================
# CORE (laisser pip gérer numpy)
# =========================
pip install scipy

# =========================
# IMAGE STACK MINIMAL
# =========================
pip install imageio imageio-ffmpeg

# =========================
# BUILD
# =========================
pip install pybind11 ninja

# =========================
# PYTORCH
# =========================
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

# =========================
# NERFSTUDIO
# =========================
pip install nerfstudio

pip install pycolmap || true

# =========================
# ENV
# =========================
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
export XDG_RUNTIME_DIR=/tmp/runtime-root

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
# 🔥 VALIDATION ANTI-CRASH
# =========================
echo "==== VALIDATION ===="

python - << 'EOF'
import sys

errors = []

# TORCH
try:
    import torch
    if not torch.cuda.is_available():
        errors.append("CUDA NOT AVAILABLE")
except:
    errors.append("TORCH IMPORT FAIL")

# NUMPY
try:
    import numpy as np
    print("NUMPY:", np.__version__)
except:
    errors.append("NUMPY IMPORT FAIL")

# NERFSTUDIO
try:
    import nerfstudio
except:
    errors.append("NERFSTUDIO IMPORT FAIL")

# ⚠️ combo dangereux connu
try:
    import numpy as np
    if np.__version__.startswith("2"):
        print("⚠️ WARNING: numpy 2 detected → nerfstudio may crash later")
except:
    pass

if errors:
    print("\n❌ ENV INVALID:")
    for e in errors:
        print(" -", e)
    sys.exit(1)

print("✅ ENV OK")
EOF

# =========================
# FINAL CHECK
# =========================
python -c "import torch; print('TORCH:', torch.__version__, torch.cuda.is_available())"
python -c "import nerfstudio; print('NERFSTUDIO OK')"

ns-train --help || (echo "❌ ns-train broken" && exit 1)
