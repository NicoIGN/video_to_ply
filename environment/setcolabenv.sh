# =========================
# CONFIG
# =========================
WORK_DIR=/content/work
BIN_DIR=$WORK_DIR/bin

rm -rf $WORK_DIR
mkdir -p $BIN_DIR

# =========================
# SYSTEM DEPENDENCIES
# =========================
apt-get update -y

apt-get install -y \
  software-properties-common \
  colmap ffmpeg cmake ninja-build \
  libgl1-mesa-glx xvfb \
  libeigen3-dev \
  libsuitesparse-dev \
  libglew-dev \
  qtbase5-dev \
  libqt5opengl5-dev

# =========================
# USE SYSTEM PYTHON 3.12 (NO OVERRIDE)
# =========================
echo "SYSTEM PYTHON:"
python3 --version

# =========================
# CLEAN OLD CONFLICTS
# =========================
rm -rf /usr/local/lib/python3.12/dist-packages/nerfstudio* || true
rm -rf /usr/local/lib/python3.12/dist-packages/torch* || true
rm -f /usr/local/bin/ns-train || true

# =========================
# PIP BASE
# =========================
python3 -m ensurepip --upgrade || true
python3 -m pip install --upgrade pip setuptools wheel

# =========================
# CORE NUMPY STACK (IMPORTANT FOR COMPATIBILITY)
# =========================
python3 -m pip install numpy==1.26.4 scipy

# =========================
# VISION STACK
# =========================
python3 -m pip install openimageio imageio imageio-ffmpeg opencv-python

# =========================
# BUILD TOOLS
# =========================
python3 -m pip install pybind11

# =========================
# PYTORCH (CUDA COMPATIBLE WITH MODERN NERFSTUDIO)
# =========================
python3 -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

# =========================
# NERFSTUDIO (MODERN VERSION FOR PYTHON 3.12)
# =========================
python3 -m pip install nerfstudio

# =========================
# OPTIONAL
# =========================
python3 -m pip install pycolmap rclone

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
# FINAL CHECK
# =========================
echo "PYTHON:"
python3 --version

echo "TORCH:"
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"

echo "NERFSTUDIO:"
python3 -c "import nerfstudio; print('OK', nerfstudio.__file__)"

echo "NS-TRAIN:"
ns-train --help || echo "FAILED"
