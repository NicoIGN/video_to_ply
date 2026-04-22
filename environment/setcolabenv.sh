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
  python3.10 python3.10-dev python3.10-venv \
  libeigen3-dev \
  libsuitesparse-dev \
  libglew-dev \
  qtbase5-dev \
  libqt5opengl5-dev

# =========================
# FORCE PYTHON 3.10 SYSTEM LEVEL
# =========================
update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
update-alternatives --set python /usr/bin/python3.10

hash -r

echo "SYSTEM PYTHON:"
python --version

# =========================
# CLEAN COLAB PYTHON CONFLICTS
# =========================
rm -rf /usr/local/lib/python3.12/dist-packages/nerfstudio* || true
rm -rf /usr/local/lib/python3.12/dist-packages/torch* || true
rm -f /usr/local/bin/ns-train || true

# =========================
# FORCE PYTHON 3.10 INTO BIN_DIR (IMPORTANT)
# =========================
ln -sf /usr/bin/python3.10 $BIN_DIR/python
ln -sf /usr/bin/python3.10 $BIN_DIR/python3

# 🔥 PRIORITY PATH (BIN_DIR FIRST)
export PATH="$BIN_DIR:$PATH"
hash -r

echo "SHIM PYTHON:"
which python
python --version

# =========================
# ENSURE PIP (SAFE ON COLAB)
# =========================
python -m ensurepip --upgrade || true
python -m pip install --upgrade pip setuptools wheel

# =========================
# NUMPY / SCIENTIFIC STACK
# =========================
python -m pip install numpy==1.26.4
python -m pip install scipy

# =========================
# VISION STACK
# =========================
python -m pip install openimageio
python -m pip install imageio imageio-ffmpeg opencv-python

# =========================
# BUILD SUPPORT
# =========================
python -m pip install pybind11

# =========================
# PYTORCH (CUDA 11.8 stable Colab)
# =========================
python -m pip install torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu118

# =========================
# NERF STACK
# =========================
python -m pip install nerfstudio==0.3.4
python -m pip install pycolmap==0.6.1

# =========================
# OPTIONAL
# =========================
python -m pip install rclone

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
# ns-train WRAPPER (SAFE)
# =========================
cat > $BIN_DIR/ns-train << 'EOF'
#!/usr/bin/env python3.10
from nerfstudio.scripts.train import entrypoint
import sys

sys.exit(entrypoint())
EOF

chmod +x $BIN_DIR/ns-train

# =========================
# FINAL CHECK
# =========================
hash -r

echo "FINAL PYTHON:"
python --version
which python

echo "IMPORT TEST:"
python -c "import nerfstudio; print('NERFSTUDIO OK')"

echo "NS-TRAIN:"
ns-train --help && echo "OK" || echo "FAILED"
