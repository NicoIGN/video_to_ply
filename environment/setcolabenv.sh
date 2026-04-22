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
  python3-pip \
  libeigen3-dev \
  libsuitesparse-dev \
  libglew-dev \
  qtbase5-dev \
  libqt5opengl5-dev

# =========================
# FORCE PYTHON 3.10
# =========================
update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
update-alternatives --set python /usr/bin/python3.10

hash -r

python --version

# =========================
# CLEAN COLAB CONFLICTS
# =========================
rm -rf /usr/local/lib/python3.12/dist-packages/nerfstudio* || true
rm -rf /usr/local/lib/python3.12/dist-packages/torch* || true
rm -f /usr/local/bin/ns-train || true

# =========================
# PIP CORE
# =========================
python -m pip install --upgrade pip setuptools wheel

# =========================
# NUMPY / SCIENTIFIC STACK (conda equivalent)
# =========================
python -m pip install numpy==1.26.4
python -m pip install scipy

# =========================
# VISION STACK (conda openimageio → pip fallback)
# =========================
python -m pip install openimageio
python -m pip install imageio imageio-ffmpeg opencv-python

# =========================
# BUILD / GEOMETRY SUPPORT
# =========================
# CGAL / suitesparse are system libs via apt above
python -m pip install pybind11

# =========================
# PYTORCH (CUDA 11.8 stable Colab)
# =========================
python -m pip install torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu118

# =========================
# NERF / 3D STACK
# =========================
python -m pip install nerfstudio==0.3.4
python -m pip install pycolmap==0.6.1

# =========================
# OPTIONAL TOOLING
# =========================
python -m pip install rclone

# =========================
# ENV FLAGS (RENDER SAFE)
# =========================
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
export OPENCV_LOG_LEVEL=ERROR
export XDG_RUNTIME_DIR=/tmp/runtime-root
export LIBGL_ALWAYS_SOFTWARE=1
export CUDA_VISIBLE_DEVICES=0

# =========================
# WRAPPER ns-train FIX
# =========================
mkdir -p $BIN_DIR

cat > $BIN_DIR/ns-train << 'EOF'
#!/usr/bin/env python3.10
from nerfstudio.scripts.train import entrypoint
import sys

sys.exit(entrypoint())
EOF

chmod +x $BIN_DIR/ns-train

export PATH="$BIN_DIR:$PATH"
hash -r

# =========================
# FINAL CHECK
# =========================
echo "PYTHON:"
python --version

echo "IMPORT TEST:"
python -c "import nerfstudio; print('NERFSTUDIO OK')"

echo "NS-TRAIN:"
ns-train --help && echo "OK" || echo "FAILED"
