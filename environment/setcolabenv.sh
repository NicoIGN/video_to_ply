# =========================
# CONFIG
# =========================
WORK_DIR=/content/work
BIN_DIR=$WORK_DIR/bin

rm -rf $WORK_DIR
mkdir -p $BIN_DIR

# =========================
# SYSTEM UPDATE
# =========================
apt-get update -y

apt-get install -y \
  software-properties-common \
  colmap ffmpeg cmake ninja-build \
  libgl1-mesa-glx xvfb \
  python3.10 python3.10-dev python3.10-venv

# =========================
# FORCE CLEAN PYTHON ENV
# =========================
update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
update-alternatives --set python /usr/bin/python3.10

hash -r

echo "System python:"
python --version
which python

# =========================
# REMOVE COLAB PYTHON 3.12 PACKAGES (CRITICAL FIX)
# =========================
rm -rf /usr/local/lib/python3.12/dist-packages/nerfstudio* || true
rm -rf /usr/local/lib/python3.12/dist-packages/torch* || true

# =========================
# FORCE PYTHON SHIM
# =========================
ln -sf /usr/bin/python3.10 $BIN_DIR/python
ln -sf /usr/bin/python3.10 $BIN_DIR/python3

export PATH="$BIN_DIR:/usr/bin:$PATH"
hash -r

echo "Shim python:"
which python
python --version

# =========================
# ENSURE PIP FOR 3.10 ONLY
# =========================
python -m ensurepip --upgrade || true
python -m pip install --upgrade pip setuptools wheel

# HARD RESET PIP CACHE
python -m pip cache purge || true

# =========================
# CLEAN ML CONFLICTS
# =========================
python -m pip uninstall -y \
  nerfstudio \
  pytensor jax jaxlib \
  tensorflow tensorflow-cpu \
  torch torchvision torchaudio || true

# =========================
# NUMPY SAFE VERSION
# =========================
python -m pip install numpy==1.26.4

# =========================
# SCIENTIFIC STACK
# =========================
python -m pip install scipy imageio imageio-ffmpeg opencv-python

# =========================
# PYTORCH CUDA 11.8 (COLAB SAFE)
# =========================
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118

# =========================
# NERF STACK CLEAN INSTALL
# =========================
python -m pip install nerfstudio==0.3.4
python -m pip install pycolmap==3.11.1

# =========================
# ENV FIXES
# =========================
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
export OPENCV_LOG_LEVEL=ERROR
export XDG_RUNTIME_DIR=/tmp/runtime-root

# IMPORTANT: avoid GPU crash fallback
export CUDA_VISIBLE_DEVICES=0
export LIBGL_ALWAYS_SOFTWARE=1

# =========================
# FINAL CHECK
# =========================
nvidia-smi || true

echo "FINAL PYTHON:"
python --version
which python

echo "NERFSTUDIO:"
ns-train --help >/dev/null && echo "OK" || echo "FAILED"
