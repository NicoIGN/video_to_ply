# =========================
# CONFIG
# =========================
WORK_DIR=/content/work
BIN_DIR=$WORK_DIR/bin

mkdir -p $BIN_DIR

# =========================
# SYSTEM + PYTHON 3.10
# =========================
apt-get update -y

apt-get install -y \
  software-properties-common \
  colmap ffmpeg cmake ninja-build \
  libgl1-mesa-glx xvfb \
  python3.10 python3.10-dev python3.10-venv

# =========================
# FORCE PYTHON 3.10 (SYSTEM LEVEL)
# =========================
update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
update-alternatives --set python /usr/bin/python3.10

hash -r

echo "Python system:"
python --version
which python

# =========================
# CREATE WORKSPACE PYTHON SHIM (IMPORTANT)
# =========================
ln -sf /usr/bin/python3.10 $BIN_DIR/python
ln -sf /usr/bin/python3.10 $BIN_DIR/python3

export PATH="$BIN_DIR:/usr/bin:$PATH"
hash -r

echo "Python shim:"
which python
python --version

# =========================
# FORCE PIP FOR PYTHON 3.10
# =========================
python3.10 -m ensurepip --upgrade || true
python3.10 -m pip install --upgrade pip setuptools wheel

# ALWAYS USE PYTHON 3.10
alias pip='python3.10 -m pip'

# =========================
# CLEAN COLAB ML CONFLICTS
# =========================
python3.10 -m pip uninstall -y pytensor jax jaxlib tensorflow tensorflow-cpu || true

# =========================
# NUMPY (NERF SAFE)
# =========================
python3.10 -m pip install numpy==1.26.4

# =========================
# SCIENTIFIC STACK
# =========================
python3.10 -m pip install scipy imageio imageio-ffmpeg opencv-python

# =========================
# PYTORCH (CUDA 11.8 STABLE COLAB)
# =========================
python3.10 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# =========================
# NERF STACK (STABLE FOR PYTHON 3.10)
# =========================
python3.10 -m pip install nerfstudio==0.3.4

# PYCOLMAP
python3.10 -m pip install pycolmap==3.11.1

# =========================
# RUNTIME FIXES (COLMAP SAFE MODE)
# =========================
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
export OPENCV_LOG_LEVEL=ERROR
export XDG_RUNTIME_DIR=/tmp/runtime-root

# FORCE CPU SAFE FOR COLMAP OPENGL
export CUDA_VISIBLE_DEVICES=""
export LIBGL_ALWAYS_SOFTWARE=1

# =========================
# OPTIONAL TOOLS
# =========================
python3.10 -m pip install rclone-python || true

# =========================
# FINAL CHECK
# =========================
nvidia-smi
echo "Final python:"
python --version
which python
