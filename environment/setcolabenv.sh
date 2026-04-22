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
# FORCE PYTHON 3.10
# =========================
update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
update-alternatives --set python /usr/bin/python3.10

python --version

# =========================
# PYTHON TOOLCHAIN
# =========================
python -m ensurepip --upgrade
pip install --upgrade pip setuptools wheel

# =========================
# CLEAN COLAB ML CONFLICTS
# =========================
pip uninstall -y pytensor jax jaxlib tensorflow tensorflow-cpu || true

# =========================
# NUMPY (NERF SAFE)
# =========================
pip install numpy==1.26.4

# =========================
# SCIENTIFIC STACK
# =========================
pip install scipy imageio imageio-ffmpeg opencv-python

# =========================
# PYTORCH (CUDA 11.8 STABLE COLAB)
# =========================
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# =========================
# NERF STACK (STABLE FOR PYTHON 3.10)
# =========================
pip install nerfstudio==0.3.4

# PYCOLMAP (MATCHES NS 0.3.4)
pip install pycolmap==3.11.1

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
pip install rclone-python || true

# =========================
# VERIFY INSTALL
# =========================
nvidia-smi
python --version
