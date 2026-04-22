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
# FORCE PYTHON 3.10 (IMPORTANT FIX)
# =========================
update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
update-alternatives --set python /usr/bin/python3.10

hash -r

python --version
which python

# =========================
# FORCE PIP OF PYTHON 3.10 (CRITICAL)
# =========================
python3.10 -m ensurepip --upgrade
python3.10 -m pip install --upgrade pip setuptools wheel

# ALWAYS USE PYTHON 3.10 PIPELINE
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

# PYCOLMAP (MATCHES NS 0.3.4)
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
# VERIFY INSTALL
# =========================
nvidia-smi
python --version
which python
