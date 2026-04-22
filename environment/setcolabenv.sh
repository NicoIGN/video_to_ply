# =========================
# SYSTEM CLEAN (COLMAP FIRST)
# =========================
apt-get update -y
apt-get install -y colmap ffmpeg cmake ninja-build libgl1-mesa-glx xvfb python3.10 python3.10-venv

# =========================
# FORCE PYTHON VERSION (IMPORTANT FIX)
# =========================
update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
python --version

# =========================
# PYTHON TOOLCHAIN
# =========================
pip install --upgrade pip setuptools wheel

# =========================
# CLEAN CONFLICTS (COLAB POLLUTION FIX)
# =========================
pip uninstall -y pytensor jax jaxlib tensorflow tensorflow-cpu || true

# =========================
# NUMPY (NERF STABLE ZONE)
# =========================
pip install numpy==1.26.4

# =========================
# SCIENTIFIC STACK
# =========================
pip install scipy imageio imageio-ffmpeg opencv-python

# =========================
# PYTORCH (STABLE CUDA COLAB)
# =========================
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# =========================
# NERF STACK (STABLE VERSION)
# =========================
pip install nerfstudio==0.3.4

# PYCOLMAP (STABLE FOR NS 0.3.4)
pip install pycolmap==3.11.1

# =========================
# RUNTIME FIXES (COLMAP SAFE MODE)
# =========================
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
export OPENCV_LOG_LEVEL=ERROR
export XDG_RUNTIME_DIR=/tmp/runtime-root

# FORCE CPU SAFE FOR COLMAP
export CUDA_VISIBLE_DEVICES=""
export LIBGL_ALWAYS_SOFTWARE=1

# =========================
# OPTIONAL
# =========================
pip install rclone-python || true

# =========================
# CHECK
# =========================
nvidia-smi
python --version
