# =========================
# SYSTEM CLEAN (COLMAP FIRST)
# =========================
apt-get update -y
apt-get install -y colmap ffmpeg cmake ninja-build libgl1-mesa-glx xvfb

# =========================
# PYTHON TOOLCHAIN STABLE
# =========================
pip install --upgrade pip setuptools wheel

# 🚨 IMPORTANT: remove ML conflict packages (Colab pollution fix)
pip uninstall -y pytensor jax jaxlib tensorflow tensorflow-cpu || true

# =========================
# NUMPY (COLMAP + NERF SAFE ZONE)
# =========================
pip install numpy==1.26.4

# =========================
# SCIENTIFIC STACK
# =========================
pip install scipy imageio imageio-ffmpeg opencv-python

# =========================
# PYTORCH (STABLE COLAB CUDA)
# =========================
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# =========================
# NERF STACK (CLEAN INSTALL)
# =========================
pip install nerfstudio

# pycolmap (DO NOT PIN ON PYTHON 3.12)
pip install pycolmap

# =========================
# RUNTIME FIXES (CRITICAL FOR COLMAP)
# =========================
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
export OPENCV_LOG_LEVEL=ERROR
export XDG_RUNTIME_DIR=/tmp/runtime-root

# 🔥 FORCE CPU SAFE MODE FOR COLMAP
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
