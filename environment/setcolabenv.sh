# =========================
# BASE SYSTEM
# =========================
apt-get update -y
apt-get install -y colmap ffmpeg cmake ninja-build libgl1-mesa-glx xvfb

# =========================
# PYTHON CORE
# =========================
pip install --upgrade pip setuptools wheel

pip install numpy scipy imageio imageio-ffmpeg opencv-python

# =========================
# TORCH (CUDA 11.8 stable Colab)
# =========================
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# =========================
# NERF STACK (Python 3.12 compatible)
# =========================
pip install nerfstudio

# pycolmap (NE PAS PINNER)
pip install pycolmap

# =========================
# COLMAP / GPU SAFE RUNTIME
# =========================
pip install pybind11

# =========================
# OPTIONAL TOOLS
# =========================
pip install rclone-python || true

# =========================
# ENV (IMPORTANT COLAB FIX)
# =========================
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
export OPENCV_LOG_LEVEL=ERROR
export XDG_RUNTIME_DIR=/tmp/runtime-root

# force CPU safe mode for COLMAP/OpenGL
export CUDA_VISIBLE_DEVICES=""
export LIBGL_ALWAYS_SOFTWARE=1

# =========================
# VERIFY
# =========================
nvidia-smi
python --version
