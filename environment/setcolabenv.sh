# =========================
# BASE SYSTEM
# =========================
apt-get update -y
apt-get install -y colmap ffmpeg cmake ninja-build libgl1-mesa-glx xvfb

# =========================
# PYTHON CORE
# =========================
pip install --upgrade pip

pip install numpy==1.26.4 scipy imageio imageio-ffmpeg opencv-python

# =========================
# TORCH STACK (COLAB SAFE)
# =========================
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# =========================
# NERF STACK (IMPORTANT FIX)
# =========================
pip install nerfstudio

# ⚠️ pycolmap MUST NOT be pinned on Python 3.12
pip install pycolmap

# =========================
# OPTIONAL TOOLS
# =========================
pip install rclone-python || true

# =========================
# VERIFY GPU
# =========================
nvidia-smi
