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
# NERF / GSPLAT STACK
# =========================
pip install nerfstudio==0.3.4

# ⚠️ IMPORTANT: pycolmap version pinned safe
pip install pycolmap==0.5.1

# =========================
# OPTIONAL TOOLS
# =========================
pip install rclone-python || true

# =========================
# VERIFY GPU
# =========================
nvidia-smi
