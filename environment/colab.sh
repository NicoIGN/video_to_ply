# =========================
# BASE SYSTEM
# =========================
apt-get update
apt-get install -y colmap ffmpeg cmake ninja-build

# =========================
# PYTHON CORE
# =========================
pip install --upgrade pip

pip install numpy==1.26.4 scipy imageio imageio-ffmpeg opencv-python

# =========================
# GPU TORCH STACK
# =========================
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# =========================
# NERF / GSPLAT STACK
# =========================
pip install nerfstudio==0.3.4
pip install pycolmap==0.6.1

# =========================
# OPTIONAL TOOLS (RCLONE, IO)
# =========================
pip install rclone-python || true

# =========================
# VERIFY GPU
# =========================
nvidia-smi
