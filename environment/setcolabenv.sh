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
  python3.10 python3.10-dev python3.10-venv \
  python3-pip

# =========================
# FORCE PYTHON 3.10 SYSTEM
# =========================
update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
update-alternatives --set python /usr/bin/python3.10

hash -r

echo "System python:"
python --version
which python

# =========================
# CRITICAL CLEAN (Colab 3.12 residue)
# =========================
rm -rf /usr/local/lib/python3.12/dist-packages/nerfstudio* || true
rm -rf /usr/local/lib/python3.12/dist-packages/torch* || true
rm -f /usr/local/bin/ns-train || true

# =========================
# FORCE SHIM PYTHON 3.10
# =========================
ln -sf /usr/bin/python3.10 $BIN_DIR/python
ln -sf /usr/bin/python3.10 $BIN_DIR/python3

export PATH="$BIN_DIR:/usr/bin:$PATH"
hash -r

echo "Shim python:"
which python
python --version

# =========================
# PIP SETUP (FIX IMPORTANT)
# =========================
python -m pip install --upgrade pip setuptools wheel
python -m pip cache purge || true

# =========================
# CLEAN ML STACK CONFLICTS
# =========================
python -m pip uninstall -y \
  nerfstudio pytorch-lightning \
  pytensor jax jaxlib \
  tensorflow tensorflow-cpu \
  torch torchvision torchaudio || true

# =========================
# CORE SCIENCE STACK
# =========================
python -m pip install numpy==1.26.4
python -m pip install scipy imageio imageio-ffmpeg opencv-python

# =========================
# PYTORCH (CUDA 11.8 COLAB SAFE)
# =========================
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118

# =========================
# NERF STACK (STABLE INSTALL)
# =========================
python -m pip install --no-cache-dir nerfstudio==0.3.4
python -m pip install pycolmap==3.11.1

# =========================
# VERIFY IMPORT
# =========================
python -c "import nerfstudio; print('NERFSTUDIO OK')"

# =========================
# FORCE CORRECT ns-train WRAPPER
# =========================
cat > $BIN_DIR/ns-train << 'EOF'
#!/usr/bin/env python3.10
import sys
from nerfstudio.scripts.train import entrypoint

sys.exit(entrypoint())
EOF

chmod +x $BIN_DIR/ns-train

export PATH="$BIN_DIR:$PATH"
hash -r

# =========================
# ENV FIXES (RENDER / COLMAP SAFE)
# =========================
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
export OPENCV_LOG_LEVEL=ERROR
export XDG_RUNTIME_DIR=/tmp/runtime-root

export CUDA_VISIBLE_DEVICES=0
export LIBGL_ALWAYS_SOFTWARE=1

# =========================
# FINAL CHECK
# =========================
nvidia-smi || true

echo "FINAL PYTHON:"
python --version
which python

echo "NS-TRAIN:"
ns-train --help && echo "OK" || echo "FAILED"
