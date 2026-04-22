#!/bin/bash
set -e

source config/config.sh


# ======================
# HELP
# ======================
show_help() {
  cat << EOF
Usage: ./run.sh --video <path> --root <dir> [options]

Required:
  --video       Path to input video

Options:
  --root        Root output directory (default: runs/default)
  --fps         Frame extraction FPS (default: 10)
  --device      cpu | gpu (default: cpu)
  --max-iter    Training iterations (default: 2000)
  --help        Show this help
EOF
}

# ======================
# DEFAULTS
# ======================
FPS=10
MAX_ITER=2000
DEVICE="cpu"
ROOT_DIR="runs/default"
VIDEO=""

# ======================
# ARG PARSING
# ======================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) VIDEO="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --root) ROOT_DIR="$2"; shift 2 ;;
    --max-iter) MAX_ITER="$2"; shift 2 ;;
    --help) show_help; exit 0 ;;
    *) echo "❌ Unknown param: $1"; show_help; exit 1 ;;
  esac
done

# ======================
# VALIDATION
# ======================
if [ -z "$VIDEO" ]; then
  echo "❌ --video is required"
  exit 1
fi

if [ ! -f "$VIDEO" ]; then
  echo "❌ video not found"
  exit 1
fi

# ======================
# CONDA
# ======================

conda config --set proxy_servers.http "$HTTP_PROXY" 2>/dev/null || true
conda config --set proxy_servers.https "$HTTPS_PROXY" 2>/dev/null || true

source "$(conda info --base)/etc/profile.d/conda.sh"

if [ "$SKIP_CONDA_UPDATE" = true ]; then
  echo "⏩ Skipping conda update (config)"
else
    # ======================
    # PROXY SETUP
    # ======================
    if [ -n "$HTTP_PROXY" ]; then
      export HTTP_PROXY="$HTTP_PROXY"
      export http_proxy="$HTTP_PROXY"
      echo "🌐 HTTP proxy enabled"
    fi

    if [ -n "$HTTPS_PROXY" ]; then
      export HTTPS_PROXY="$HTTPS_PROXY"
      export https_proxy="$HTTPS_PROXY"
      echo "🌐 HTTPS proxy enabled"
    fi

  if conda env list | grep -q "$CONDA_ENV_NAME"; then
    echo "🔁 Updating env"
    conda env update -n "$CONDA_ENV_NAME" -f "$CONDA_ENV_FILE" --prune > /dev/null 2>&1
  else
    echo "🆕 Creating env"
    conda env create -n "$CONDA_ENV_NAME" -f "$CONDA_ENV_FILE" > /dev/null 2>&1
  fi
fi

conda activate "$CONDA_ENV_NAME"

# ======================
# MODEL VALIDATION
# ======================

CUDA_AVAILABLE=false
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi >/dev/null 2>&1; then
    CUDA_AVAILABLE=true
  fi
fi

# GPU CHECK
if [ "$DEVICE" == "gpu" ] && [ "$CUDA_AVAILABLE" = false ]; then
  echo "❌ ERROR: GPU requested but CUDA is not available."
  echo "👉 Switch to DEVICE=cpu or install CUDA"
  exit 1
fi

# CPU COMPATIBILITY CHECK
if [ "$DEVICE" == "cpu" ]; then
  case "$MODEL" in
    nerfacto|nerf|kplanes|tensorf)
      echo "🧠 CPU model OK: $MODEL"
      ;;
    *)
      echo "❌ MODEL '$MODEL' not supported on CPU"
      echo "👉 Allowed: nerfacto | nerf | kplanes | tensorf"
      exit 1
      ;;
  esac
fi

# GPU COMPATIBILITY CHECK (warning only)
if [ "$DEVICE" == "gpu" ]; then
  case "$MODEL" in
    splatfacto|splatfacto-w|instant-ngp|zip-nerf|pynerf|feature-splatting)
      echo "🚀 GPU model OK: $MODEL"
      ;;
    *)
      echo "⚠️ MODEL '$MODEL' is not GPU-optimized (will run but may be slow)"
      ;;
  esac
fi
# ======================
# STRUCTURE (SIMPLIFIÉE)
# ======================
INPUT_DIR="$ROOT_DIR/input"
ORI_DIR="$ROOT_DIR/ori"
IMAGE_DIR="$ORI_DIR/images"
OUTPUT_DIR="$ROOT_DIR/outputs"
EXPORT_DIR="$ROOT_DIR/exports"
TRAIN_DIR="$ROOT_DIR"

mkdir -p "$INPUT_DIR" "$IMAGE_DIR" "$OUTPUT_DIR" "$EXPORT_DIR" "$TRAIN_DIR"

# copy video
if [ ! -f "$INPUT_DIR/video.mov" ]; then
  cp "$VIDEO" "$INPUT_DIR/video.mov"
fi
VIDEO="$INPUT_DIR/video.mov"

echo "📦 ROOT: $ROOT_DIR"

# ======================
# SKIP LOGIC
# ======================

# ----------------------
# 1. FRAME EXTRACTION
# ----------------------
if [ "$SKIP_FRAME_EXTRACTION" = true ]; then
  echo "⏩ Skipping frame extraction (config)"
elif [ -d "$IMAGE_DIR" ] && [ "$(ls -A "$IMAGE_DIR" 2>/dev/null)" ]; then
  echo "⏩ Skipping frame extraction"
else
  echo "🎬 Extracting frames → $IMAGE_DIR"
  bash scripts/extract_frames.sh "$VIDEO" "$FPS" "$IMAGE_DIR"
fi

# ----------------------
# 2. COLMAP
# ----------------------
if [ "$SKIP_COLMAP" = true ]; then
  echo "⏩ Skipping COLMAP (config)"
elif [ -f "$ORI_DIR/transforms.json" ]; then
  echo "⏩ Skipping COLMAP"
else
  echo "🧭 Running COLMAP..."
  bash scripts/preprocess_colmap.sh "$IMAGE_DIR" "$ORI_DIR"
fi

# ----------------------
# 3. TRAIN
# ----------------------
if [ "$SKIP_TRAINING" = true ]; then
  echo "⏩ Skipping training (config)"
else
  LATEST_RUN=$(ls -td "$OUTPUT_DIR"/ori/$MODEL/* 2>/dev/null | head -n 1 || true)

  if [ -n "$LATEST_RUN" ] && [ -d "$LATEST_RUN/nerfstudio_models" ]; then
    echo "⏩ Skipping training"
  else
    echo "🧠 Training..."

    MODEL="$MODEL" \
    DEVICE="$DEVICE" \
    MAX_ITER="$MAX_ITER" \
    DATA="$ORI_DIR" \
    OUTPUTDIR="$TRAIN_DIR" \
    TRAIN_RAYS_PER_BATCH="$TRAIN_RAYS_PER_BATCH" \
    CAMERA_RES_SCALE_FACTOR="$CAMERA_RES_SCALE_FACTOR" \
    NUM_NERF_SAMPLES_PER_RAY="$NUM_NERF_SAMPLES_PER_RAY" \
    NUM_PROPOSAL_SAMPLES_PER_RAY="$NUM_PROPOSAL_SAMPLES_PER_RAY" \
    MAX_RES="$MAX_RES" \
    MODEL_IMPLEMENTATION="$MODEL_IMPLEMENTATION" \
    bash scripts/train.sh
  fi
fi

# ----------------------
# 4. EXPORT
# ----------------------
if [ "$SKIP_EXPORT" = true ]; then
  echo "⏩ Skipping export (config)"
elif ls "$EXPORT_DIR"/*.ply >/dev/null 2>&1; then
  echo "⏩ Skipping export"
else
  echo "📦 Exporting PLY..."
  bash scripts/export.sh "$EXPORT_DIR" "$OUTPUT_DIR"
fi

echo "✅ DONE → $ROOT_DIR"
