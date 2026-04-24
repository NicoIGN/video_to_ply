#!/bin/bash
set -e


# ======================
# DEFAULTS
# ======================
FPS=10
MAX_ITER=2000
DEVICE="cpu"
ROOT_DIR="runs/default"
VIDEO=""
SKIP_CONDA=false
NO_PROXY=false

# ======================
# PIPELINE SKIP DEFAULTS (from config.sh, overridable by CLI)
# ======================
SKIP_FRAME_EXTRACTION=false
SKIP_COLMAP=false
SKIP_TRAINING=false
SKIP_EXPORT=false

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
  --root                 Root output directory (default: runs/default)
  --fps                  Frame extraction FPS (default: 10)
  --device               cpu | gpu (default: cpu)
  --max-iter             Training iterations (default: 2000)
  --skip-conda           Skip conda environment setup (useful for Colab)

  --skip-frame-extraction  Skip frame extraction step
  --skip-colmap            Skip COLMAP step
  --skip-training          Skip training step
  --skip-export            Skip export step
  --no-proxy               ignore all proxy config

  --help                 Show this help
EOF
}

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
    --skip-conda) SKIP_CONDA=true; shift ;;
    --no-proxy) NO_PROXY=true; shift ;;

    # ======================
    # PIPELINE OVERRIDES
    # ======================
    --skip-frame-extraction) SKIP_FRAME_EXTRACTION=true; shift ;;
    --skip-colmap) SKIP_COLMAP=true; shift ;;
    --skip-training) SKIP_TRAINING=true; shift ;;
    --skip-export) SKIP_EXPORT=true; shift ;;

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
if [ "$SKIP_CONDA" = true ]; then
  echo "⏩ Skipping conda setup (--skip-conda enabled)"
else

  # ======================
  # LOAD CONDA FIRST
  # ======================
  source "$(conda info --base)/etc/profile.d/conda.sh"

  # ======================
  # PROXY SETUP (RUNTIME FIRST)
  # ======================
  if [ -n "$HTTP_PROXY" && "$NO_PROXY" != true ]; then
    export HTTP_PROXY="$HTTP_PROXY"
    export http_proxy="$HTTP_PROXY"
    echo "🌐 HTTP proxy enabled"
  fi

  if [ -n "$HTTPS_PROXY" && "$NO_PROXY" != true ]; then
    export HTTPS_PROXY="$HTTPS_PROXY"
    export https_proxy="$HTTPS_PROXY"
    echo "🌐 HTTPS proxy enabled"
  fi

  # ======================
  # CONDA PROXY CONFIG (SECONDARY)
  # ======================
  if [ "$NO_PROXY" != true ]; then
    conda config --set proxy_servers.http "$HTTP_PROXY" 2>/dev/null || true
    conda config --set proxy_servers.https "$HTTPS_PROXY" 2>/dev/null || true
  fi
  # ======================
  # ENV CREATE / UPDATE
  # ======================
  set +e

  if conda env list | awk '{print $1}' | grep -qw "$CONDA_ENV_NAME"; then
    echo "🔁 Updating env: $CONDA_ENV_NAME"
    CONDA_CMD="conda env update -n $CONDA_ENV_NAME -f $CONDA_ENV_FILE --prune"
  else
    echo "🆕 Creating env: $CONDA_ENV_NAME"
    CONDA_CMD="conda env create -n $CONDA_ENV_NAME -f $CONDA_ENV_FILE"
  fi

  echo "⚙️ Running: $CONDA_CMD"

  $CONDA_CMD
  STATUS=$?

  if [ $STATUS -ne 0 ]; then
    echo "❌ Conda failed ($CONDA_CMD)"
    echo "👉 Run manually for debug"
    exit 1
  fi

  set -e

  # ======================
  # ACTIVATE ENV (IMPORTANT FIX)
  # ======================
  echo "🔌 Activating env: $CONDA_ENV_NAME"
  conda activate "$CONDA_ENV_NAME"

fi

# ======================
# PYTHON VERSION CHECK
# ======================
PY_VER=$(python --version 2>&1)

if [[ "$PY_VER" != *"3.10"* && "$PY_VER" != *"3.11"* ]]; then
  echo "❌ ERROR: Python 3.10 or 3.11 required but found: $PY_VER"
  echo "👉 Supported versions: 3.10.x, 3.11.x"
  echo "👉 Aborting execution"
  exit 1
fi

echo "✅ Using python: $PY_VER "

# ======================
# MODEL VALIDATION
# ======================
CUDA_AVAILABLE=false
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi >/dev/null 2>&1; then
    CUDA_AVAILABLE=true
  fi
fi

# ======================
# MODEL VALIDATION
# ======================
CUDA_AVAILABLE=false
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi >/dev/null 2>&1; then
    CUDA_AVAILABLE=true
  fi
fi

if [ "$DEVICE" == "gpu" ] && [ "$CUDA_AVAILABLE" = false ]; then
  echo "❌ ERROR: GPU requested but CUDA is not available."
  exit 1
fi

if [ "$DEVICE" == "cpu" ]; then
  case "$MODEL" in
    nerfacto|nerf|kplanes|tensorf)
      echo "🧠 CPU model OK: $MODEL"
      ;;
    *)
      echo "❌ MODEL '$MODEL' not supported on CPU"
      exit 1
      ;;
  esac
fi

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
# STRUCTURE
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
    echo ""
    echo ""
  if [ "$WITH_COLMAP" = true ]; then
      echo "🧭 Running COLMAP..."
      bash scripts/preprocess_colmap.sh "$IMAGE_DIR" "$ORI_DIR"
  elif [ "$WITH_NERFSTUDIO" = true ]; then
      echo "🧭 Running COLMAP through NerfStudio..."
      bash scripts/preprocess_nerfstudio.sh "$IMAGE_DIR" "$ORI_DIR"
  else
    echo "❌ CONFIGURATION ERROR"
    echo "   → Neither WITH_NERFSTUDIO nor WITH_COLMAP is enabled"
    echo ""
    echo "📌 Required fix:"
    echo "   - set WITH_NERFSTUDIO=true  OR"
    echo "   - set WITH_COLMAP=true"
    echo ""
    echo "🧠 Current state:"
    echo "   WITH_NERFSTUDIO=$WITH_NERFSTUDIO"
    echo "   WITH_COLMAP=$WITH_COLMAP"
    echo ""
    exit 1
  fi
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
    echo ""
    echo ""
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
elif find "$EXPORT_DIR" -name "*.ply" | grep -q .; then
  echo "⏩ Skipping export"
else
  echo "📦 Exporting PLY..."
  RUNS_DIR="$OUTPUT_DIR"
  bash scripts/export.sh "$EXPORT_DIR" "$RUNS_DIR"
fi

echo "✅ DONE → $ROOT_DIR"
