#!/bin/bash
set -e


# ======================
# DEFAULTS
# ======================
DEVICE="cpu"
ROOT_DIR="runs/default"
SKIP_CONDA=false
NO_PROXY=false

# ======================
# PIPELINE SKIP DEFAULTS (from config.sh, overridable by CLI)
# ======================
SKIP_FRAME_EXTRACTION=false
SKIP_COLMAP=false
SKIP_TRAINING=false
SKIP_EXPORT=false
SKIP_FILTER=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


source $SCRIPT_DIR/config/config.sh

cd $SCRIPT_DIR
# ======================
# INPUT MODE
# ======================
INPUT_MODE="video"
VIDEO=""
IMAGES=""

unset FPS
unset NUM_FRAMES

# ======================
# HELP
# ======================
show_help() {
  cat << EOF
Usage:
  Video mode:
    ./run.sh --video <path> --root <dir> [options]

  Images mode:
    ./run.sh --images <dir> --root <dir> [options]

Required:
  --video <file>         Input video
       or
  --images <directory>   Directory containing source images

Options:
  --root                 Root output directory (default: runs/default)
  --num-frames           Number of frames to extract (video mode only)
  --skip-conda           Skip conda environment setup
  --profile              fast | balanced | quality | best
  --name                 base name of the outputfile

  --skip-frame-extraction
  --skip-colmap
  --skip-training
  --skip-export
  --skip-filter
  --no-proxy
  --help
EOF
}

# ======================
# ARG PARSING
# ======================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) VIDEO="$2"; INPUT_MODE="video"; shift 2 ;;
    --images) IMAGES="$2"; INPUT_MODE="images"; shift 2 ;;
    --num-frames) NUM_FRAMES="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --name) BASENAME="$2"; shift 2 ;;
    --root) ROOT_DIR="$2"; shift 2 ;;
    --skip-conda) SKIP_CONDA=true; shift ;;
    --no-proxy) NO_PROXY=true; shift ;;

    # ======================
    # PIPELINE OVERRIDES
    # ======================
    --skip-frame-extraction) SKIP_FRAME_EXTRACTION=true; shift ;;
    --skip-colmap) SKIP_COLMAP=true; shift ;;
    --skip-training) SKIP_TRAINING=true; shift ;;
    --skip-export) SKIP_EXPORT=true; shift ;;
    --skip-filter) SKIP_FILTER=true; shift ;;

    --help) show_help; exit 0 ;;
    *) echo "❌ Unknown param: $1"; show_help; exit 1 ;;
  esac
done


# ======================
# VALIDATION
# ======================

BASENAME=${BASENAME:-gsplat_$(date +%Y%m%d_%H%M%S)}

case "$INPUT_MODE" in
  video)
    if [ -z "$VIDEO" ]; then
      echo "❌ --video is required in $INPUT_MODE mode"
      exit 1
    fi

    if [ ! -f "$VIDEO" ]; then
      echo "❌ Video not found: $VIDEO"
      exit 1
    fi
    ;;

  images)
    if [ -z "$IMAGES" ]; then
      echo "❌ --images is required in $INPUT_MODE mode"
      exit 1
    fi

    if [ ! -d "$IMAGES" ]; then
      echo "❌ Images directory not found: $IMAGES"
      exit 1
    fi

    IMAGE_COUNT=$(find "$IMAGES" -maxdepth 1 -type f \
      \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) \
      | wc -l | tr -d ' ')

    if [ "$IMAGE_COUNT" -lt 2 ]; then
      echo "❌ At least 2 images are required"
      echo "   Found: $IMAGE_COUNT"
      exit 1
    fi
    ;;

  *)
    echo "❌ Invalid INPUT_MODE: $INPUT_MODE"
    exit 1
    ;;
esac



if [ -z "$PROFILE" ]; then
  echo "⚠️  no profile loaded"
else
  if [ -f "profiles/${PROFILE}.sh" ]; then
    source profiles/${PROFILE}.sh
     echo "👉 using profile: ${TRAINING_PROFILE}"
  else
    echo "❌  profile ${PROFILE} not found"
    echo "❌  use profile [gpu|cpu]/fast|quality|balanced|best"
    exit 1
  fi
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

if [ "$NO_PROXY" != true ]; then
  echo NO_PROXY: $NO_PROXY
  
  if [ -n "$HTTP_PROXY" ]; then
    export HTTP_PROXY="$HTTP_PROXY"
    export http_proxy="$HTTP_PROXY"
    echo "🌐 HTTP proxy enabled (1)"
  fi

  if [ -n "$HTTPS_PROXY" ]; then
    export HTTPS_PROXY="$HTTPS_PROXY"
    export https_proxy="$HTTPS_PROXY"
    echo "🌐 HTTPS proxy enabled (2)"
  fi

  # ======================
  # CONDA PROXY CONFIG (SECONDARY)
  # ======================
  conda config --set proxy_servers.http "$HTTP_PROXY" 2>/dev/null || true
  conda config --set proxy_servers.https "$HTTPS_PROXY" 2>/dev/null || true

else
  echo "🚫 Proxy disabled via NO_PROXY=true"
  # 🔥 clean conda config
  conda config --remove-key proxy_servers.http 2>/dev/null || true
  conda config --remove-key proxy_servers.https 2>/dev/null || true
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
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
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
OUTPUT_DIR="$ROOT_DIR/model3d"
EXPORT_DIR="$ROOT_DIR/exports"
TRAIN_DIR="$ROOT_DIR"

mkdir -p "$INPUT_DIR" "$IMAGE_DIR" "$OUTPUT_DIR" "$EXPORT_DIR" "$TRAIN_DIR"

# copy input dataset
case "$INPUT_MODE" in
  video)
    if [ ! -f "$INPUT_DIR/video.mov" ]; then
      cp "$VIDEO" "$INPUT_DIR/video.mov"
    fi
    VIDEO="$INPUT_DIR/video.mov"
    ;;

  images)
    echo "🖼️ Importing images from: $IMAGES"

    mkdir -p "$INPUT_DIR"

    if [ ! -d "$IMAGES" ]; then
      echo "❌ Images directory not found: $IMAGES"
      exit 1
    fi

    SOURCE_COUNT=$(find "$IMAGES" -maxdepth 1 -type f \
      \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)

    TARGET_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -type f \
      \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)

    if [ "$SOURCE_COUNT" -eq 0 ]; then
      echo "❌ No images found in: $IMAGES"
      exit 1
    fi

    if [ "$TARGET_COUNT" -eq "$SOURCE_COUNT" ]; then
      echo "⏩ All images already imported in $INPUT_DIR, skipping copy"
    else
      echo "📥 Copying $SOURCE_COUNT images to $INPUT_DIR"
      find "$IMAGES" -maxdepth 1 -type f \
        \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) \
        -exec cp {} "$INPUT_DIR"/ \;
    fi
    ;;

  *)
    echo "❌ Invalid INPUT_MODE: $INPUT_MODE"
    exit 1
    ;;
esac

echo "📦 ROOT: $ROOT_DIR"


# ----------------------
# 1. INPUT PREPARATION
# ----------------------
case "$INPUT_MODE" in
  video)
    if [ "$SKIP_FRAME_EXTRACTION" = true ]; then
      echo "⏩ Skipping frame extraction (config)"

    elif [ -d "$IMAGE_DIR" ] && [ "$(ls -A "$IMAGE_DIR" 2>/dev/null)" ]; then
      echo "⏩ Skipping frame extraction"
    else
      if [[ -n "${FPS:-}" ]]; then
        echo "🎬 Extracting frames at ${FPS} FPS → $IMAGE_DIR"
        FPS="$FPS" \
        IMAGE_DIR="$IMAGE_DIR" \
        VIDEO="$VIDEO" \
        bash scripts/extract_frames.sh
      else
        echo "🎬 Extracting $NUM_FRAMES sharp frames → $IMAGE_DIR"
        NUM_FRAMES="$NUM_FRAMES" \
        IMAGE_DIR="$IMAGE_DIR" \
        VIDEO="$VIDEO" \
        bash scripts/extract_frames.sh
      fi
    fi
    ;;

  images)
    if [ ! -d "$IMAGES" ]; then
      echo "❌ Images directory not found: $IMAGES"
      exit 1
    fi

    if [ "$SKIP_FRAME_EXTRACTION" = true ]; then
      echo "⏩ Skipping image preparation (config)"
    elif [ -d "$IMAGE_DIR" ] && [ "$(ls -A "$IMAGE_DIR" 2>/dev/null)" ]; then
      echo "⏩ Skipping image preparation"
    else
      echo "🖼️ Preparing images → $IMAGE_DIR"
      bash scripts/prepare_images.sh "$IMAGES" "$IMAGE_DIR"
    fi
    ;;

  *)
    echo "❌ Invalid INPUT_MODE: $INPUT_MODE"
    exit 1
    ;;
esac

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
    echo "🧭 Running COLMAP through NerfStudio..."

    DATA_DIR="$IMAGE_DIR" \
    OUTPUT_DIR="$ORI_DIR" \
    DEVICE="$DEVICE" \
    CAMERA_TYPE="$CAMERA_TYPE" \
    MATCHING_METHOD="$MATCHING_METHOD" \
    NUM_DOWNSCALES="$NUM_DOWNSCALES" \
    SFMT_TOOL="$SFMT_TOOL" \
    FEATURE_TYPE="$FEATURE_TYPE" \
    MATCHER_TYPE="$MATCHER_TYPE" \
    PERCENT_RADIUS_CROP="$PERCENT_RADIUS_CROP" \
    CAMERA_RES_SCALE_FACTOR="$CAMERA_RES_SCALE_FACTOR" \
    USE_SINGLE_CAMERA_MODE="$USE_SINGLE_CAMERA_MODE" \
    REFINE_INTRINSICS="$REFINE_INTRINSICS" \
    CROP_FACTOR="$CROP_FACTOR" \
    bash scripts/preprocess_nerfstudio.sh
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
    MODEL_IMPLEMENTATION="$MODEL_IMPLEMENTATION" \
    DEVICE="$DEVICE" \
    MAX_ITER="$MAX_ITER" \
    REFINE_EVERY="$REFINE_EVERY" \
    MAX_JOBS="$MAX_JOBS" \
    STEPS_PER_SAVE="$STEPS_PER_SAVE" \
    STEPS_PER_EVAL_ALL_IMAGES="$STEPS_PER_EVAL_ALL_IMAGES" \
    DATA="$ORI_DIR" \
    EXPERIMENT_NAME="$EXPERIMENT_NAME" \
    OUTPUTDIR="$TRAIN_DIR" \
    TRAIN_RAYS_PER_BATCH="$TRAIN_RAYS_PER_BATCH" \
    CAMERA_RES_SCALE_FACTOR="$CAMERA_RES_SCALE_FACTOR" \
    NUM_NERF_SAMPLES_PER_RAY="$NUM_NERF_SAMPLES_PER_RAY" \
    NUM_PROPOSAL_SAMPLES_PER_RAY="$NUM_PROPOSAL_SAMPLES_PER_RAY" \
    MAX_RES="$MAX_RES" \
    DENSIFY_GRAD_THRESH="$DENSIFY_GRAD_THRESH" \
    CULL_ALPHA_THRESH="$CULL_ALPHA_THRESH" \
    CULL_SCREEN_SIZE="$CULL_SCREEN_SIZE" \
    SPLIT_SCREEN_SIZE="$SPLIT_SCREEN_SIZE" \
    bash scripts/train.sh
  fi
fi

# ----------------------
# 4. EXPORT
# ----------------------

if [[ "$SKIP_EXPORT" == "true" ]]; then
  echo "⏩ Skipping export (config)"
else

  PLY_FOUND=$(find "$OUTPUT_DIR" -type f -name "*.ply" | head -n 1)

  if [[ -n "$PLY_FOUND" ]]; then
    echo "📦 Existing PLY found: $PLY_FOUND"
    # echo "⏩ Skipping export (PLY already exists)"
    cp $PLY_FOUND ${PLY_FOUND}.bkp
  fi

    echo "🚀 Exporting model in $EXPORT_DIR..."

    case "$MODEL" in
      *nerfacto*)
        echo "📦 Exporting Nerfacto point cloud (.ply)..."

        OUTPUT_DIR="$TRAIN_DIR/$EXPERIMENT_NAME" \
        EXPORT_DIR="$OUTPUT_DIR" \
        NUM_POINTS="$EXPORT_NUM_POINTS" \
        NORMAL_METHOD="$NORMAL_METHOD" \
        REMOVE_OUTLIERS="$REMOVE_OUTLIERS" \
        bash scripts/export_nerf_to_ply.sh
        ;;

      *splatfacto*)
        echo "📦 Exporting Gaussian Splat (.ply)..."

        OUTPUT_DIR="$TRAIN_DIR/$EXPERIMENT_NAME" \
        EXPORT_DIR="$OUTPUT_DIR" \
        bash scripts/export_splat_to_ply.sh
        ;;

      *)
        echo "⚠️ Unsupported model for export: $MODEL"
        exit 1
        ;;
    esac

    # ======================
    # VALIDATION
    # ======================
    PLY_FILE=$(find "$OUTPUT_DIR" -type f -name "*.ply" | head -n 1)

    if [[ -f "$PLY_FILE" ]]; then
      echo "✅ PLY export successful: $PLY_FILE"
    else
      echo "❌ PLY export failed"
      exit 1
    fi
fi

# ======================
# 5. CLEAN PLY
# ======================

echo "🧹 Removing filtered Gaussian Splat files..."

FILES_TO_DELETE=($(find "$EXPORT_DIR" -type f -name "${BASENAME}_*.ply" | sort))

if [[ ${#FILES_TO_DELETE[@]} -eq 0 ]]; then
    echo "⚠️ No filtered files to remove in $EXPORT_DIR for basename: $BASENAME"
else

    echo "📦 Found ${#FILES_TO_DELETE[@]} file(s) to delete"

    for FILE in "${FILES_TO_DELETE[@]}"; do
        echo "🗑️ Deleting: $(basename "$FILE")"
        rm -f "$FILE"
    done

    echo "✅ Cleanup complete"
fi

echo "📦 Source PLY: $PLY_FILE"

PLY_FILE="$PLY_FILE" \
EXPORT_DIR="$EXPORT_DIR" \
BASENAME="$BASENAME" \
SKIP_FILTER="$SKIP_FILTER" \
bash "$SCRIPT_DIR/scripts/filter_ply.sh"

