#!/bin/bash
set -e

# ======================
# TIMING UTILS
# ======================

SCRIPT_START=$(date +%s)

format_duration() {
  local seconds=$1

  local h=$((seconds / 3600))
  local m=$(((seconds % 3600) / 60))
  local s=$((seconds % 60))

  if [ $h -gt 0 ]; then
    printf "%02dh %02dm %02ds" "$h" "$m" "$s"
  elif [ $m -gt 0 ]; then
    printf "%02dm %02ds" "$m" "$s"
  else
    printf "%02ds" "$s"
  fi
}

print_step_time() {
  local label="$1"
  local start_ts="$2"

  local end_ts=$(date +%s)
  local elapsed=$((end_ts - start_ts))

  echo ""
  echo "⏱️  ${label} completed in $(format_duration "$elapsed")"
  echo ""
}

# ======================
# DEFAULTS
# ======================
DEVICE="cpu"
ROOT_DIR="runs/default"
SKIP_CONDA=false
IGNORE_PROXY=false
AUTOMASK=false

# ======================
# PIPELINE SKIP DEFAULTS (from config.sh, overridable by CLI)
# ======================
SKIP_FRAME_EXTRACTION=false
SKIP_PREPROCESS=false
SKIP_TRAINING=false
SKIP_EXPORT=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PREPROCESS_PROFILE=""
GSPLAT_PROFILE=""

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
  --root <dir>                 Root output directory (default: runs/default)
  --name <name>                Base name of outputs (default: gsplat_<timestamp>)

  --num-frames <int>          Number of frames to extract (video mode only)
      or
  --fps <int>                 Extract frames at fixed FPS (video mode only)

  --preprocess-profile <name> colmap | hloc | hloc-lightblue
  --gsplat-profile <name>     fast | balanced | quality | quality_plus
  --automask                  Enable automatic masking (default: false)


  # Pipeline skips
  --skip-conda                Skip conda environment setup
  --no-proxy                  Disable proxy configuration
  --skip-frame-extraction
  --skip-preprocess
  --skip-training
  --skip-export

Optional environment variables:
  COLMAP_ARCHIVE=<file>      Path to COLMAP zip archive
                              (restores ORI_DIR/colmap + transforms.json before preprocess)

  VIDEO_START=<seconds>       Start time for video trimming
  VIDEO_END=<seconds>         End time for video trimming

Examples:
  Video:
    ./run.sh --video input.mov --fps 2 --gsplat-profile quality

  Images:
    ./run.sh --images ./imgs --preprocess-profile colmap

  Resume COLMAP:
    COLMAP_ARCHIVE=colmap.zip ./run.sh --video input.mov
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
    --preprocess-profile) PREPROCESS_PROFILE="$2"; shift 2 ;;
    --gsplat-profile) GSPLAT_PROFILE="$2"; shift 2 ;;
    --name) BASENAME="$2"; shift 2 ;;
    --root) ROOT_DIR="$2"; shift 2 ;;
    --skip-conda) SKIP_CONDA=true; shift ;;
    --no-proxy) IGNORE_PROXY=true; shift ;;
    --automask) AUTOMASK=true; shift ;;
    # ======================
    # PIPELINE OVERRIDES
    # ======================
    --skip-frame-extraction) SKIP_FRAME_EXTRACTION=true; shift ;;
    --skip-preprocess) SKIP_PREPROCESS=true; shift ;;
    --skip-training) SKIP_TRAINING=true; shift ;;
    --skip-export) SKIP_EXPORT=true; shift ;;

    --help) show_help; exit 0 ;;
    *) echo "❌ Unknown param: $1"; show_help; exit 1 ;;
  esac
done


NO_PROXY="$IGNORE_PROXY" \
MAX_JOBS="$MAX_JOBS" \
source $SCRIPT_DIR/config/config.sh

# ======================
# SETUP VALIDATION
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

if [ -z "$PREPROCESS_PROFILE" ]; then
  echo "⚠️  no preprocessing profile loaded"
else
  if [ -f "profiles/preprocess/${PREPROCESS_PROFILE}.sh" ]; then
    source profiles/preprocess/${PREPROCESS_PROFILE}.sh
     echo "👉 using profile: preprocess/${PREPROCESS_PROFILE}"
  else
    echo "❌  profile preprocess/${PREPROCESS_PROFILE}.sh not found"
    echo "❌  use profil hloc | colmap"
    exit 1
  fi
fi

if [ -z "$GSPLAT_PROFILE" ]; then
  echo "⚠️  no gsplat profile loaded"
else
  if [ -f "profiles/gsplat/${GSPLAT_PROFILE}.sh" ]; then
    source profiles/gsplat/${GSPLAT_PROFILE}.sh
     echo "👉 using profile: gsplat/${GSPLAT_PROFILE}"
  else
    echo "❌  profile gsplat/${GSPLAT_PROFILE} not found"
    echo "❌  use profile fast|quality|balanced|best"
    exit 1
  fi
fi

# ======================
# CONDA ENVIRONMENT
# ======================
if [ "$SKIP_CONDA" = true ]; then
    echo "⏩ Skipping conda setup (--skip-conda enabled)"
else
    ### LOAD CONDA FIRST
    source "$(conda info --base)/etc/profile.d/conda.sh"

    ### PROXY SETUP (RUNTIME FIRST)
    if [ "$IGNORE_PROXY" != true ]; then
      echo IGNORE_PROXY: $IGNORE_PROXY
      
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

      ### CONDA PROXY CONFIG
      conda config --set proxy_servers.http "$HTTP_PROXY" 2>/dev/null || true
      conda config --set proxy_servers.https "$HTTPS_PROXY" 2>/dev/null || true

    else
      echo "🚫 Proxy disabled via IGNORE_PROXY=true"
      # 🔥 clean conda config
      conda config --remove-key proxy_servers.http 2>/dev/null || true
      conda config --remove-key proxy_servers.https 2>/dev/null || true
    fi

    ### ENV CREATE / UPDATE
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

    ### ACTIVATE ENV
    echo "🔌 Activating env: $CONDA_ENV_NAME"
    conda activate "$CONDA_ENV_NAME"

    bash scripts/check_torch_stack.sh
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
# DATA STRUCTURE
# ======================
INPUT_DIR="$ROOT_DIR/input"
ORI_DIR="$ROOT_DIR/ori"
IMAGE_DIR="$ORI_DIR/images"
OUTPUT_DIR="$ROOT_DIR/model3d"
EXPORT_DIR="$ROOT_DIR/exports"
TRAIN_DIR="$ROOT_DIR"

mkdir -p "$INPUT_DIR/images" "$IMAGE_DIR" "$OUTPUT_DIR" "$EXPORT_DIR" "$TRAIN_DIR"

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

    mkdir -p "$INPUT_DIR/images"

    if [ ! -d "$IMAGES" ]; then
      echo "❌ Images directory not found: $IMAGES"
      exit 1
    fi

    SOURCE_COUNT=$(find "$IMAGES" -maxdepth 1 -type f \
      \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)

    TARGET_COUNT=$(find "$INPUT_DIR/images" -maxdepth 1 -type f \
      \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)

    if [ "$SOURCE_COUNT" -eq 0 ]; then
      echo "❌ No images found in: $IMAGES"
      exit 1
    fi

    if [ "$TARGET_COUNT" -eq "$SOURCE_COUNT" ]; then
      echo "⏩ All images already imported in $INPUT_DIR/images, skipping copy"
    else
      echo "📥 Copying $SOURCE_COUNT images to $INPUT_DIR/images"
      find "$IMAGES" -maxdepth 1 -type f \
        \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) \
        -exec cp {} "$INPUT_DIR/images"/ \;
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
STEP_START=$(date +%s)

case "$INPUT_MODE" in
  video)
    if [ "$SKIP_FRAME_EXTRACTION" = true ]; then
      echo "⏩ Skipping frame extraction (config)"

    elif [ -d "$IMAGE_DIR" ] && [ "$(ls -A "$IMAGE_DIR" 2>/dev/null)" ]; then
      echo "⏩ Skipping frame extraction"

    else

      # optional trim env vars
      EXTRA_ENV=()

      [[ -n "${VIDEO_START:-}" ]] && EXTRA_ENV+=(VIDEO_START="$VIDEO_START")
      [[ -n "${VIDEO_END:-}" ]] && EXTRA_ENV+=(VIDEO_END="$VIDEO_END")

      # extraction mode
      if [[ -n "${FPS:-}" ]]; then
        EXTRA_ENV+=(FPS="$FPS")
        echo "🎬 Extracting frames at ${FPS} FPS → $INPUT_DIR/images"

      else
        EXTRA_ENV+=(NUM_FRAMES="$NUM_FRAMES")
        echo "🎬 Extracting $NUM_FRAMES sharp frames → $INPUT_DIR/images"
      fi

      env \
        IMAGE_DIR="$IMAGE_DIR" \
        VIDEO="$VIDEO" \
        "${EXTRA_ENV[@]}" \
        bash scripts/extract_frames.sh

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

print_step_time "INPUT PREPARATION" "$STEP_START"

# ----------------------
# 2. PREPROCESS
# ----------------------

# OPTIONAL COLMAP ARCHIVE RESTORE
if [ -n "${COLMAP_ARCHIVE:-}" ]; then
  echo ""
  echo "📦 COLMAP_ARCHIVE detected:"
  echo "   $COLMAP_ARCHIVE"

  if [ ! -f "$COLMAP_ARCHIVE" ]; then
    echo "❌ COLMAP archive not found: $COLMAP_ARCHIVE"
    exit 1
  fi

  echo "📂 Extracting COLMAP archive into: $ORI_DIR"

  unzip -o "$COLMAP_ARCHIVE" -d "$ORI_DIR"

  if [ ! -d "$ORI_DIR/colmap" ]; then
    echo "❌ Extraction failed: colmap directory not found after unzip"
    exit 1
  fi

  echo "✅ COLMAP archive restored successfully"
fi

if [ "$SKIP_PREPROCESS" = true ]; then
  echo "⏩ Skipping PREPROCESS (config)"
elif [ -f "$ORI_DIR/transforms.json" ]; then
  echo "⏩ Skipping PREPROCESS"
else
    echo ""
    echo ""
    echo "🧭 Preprocessing $PREPROCESS_PROFILE through NerfStudio..."
    STEP_START=$(date +%s)


    HTTP_PROXY="$HTTP_PROXY" \
    HTTPS_PROXY="$HTTPS_PROXY" \
    NO_PROXY="$NO_PROXY" \
    DATA_DIR="$IMAGE_DIR" \
    OUTPUT_DIR="$ORI_DIR" \
    DEVICE="$DEVICE" \
    CAMERA_TYPE="$CAMERA_TYPE" \
    MATCHING_METHOD="$MATCHING_METHOD" \
    SFMT_TOOL="$SFMT_TOOL" \
    FEATURE_TYPE="$FEATURE_TYPE" \
    MATCHER_TYPE="$MATCHER_TYPE" \
    NUM_DOWNSCALES="$NUM_DOWNSCALES" \
    PERCENT_RADIUS_CROP="$PERCENT_RADIUS_CROP" \
    CROP_FACTOR="$CROP_FACTOR" \
    USE_SINGLE_CAMERA_MODE="$USE_SINGLE_CAMERA_MODE" \
    REFINE_INTRINSICS="$REFINE_INTRINSICS" \
    REFINE_PIXSFM="$REFINE_PIXSFM" \
    USE_SFM_DEPTH="$USE_SFM_DEPTH" \
    INCLUDE_DEPTH_DEBUG="$INCLUDE_DEPTH_DEBUG" \
    SAME_DIMENSIONS="$SAME_DIMENSIONS" \
    SKIP_IMAGE_PROCESSING="$SKIP_IMAGE_PROCESSING" \
    SKIP_COLMAP="$SKIP_COLMAP" \
    COLMAP_MODEL_PATH="$COLMAP_MODEL_PATH" \
    COLMAP_CMD="$COLMAP_CMD" \
    bash scripts/preprocess_nerfstudio.sh
    
    ### ZIP COLMAP DATA
    echo ""
    echo "🗜️ Zipping COLMAP directory from ORI_DIR..."

    COLMAP_ZIP_NAME="colmap_${BASENAME}.zip"
    COLMAP_ZIP_PATH="$ORI_DIR/$COLMAP_ZIP_NAME"

    mkdir -p "$EXPORT_DIR"
    (
      cd "$ORI_DIR"
      zip -r "$COLMAP_ZIP_NAME" "colmap" "transforms.json" > /dev/null
    )

    if [ ! -f "$COLMAP_ZIP_PATH" ]; then
      echo "❌ Failed to create zip archive"
      exit 1
    fi

    mv "$COLMAP_ZIP_PATH" "$EXPORT_DIR/"

    echo "✅ COLMAP zipped and moved:"
    echo "   $EXPORT_DIR/$COLMAP_ZIP_NAME"
    
    print_step_time "PREPROCESS" "$STEP_START"
fi

# ----------------------
# 2.5.AUTOMASK PIPELINE
# ----------------------

if [ "${AUTOMASK:-false}" = true ]; then
  echo ""
  echo "🧠 AUTOMASK enabled → generating foreground masks..."

  STEP_START=$(date +%s)

  MASK_DIR="$ORI_DIR/masks"
  mkdir -p "$MASK_DIR"

  # Generate masks
  echo "🎯 Running auto_image_masker..."

  python scripts/masking/auto_image_masker2.py \
    --input "$ORI_DIR/images" \
    --outdir "$MASK_DIR" \
    --sam_checkpoint "checkpoints/sam_vit_b.pth" \
    --model_type "vit_b" \
    --max_size 1024 \
    --points_per_side 8 \
    --margin_ratio 0.30

  if [ $? -ne 0 ]; then
    echo "❌ auto_image_masker failed"
    exit 1
  fi
  
  
  #  Inject masks
  if [ -f "$ORI_DIR/transforms.json" ]; then
    echo "🧩 Injecting masks into transforms.json..."

    cp "$ORI_DIR/transforms.json" "$ORI_DIR/transforms_backup.json"

    python scripts/masking/inject_masks_into_transforms.py \
      --transforms "$ORI_DIR/transforms.json" \
      --masks "$MASK_DIR" \
      --output "$ORI_DIR/transforms.json"

    if [ $? -ne 0 ]; then
      echo "❌ mask injection failed"
      exit 1
    fi

  else
    echo "❌ transforms.json not found, skipping mask injection"
    exit 1
  fi

  # Timing
  print_step_time "AUTOMASK" "$STEP_START"
fi

# ----------------------
# 3. TRAIN
# ----------------------

if [ "$SKIP_TRAINING" = true ]; then
  echo "⏩ Skipping training (config)"
else
    ### ESTIMATE NEAR / FAR FROM COLMAP
    COLMAP_DIR="$ORI_DIR/colmap/sparse/0"

    if [ ! -d "$COLMAP_DIR" ]; then
      echo "❌ COLMAP directory not found: $COLMAP_DIR"
      exit 1
    fi

    echo ""
    echo "📏 Estimating near/far planes from COLMAP..."

    ESTIMATE_SCRIPT="$SCRIPT_DIR/scripts/estimate_planes.py"

    if [ ! -f "$ESTIMATE_SCRIPT" ]; then
      echo "❌ Missing script: $ESTIMATE_SCRIPT"
      exit 1
    fi

    EST_OUTPUT=$(python3 "$ESTIMATE_SCRIPT" --input "$COLMAP_DIR")

    NEAR=$(echo "$EST_OUTPUT" | grep NEAR | cut -d= -f2)
    FAR=$(echo "$EST_OUTPUT" | grep FAR  | cut -d= -f2)

    # validation minimale
    if [[ -z "$NEAR" || -z "$FAR" || "$NEAR" == "nan" || "$FAR" == "nan" ]]; then
      echo "❌ Invalid near/far values"
      echo "$EST_OUTPUT"
      exit 1
    fi

    echo "✅ Estimated:"
    echo "   near = $NEAR"
    echo "   far  = $FAR"

    export COLLIDER_NEAR="$NEAR"
    export COLLIDER_FAR="$FAR"
    export ENABLE_COLLIDER="True"

    LATEST_RUN=$(ls -td "$OUTPUT_DIR"/ori/$MODEL/* 2>/dev/null | head -n 1 || true)

    if [ -n "$LATEST_RUN" ] && [ -d "$LATEST_RUN/nerfstudio_models" ]; then
        echo "⏩ Skipping training"
        else
        echo ""
        echo ""
        echo "🧠 Training..."

        STEP_START=$(date +%s)

        echo MAX_JOBS: $MAX_JOBS


        HTTP_PROXY="$HTTP_PROXY" \
        HTTPS_PROXY="$HTTPS_PROXY" \
        NO_PROXY="$NO_PROXY" \
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
        MAX_GAUSS_RATIO="$MAX_GAUSS_RATIO" \
        DENSIFY_GRAD_THRESH="$DENSIFY_GRAD_THRESH" \
        CULL_ALPHA_THRESH="$CULL_ALPHA_THRESH" \
        CULL_SCREEN_SIZE="$CULL_SCREEN_SIZE" \
        SPLIT_SCREEN_SIZE="$SPLIT_SCREEN_SIZE" \
        STOP_SPLIT_AT="$STOP_SPLIT_AT" \
        CULL_SCALE_THRESH="$CULL_SCALE_THRESH" \
        RESET_ALPHA_EVERY="$RESET_ALPHA_EVERY" \
        USE_SCALE_REGULARIZATION="$USE_SCALE_REGULARIZATION" \
        SSIM_LAMBDA="$SSIM_LAMBDA" \
        MAX_GAUSSIANS="$MAX_GAUSSIANS" \
        COLLIDER_NEAR="$COLLIDER_NEAR" \
        COLLIDER_FAR="$COLLIDER_FAR" \
        ENABLE_COLLIDER="$ENABLE_COLLIDER" \
        USE_BILATERAL_GRID="$USE_BILATERAL_GRID" \
        bash scripts/train.sh

        print_step_time "TRAINING" "$STEP_START"
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
    echo "📦 Existing PLY found: $PLY_FOUND, backing it up into ${PLY_FOUND}.bkp"
    # echo "⏩ Skipping export (PLY already exists)"
    cp $PLY_FOUND ${PLY_FOUND}.bkp
  fi

    echo "🚀 Exporting model in $EXPORT_DIR..."
    STEP_START=$(date +%s)
    
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
    
    ### VALIDATION
    
    PLY_FILE=$(find "$OUTPUT_DIR" -type f -name "*.ply" | head -n 1)

    if [[ -f "$PLY_FILE" ]]; then
      echo "✅ PLY export successful: $PLY_FILE"
      print_step_time "EXPORT" "$STEP_START"
    else
      echo "❌ PLY export failed"
      exit 1
    fi
    
    ### TRANSFER ARCHIVE OF THE TRAINING TO EXPORT_DIR
    
    LATEST_ZIP=$(ls -t "$OUTPUT_DIR"/*.zip 2>/dev/null | head -n 1)

    if [ -z "$LATEST_ZIP" ]; then
      echo "⚠️ No .zip file found in $OUTPUT_DIR"
    else
      mv "$LATEST_ZIP" "$EXPORT_DIR/" && \
      if [ -f "$EXPORT_DIR/$(basename "$LATEST_ZIP")" ]; then
        echo "📦 Moved $(basename "$LATEST_ZIP") to $EXPORT_DIR"
      else
        echo "❌ Failed to move $(basename "$LATEST_ZIP")" >&2
      fi
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

STEP_START=$(date +%s)

# ======================
# RUN RESOLUTION
# ======================
RUN_ROOT="$OUTPUT_DIR/$MODEL"

LATEST_RUN=$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)

if [ -z "$LATEST_RUN" ] || [ ! -d "$LATEST_RUN" ]; then
  echo "[ERROR] No run found in $RUN_ROOT"
  exit 1
fi

TRANSFORM_FILE="$LATEST_RUN/dataparser_transforms.json"

if [ ! -f "$TRANSFORM_FILE" ]; then
  echo "[ERROR] Transform file not found: $TRANSFORM_FILE"
  exit 1
fi

# ======================
# INPUT VALIDATION
# ======================
if [[ ! -f "$PLY_FILE" ]]; then
  echo "❌ PLY not found: $PLY_FILE"
  exit 1
fi

COLMAP_POINTS="$ORI_DIR/colmap/sparse/0/points3D.bin"

if [[ ! -f "$COLMAP_POINTS" ]]; then
  echo "❌ COLMAP points not found: $COLMAP_POINTS"
  exit 1
fi

# ======================
# CLEANING STEP
# ======================
python scripts/cleaning/clean-ply.py \
  --in-ply "$PLY_FILE" \
  --points "$COLMAP_POINTS" \
  --out-ply "$EXPORT_DIR/${BASENAME}.ply" \
  --transform "$TRANSFORM_FILE"

print_step_time "CLEAN PLY" "$STEP_START"
