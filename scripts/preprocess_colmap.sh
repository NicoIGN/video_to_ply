#!/bin/bash

# ======================
# LOAD CONFIG
# ======================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config/config.sh"

# ======================
# INPUTS
# ======================
DATA_DIR=${1:-dataset/images}
OUTPUT_DIR=${2:-dataset/ori}

# ======================
# CHECKS
# ======================
if [ ! -d "$DATA_DIR" ]; then
  echo "❌ Data dir not found: $DATA_DIR"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ======================
# CONFIG
# ======================
MATCHING="$MATCHING_METHOD"
SFMT_TOOL="$SFMT_TOOL"
NUM_DOWNSCALES="$NUM_DOWNSCALES"
SKIP_IMG="$SKIP_IMAGE_PROCESSING"
DEVICE="${DEVICE:-cpu}"
CLEAN_ARTEFACTS="${CLEAN_ARTEFACTS:-false}"

# ======================
# CPU SAFE MODE
# ======================
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
export OPENCV_LOG_LEVEL=ERROR
export XDG_RUNTIME_DIR=/tmp/runtime-root
export CUDA_VISIBLE_DEVICES=""
export LIBGL_ALWAYS_SOFTWARE=1

# 👉 IMPORTANT debug Nerfstudio
export NS_DEBUG=1
export NS_LOG_LEVEL=debug

LOG_FILE="/tmp/colmap_error.log"

if [ -e "$LOG_FILE" ]; then
    rm "$LOG_FILE"
fi

echo "🧭 COLMAP preprocessing"
echo "📁 Input: $DATA_DIR"
echo "📁 Output: $OUTPUT_DIR"
echo "⚙️ Device: $DEVICE"
echo "🧹 CLEAN_ARTEFACTS=$CLEAN_ARTEFACTS"

SKIP_FLAG=""
if [[ "$SKIP_IMG" == "true" || "$SKIP_IMG" == "1" ]]; then
  SKIP_FLAG="--skip-image-processing"
fi


# ======================
# 🚀 CPU MODE → COLMAP PIPELINE
# ======================
if [[ "$DEVICE" == "cpu" ]]; then

  echo "🧠 CPU MODE → native COLMAP pipeline"
  
    # ======================
    # ARTEFACT DETECTION
    # ======================
    DB="$OUTPUT_DIR/colmap/database.db"
    SPARSE="$OUTPUT_DIR/colmap/sparse"
    TRANSFORMS="$OUTPUT_DIR/transforms.json"

    HAS_COLMAP_ARTEFACTS=false
    HAS_TRANSFORMS=false

    if [ -f "$DB" ] && [ -d "$SPARSE/0" ] && [ -f "$SPARSE/0/images.bin" ]; then
      HAS_COLMAP_ARTEFACTS=true
    fi

    if [ -f "$TRANSFORMS" ]; then
      HAS_TRANSFORMS=true
    fi

    echo "📦 COLMAP cache: $HAS_COLMAP_ARTEFACTS"
    echo "📦 transforms cache: $HAS_TRANSFORMS"
  
  
  

  mkdir -p "$SPARSE"

  set +e

  # ======================
  # FEATURE EXTRACTOR
  # ======================
  if [[ "$CLEAN_ARTEFACTS" == "true" || "$HAS_COLMAP_ARTEFACTS" == "false" ]]; then
    echo "📌 feature_extractor"

    colmap feature_extractor \
      --database_path "$DB" \
      --image_path "$DATA_DIR" \
      --ImageReader.single_camera 1 \
      --ImageReader.camera_model OPENCV \
      --SiftExtraction.use_gpu 0 \
      >> "$LOG_FILE" 2>&1

    FEAT_STATUS=$?
  else
    echo "⏩ Skipping feature_extractor (cache)"
    FEAT_STATUS=0
  fi

  # ======================
  # MATCHER
  # ======================
  if [[ "$CLEAN_ARTEFACTS" == "true" || "$HAS_COLMAP_ARTEFACTS" == "false" ]]; then
    echo "📌 matcher"

    if [[ "$MATCHING" == "sequential" ]]; then
      colmap sequential_matcher \
        --database_path "$DB" \
        --SiftMatching.use_gpu 0 \
        >> "$LOG_FILE" 2>&1
    else
      colmap exhaustive_matcher \
        --database_path "$DB" \
        --SiftMatching.use_gpu 0 \
        >> "$LOG_FILE" 2>&1
    fi

    MATCH_STATUS=$?
  else
    echo "⏩ Skipping matcher (cache)"
    MATCH_STATUS=0
  fi

  # ======================
  # MAPPER
  # ======================
  if [[ "$CLEAN_ARTEFACTS" == "true" || "$HAS_COLMAP_ARTEFACTS" == "false" ]]; then
    echo "📌 mapper"

    colmap mapper \
      --database_path "$DB" \
      --image_path "$DATA_DIR" \
      --output_path "$SPARSE" \
      >> "$LOG_FILE" 2>&1

    MAP_STATUS=$?
  else
    echo "⏩ Skipping mapper (cache)"
    MAP_STATUS=0
  fi

  # ======================
  # DEBUG COLMAP STRUCTURE
  # ======================
  echo "────────────────────────────────────────────"
  echo "🔍 COLMAP STRUCTURE CHECK"
  echo "DATA_DIR   = $DATA_DIR"
  echo "OUTPUT_DIR = $OUTPUT_DIR"
  echo "DB         = $DB"
  echo "SPARSE     = $SPARSE"
  ls -R "$SPARSE"
  echo "────────────────────────────────────────────"

# ======================
# TRANSFORMS GENERATION (NO NERFSTUDIO)
# ======================
if [[ "$CLEAN_ARTEFACTS" == "true" || ! -f "$TRANSFORMS" ]]; then

  echo "📦 Generating transforms.json from COLMAP (external script)"

  SCRIPT_PATH="$SCRIPT_DIR/colmap_to_transforms.py"

  if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ Missing Python script: $SCRIPT_PATH"
    exit 1
  fi

  echo "🚀 Running COLMAP → transforms exporter"

  python3 "$SCRIPT_PATH" "$OUTPUT_DIR"
  GEN_STATUS=$?

  # ======================
  # VALIDATION STEP
  # ======================
  if [ $GEN_STATUS -ne 0 ]; then
    echo "❌ Python export script failed (exit code $GEN_STATUS)"
    echo "📄 Check logs above"
    exit 1
  fi

  if [ ! -f "$TRANSFORMS" ]; then
    echo "❌ transforms.json NOT FOUND at expected path:"
    echo "   $TRANSFORMS"
    exit 1
  fi

  # check file is not empty / corrupted
  if [ ! -s "$TRANSFORMS" ]; then
    echo "❌ transforms.json is empty or corrupted"
    exit 1
  fi

  echo "✅ transforms.json generated successfully: $TRANSFORMS"

else
  echo "⏩ transforms.json already exists"
  GEN_STATUS=0
fi

set -e

  # ======================
  # STATUS
  # ======================
  STATUS=0

  [[ $FEAT_STATUS -ne 0 ]] && STATUS=1
  [[ $MATCH_STATUS -ne 0 ]] && STATUS=1
  [[ $MAP_STATUS -ne 0 ]] && STATUS=1
  [[ $GEN_STATUS -ne 0 ]] && STATUS=1

  if [ $STATUS -eq 0 ]; then
    echo "✅ COLMAP pipeline SUCCESS"
  else
    echo "❌ COLMAP pipeline FAILED"
  fi

else

# ======================
# GPU MODE
# ======================

  echo "🚀 GPU MODE → ns-process-data"

  set +e

  ns-process-data images \
    --data "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --camera-type perspective \
    --sfm-tool "$SFMT_TOOL" \
    --matching-method "$MATCHING" \
    --num-downscales "$NUM_DOWNSCALES" \
    $SKIP_FLAG \
    2> "$LOG_FILE"

  STATUS=$?
  set -e

fi

# ======================
# FINAL CHECK
# ======================
echo "STATUS: $STATUS"

if [ "$STATUS" -ne 0 ]; then
  echo "❌ ❌ ❌ PIPELINE FAILED ❌ ❌ ❌"
  echo "📄 Log: $LOG_FILE"
  tail -n 30 "$LOG_FILE"
  exit 1
fi

echo "✅ COLMAP done successfully"
