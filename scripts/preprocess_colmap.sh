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

# CPU SAFE MODE
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
export OPENCV_LOG_LEVEL=ERROR
export XDG_RUNTIME_DIR=/tmp/runtime-root
export CUDA_VISIBLE_DEVICES=""
export LIBGL_ALWAYS_SOFTWARE=1

LOG_FILE="/tmp/colmap_error.log"

echo "🧭 COLMAP preprocessing"
echo "📁 Input: $DATA_DIR"
echo "📁 Output: $OUTPUT_DIR"
echo "⚙️ Device: $DEVICE"

SKIP_FLAG=""
if [[ "$SKIP_IMG" == "true" || "$SKIP_IMG" == "1" ]]; then
  SKIP_FLAG="--skip-image-processing"
fi

# ======================
# 🚀 CPU MODE → DIRECT COLMAP (NO NERFSTUDIO)
# ======================
if [[ "$DEVICE" == "cpu" ]]; then

  echo "🧠 CPU MODE → using native COLMAP pipeline"

  DB="$OUTPUT_DIR/database.db"
  SPARSE="$OUTPUT_DIR/sparse"
  mkdir -p "$SPARSE"

    set +e

    echo "📌 feature_extractor"
    colmap feature_extractor \
    --database_path "$DB" \
    --image_path "$DATA_DIR" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV \
    --SiftExtraction.use_gpu 0 \
    >> "$LOG_FILE" 2>&1

    FEAT_STATUS=$?

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

    echo "📌 mapper"
    colmap mapper \
    --database_path "$DB" \
    --image_path "$DATA_DIR" \
    --output_path "$SPARSE" \
    >> "$LOG_FILE" 2>&1

    MAP_STATUS=$?
    
    

    TRANSFORMS="$OUTPUT_DIR/transforms.json"

    if [ ! -f "$TRANSFORMS" ]; then
      echo "📦 Generating transforms.json from COLMAP..."

      ns-process-data images \
        --data "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --skip-colmap \
        --sfm-tool colmap \
        --camera-type perspective \
        >> "$LOG_FILE" 2>&1

      GEN_STATUS=$?
      echo "✅ transforms.json created at $TRANSFORMS"
    else
      echo "⏩ transforms.json already exists → skipping generation"
    fi

    set -e
    

    STATUS=0

    if [ $FEAT_STATUS -ne 0 ]; then
      echo "❌ feature_extractor failed"
      STATUS=1
    fi

    if [ $MATCH_STATUS -ne 0 ]; then
      echo "❌ matcher failed"
      STATUS=1
    fi

    if [ $MAP_STATUS -ne 0 ]; then
      echo "❌ mapper failed"
      STATUS=1
    fi

    if [ $STATUS -eq 0 ]; then
      echo "✅ COLMAP pipeline SUCCESS"
    else
      echo "❌ COLMAP pipeline FAILED"
    fi
    
    
    if [ $GEN_STATUS -ne 0 ]; then
        echo "❌ Failed to generate transforms.json"
        STATUS=1
    fi

else

# ======================
# GPU MODE → NERFSTUDIO PIPELINE
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
# ERROR HANDLING
# ======================
echo STATUS: $STATUS

STATUS=${STATUS:0}
if [ "$STATUS" -ne 0 ]; then
  echo ""
  echo "❌ ❌ ❌ PIPELINE FAILED ❌ ❌ ❌"
  echo "📄 Log: $LOG_FILE"
  echo ""
  echo "🔍 Last errors:"
  tail -n 30 "$LOG_FILE"
  exit 1
fi

echo "✅ COLMAP done successfully"
