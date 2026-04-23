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
# CONFIG VARIABLES
# ======================
MATCHING="$MATCHING_METHOD"
SFMT_TOOL="$SFMT_TOOL"
NUM_DOWNSCALES="$NUM_DOWNSCALES"
SKIP_IMG="$SKIP_IMAGE_PROCESSING"
DEVICE="${DEVICE:-cpu}"

# ======================
# SAFE ENV (cross-platform)
# ======================
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
export OPENCV_LOG_LEVEL=ERROR
export XDG_RUNTIME_DIR=/tmp/runtime-root
export CUDA_VISIBLE_DEVICES=""
export LIBGL_ALWAYS_SOFTWARE=1

# ======================
# LOG
# ======================
echo "🧭 Running COLMAP preprocessing"
echo "📁 Input: $DATA_DIR"
echo "📁 Output: $OUTPUT_DIR"
echo "🔗 Matching: $MATCHING"
echo "🧱 SfM tool: $SFMT_TOOL"
echo "📉 Downscale: $NUM_DOWNSCALES"
echo "🚫 Skip image processing: $SKIP_IMG"
echo "⚙️ Device (NERF only): $DEVICE"

# ======================
# SKIP FLAG
# ======================
SKIP_FLAG=""
if [[ "$SKIP_IMG" == "true" || "$SKIP_IMG" == "1" ]]; then
  SKIP_FLAG="--skip-image-processing"
fi

# ======================
# EXEC MODE (FIXED)
# ======================
LOG_FILE="/tmp/colmap_error.log"

echo "⚙️ COLMAP SIFT: CPU MODE FORCED"

# detect OS
RUN_PREFIX=""
if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "🍏 macOS detected → no xvfb-run"
  RUN_PREFIX=""
else
  echo "🐧 Linux detected → using xvfb-run"
  RUN_PREFIX="xvfb-run -a"
fi

# ======================
# RUN
# ======================
set +e

$RUN_PREFIX ns-process-data images \
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

# ======================
# ERROR HANDLING
# ======================
if [ $STATUS -ne 0 ]; then
  echo ""
  echo "❌ ❌ ❌ COLMAP FAILED ❌ ❌ ❌"
  echo "📄 Log saved to: $LOG_FILE"
  echo ""
  echo "🔍 Last errors:"
  tail -n 30 "$LOG_FILE"
  exit 1
fi

echo "✅ COLMAP done successfully"
