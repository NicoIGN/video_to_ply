#!/bin/bash
set -e

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
# 🧠 COLMAP SAFE MODE (COLAB FIX)
# ======================
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
export OPENCV_LOG_LEVEL=ERROR
export XDG_RUNTIME_DIR=/tmp/runtime-root

# 🔥 HARD FORCE: NO GPU FOR COLMAP (IMPORTANT)
export COLMAP_SIFT_GPU=0

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
# SKIP IMAGE PROCESSING FLAG
# ======================
SKIP_FLAG=""
if [ "$SKIP_IMG" = "true" ] || [ "$SKIP_IMG" = true ]; then
  SKIP_FLAG="--skip-image-processing"
fi

# ======================
# 🧠 IMPORTANT NOTE
# ======================
echo "⚙️ COLMAP SIFT: FORCED CPU MODE (GPU disabled for stability)"

# ======================
# PIPELINE EXECUTION
# ======================
ns-process-data images \
  --data "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --camera-type perspective \
  --sfm-tool "$SFMT_TOOL" \
  --matching-method "$MATCHING" \
  --num-downscales "$NUM_DOWNSCALES" \
  $SKIP_FLAG

echo "✅ COLMAP done"
