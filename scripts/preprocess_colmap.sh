#!/bin/bash
set -e

# ======================
# LOAD CONFIG (CRITICAL)
# ======================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config/config.sh"


# ======================
# INPUTS (only override if needed)
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
# PARAMS FROM CONFIG
# ======================
MATCHING="${MATCHING_METHOD}"
SFMT_TOOL="${SFMT_TOOL}"
NUM_DOWNSCALES="${NUM_DOWNSCALES}"
SKIP_IMG="${SKIP_IMAGE_PROCESSING}"


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


# ======================
# BUILD FLAGS
# ======================
SKIP_FLAG=""
if [ "$SKIP_IMG" = true ]; then
  SKIP_FLAG="--skip-image-processing"
fi


# ======================
# RUN PIPELINE
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
