#!/bin/bash
set -e

# ======================
# LOAD CONFIG (CRITICAL)
# ======================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config/config.sh"

EXPORT_DIR="$1"
OUTPUT_DIR="$2"

# ======================
# SAFETY CHECKS
# ======================
if [ -z "$EXPORT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "❌ Usage: export.sh <EXPORT_DIR> <OUTPUT_DIR>"
  exit 1
fi

# ======================
# FIND LATEST NERFACTO RUN (IMPORTANT FIX)
# ======================
RUN_DIR=$(find "$OUTPUT_DIR" -type d -path "*/nerfacto/*" | sort | tail -n 1)

if [ -z "$RUN_DIR" ]; then
  echo "❌ No nerfacto run directory found in $OUTPUT_DIR"
  exit 1
fi

CONFIG=$(find "$RUN_DIR" -maxdepth 2 -name "config.yml" | head -n 1)

# ======================
# CHECK CONFIG
# ======================
if [ -z "$CONFIG" ] || [ ! -f "$CONFIG" ]; then
  echo "❌ No config.yml found in run directory:"
  echo "$RUN_DIR"
  exit 1
fi

mkdir -p "$EXPORT_DIR"

echo "📦 Using run dir: $RUN_DIR"
echo "📦 Using config: $CONFIG"
echo "📁 Export dir: $EXPORT_DIR"
echo "⚙️ Export mode: $EXPORT_MODE"

# ======================
# EXPORT MODE RESOLUTION
# ======================
case "$EXPORT_MODE" in
  fast)
    NUM_POINTS="$EXPORT_NUM_POINTS_FAST"
    NORMAL_METHOD="$EXPORT_NORMALS_FAST"
    REMOVE_OUTLIERS="$EXPORT_REMOVE_OUTLIERS_FAST"
    DOWNSAMPLE="$EXPORT_DOWNSAMPLE_FAST"
    ;;

  quality)
    NUM_POINTS="$EXPORT_NUM_POINTS_QUALITY"
    NORMAL_METHOD="$EXPORT_NORMALS_QUALITY"
    REMOVE_OUTLIERS="$EXPORT_REMOVE_OUTLIERS_QUALITY"
    DOWNSAMPLE="$EXPORT_DOWNSAMPLE_QUALITY"
    ;;

  *)
    NUM_POINTS="$EXPORT_NUM_POINTS_BALANCED"
    NORMAL_METHOD="$EXPORT_NORMALS_BALANCED"
    REMOVE_OUTLIERS="$EXPORT_REMOVE_OUTLIERS_BALANCED"
    DOWNSAMPLE="$EXPORT_DOWNSAMPLE_BALANCED"
    ;;
esac

# ======================
# EXPORT
# ======================
echo "🚀 Export settings:"
echo "   - points: $NUM_POINTS"
echo "   - normals: $NORMAL_METHOD"
echo "   - downsample: $DOWNSAMPLE"
echo "   - outliers: $REMOVE_OUTLIERS"

ns-export pointcloud \
  --load-config "$CONFIG" \
  --output-dir "$EXPORT_DIR" \
  --num-points "$NUM_POINTS" \
  --normal-method "$NORMAL_METHOD" \
  --downsample-factor "$DOWNSAMPLE" \
  --remove-outliers "$REMOVE_OUTLIERS"
