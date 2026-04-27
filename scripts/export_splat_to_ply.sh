#!/bin/bash
set -e

# ======================
# CHECKS
# ======================
if [[ -z "$EXPORT_DIR" || -z "$OUTPUT_DIR" ]]; then
  echo "❌ Missing required environment variables:"
  echo "   EXPORT_DIR=$EXPORT_DIR"
  echo "   OUTPUT_DIR=$OUTPUT_DIR"
  exit 1
fi

if [[ ! -d "$EXPORT_DIR" ]]; then
  echo "❌ Export directory does not exist: $EXPORT_DIR"
  exit 1
fi

if [[ ! -d "$OUTPUT_DIR" ]]; then
  echo "❌ Output directory does not exist: $OUTPUT_DIR"
  exit 1
fi


SPLAT_ROOT="$OUTPUT_DIR/splatfacto"

if [ ! -d "$SPLAT_ROOT" ]; then
  echo "❌ splatfacto folder not found: $SPLAT_ROOT"
  exit 1
fi

# ======================
# FIND LATEST RUN
# ======================
RUN_DIR=$(ls -dt "$SPLAT_ROOT"/* 2>/dev/null | head -n 1)

if [ -z "$RUN_DIR" ]; then
  echo "❌ No splatfacto runs found in $SPLAT_ROOT"
  exit 1
fi

# ======================
# CONFIG
# ======================
CONFIG="$RUN_DIR/config.yml"

if [ ! -f "$CONFIG" ]; then
  echo "❌ config.yml not found:"
  echo "$RUN_DIR"
  exit 1
fi

mkdir -p "$EXPORT_DIR"

echo "────────────────────────────────────────────"
echo "📦 ROOT DIR       : $ROOT_DIR"
echo "📦 SPLAT ROOT      : $SPLAT_ROOT"
echo "📦 RUN DIR        : $RUN_DIR"
echo "📄 CONFIG         : $CONFIG"
echo "📁 EXPORT DIR     : $EXPORT_DIR"
echo "────────────────────────────────────────────"

# ======================
# ENV FLAGS
# ======================
# export TORCHDYNAMO_DISABLE=1
# export OMP_NUM_THREADS=1

# ======================
# DEBUG MODEL
# ======================
echo "📊 Checking checkpoint..."
ls "$RUN_DIR/nerfstudio_models" || echo "⚠️ No models folder"

# ======================
# EXPORT PLY
# ======================
echo "🚀 Exporting Gaussian Splat (.ply)..."

ns-export gaussian-splat \
  --load-config "$CONFIG" \
  --output-dir "$EXPORT_DIR"

PLY_FILE=$(find "$EXPORT_DIR" -name "*.ply" | head -n 1)

if [ -z "$PLY_FILE" ]; then
  echo "❌ Gaussian PLY export failed"
  exit 1
fi

echo "✅ PLY exported:"
echo "   $PLY_FILE"

# ======================
# CLEAN PLY
# ======================
echo "🧹 Cleaning Gaussian Splat..."

CLEANED_PLY="${PLY_FILE%.ply}_cleaned.ply"

python3 "$ROOT_DIR/scripts/clean_gaussian_ply.py" \
    "$PLY_FILE" \
    "$CLEANED_PLY" \
    --nb-neighbors 32 \
    --std-ratio 1.5 \
    --dbscan-eps 0.05 \
    --dbscan-min-points 50

if [[ ! -f "$CLEANED_PLY" ]]; then
    echo "❌ PLY cleaning failed"
    exit 1
fi

echo "✅ Cleaned PLY:"
echo "   $CLEANED_PLY"
