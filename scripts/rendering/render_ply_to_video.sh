#!/bin/bash
set -euo pipefail

# -----------------------------
# SCRIPT DIR (where .sh lives)
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BLENDER_SCRIPT="$SCRIPT_DIR/orbit_spiral.py"
RENDER_SCRIPT="$SCRIPT_DIR/render_gsplat_video.py"

# -----------------------------
# ENV VARIABLES
# -----------------------------
PLY_PATH="${PLY_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

# -----------------------------
# VALIDATION
# -----------------------------
if [ -z "$PLY_PATH" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "❌ Missing environment variables"
  echo "Required:"
  echo "  export PLY_FILE=..."
  echo "  export OUTPUT_DIR=..."
  exit 1
fi

if [ ! -f "$PLY_PATH" ]; then
  echo "❌ PLY file not found:"
  echo "$PLY_PATH"
  exit 1
fi

if [ ! -f "$BLENDER_SCRIPT" ]; then
  echo "❌ Missing Blender script:"
  echo "$BLENDER_SCRIPT"
  exit 1
fi

if [ ! -f "$RENDER_SCRIPT" ]; then
  echo "❌ Missing render script:"
  echo "$RENDER_SCRIPT"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=============================="
echo "SCRIPT DIR : $SCRIPT_DIR"
echo "PLY        : $PLY_PATH"
echo "OUTPUT     : $OUTPUT_DIR"
echo "=============================="

# -----------------------------
# STEP 1 - BLENDER TRAJECTORY
# -----------------------------
echo "🚀 Step 1: Generating trajectory (Blender)..."

blender -b -P "$BLENDER_SCRIPT" -- \
  --ply "$PLY_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --trajectory spiral \
  --turns 2 \
  --frames 240

echo "✅ Trajectory generated"

TRANSFORMS_PATH="$OUTPUT_DIR/transforms.json"

if [ ! -f "$TRANSFORMS_PATH" ]; then
  echo "❌ Missing output transforms.json"
  echo "$TRANSFORMS_PATH"
  exit 1
fi

# -----------------------------
# STEP 2 - GSPLAT RENDER
# -----------------------------
echo "🚀 Step 2: Rendering with gsplat..."

python "$RENDER_SCRIPT" \
  --ply "$PLY_PATH" \
  --transforms "$TRANSFORMS_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --fps 24 \
  --width 1280 \
  --height 720

VIDEO_PATH="$OUTPUT_DIR/video.mp4"

if [ ! -f "$VIDEO_PATH" ]; then
  echo "❌ Video not generated"
  exit 1
fi

echo "=============================="
echo "🎬 DONE"
echo "VIDEO: $VIDEO_PATH"
echo "=============================="
