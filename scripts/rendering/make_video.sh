#!/bin/bash
set -euo pipefail

# ============================================================
# MAKE VIDEO FROM IMAGE SEQUENCE (FFMPEG WRAPPER)
# ============================================================
# Usage examples:
#
#   1) Default (expects images/scene*.png → perfume.mp4 at 30 FPS)
#       bash make_video.sh
#
#   2) Change FPS and output file
#       FPS=15 OUTPUT=out.mp4 bash make_video.sh
#
#   3) Different input folder
#       INPUT_DIR=frames OUTPUT=video.mp4 bash make_video.sh
#
#   4) Different filename pattern
#       PATTERN="img*.png" bash make_video.sh
#
# Environment variables:
#   INPUT_DIR : folder containing images (default: images)
#   PATTERN   : glob pattern (default: scene*.png)
#   OUTPUT    : output video file (default: perfume.mp4)
#   FPS       : frame rate (default: 30)
#   CRF       : quality (default: 18, lower = better quality)
#   PRESET    : encoding speed/quality tradeoff (default: slow)
#
# Notes:
# - Uses glob mode to handle non-contiguous frames safely
# - Forces yuv420p for maximum compatibility


# Exemple avec des images png dans le sous-dossier images:

# INPUT_DIR=images \
# PATTERN="scene%05d.png" \
# FPS=30 \
# OUTPUT=perfume.mp4 \
# bash make_video.sh
# ============================================================

INPUT_DIR="${INPUT_DIR:-images}"
PATTERN="${PATTERN:-scene*.png}"
OUTPUT="${OUTPUT:-perfume.mp4}"

FPS="${FPS:-30}"
CRF="${CRF:-18}"
PRESET="${PRESET:-slow}"

echo "🎬 Building video..."
echo "📁 Input : $INPUT_DIR/$PATTERN"
echo "🎞 FPS   : $FPS"
echo "📦 Output: $OUTPUT"

ffmpeg \
  -framerate "$FPS" \
  -pattern_type glob \
  -i "$INPUT_DIR/$PATTERN" \
  -c:v libx264 \
  -crf "$CRF" \
  -preset "$PRESET" \
  -pix_fmt yuv420p \
  -r "$FPS" \
  -movflags +faststart \
  "$OUTPUT"

echo "✅ Done: $OUTPUT"
