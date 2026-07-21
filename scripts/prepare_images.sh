#!/bin/bash
set -euo pipefail

# ======================
# PATHS
# ======================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC_DIR="${1:-}"
OUT_DIR="${2:-dataset/images}"
MAX_SIZE="${3:-1280}"   # largeur max (par défaut 1280)

# ======================
# CHECKS
# ======================
if [[ -z "$SRC_DIR" ]]; then
  echo "❌ Usage: $0 <source_images_dir> [output_dir] [max_size]"
  exit 1
fi

if [[ ! -d "$SRC_DIR" ]]; then
  echo "❌ Source directory not found: $SRC_DIR"
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "❌ ffmpeg is required but not found in PATH"
  exit 1
fi

if ! [[ "$MAX_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "❌ max_size must be a strictly positive integer, got: $MAX_SIZE"
  exit 1
fi

# ======================
# PREPARE OUTPUT
# ======================
echo "🧹 Preparing output directory: $OUT_DIR"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "🖼️ Preparing images from: $SRC_DIR"
echo "📁 Output: $OUT_DIR"
echo "📏 Max width: $MAX_SIZE"

# ======================
# DISCOVER INPUTS
# ======================
mapfile -d '' SOURCE_IMAGES < <(
  find "$SRC_DIR" -maxdepth 1 -type f \
    \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.tif" -o -iname "*.tiff" \) \
    -print0 | sort -z
)

TOTAL_SOURCE="${#SOURCE_IMAGES[@]}"

if [[ "$TOTAL_SOURCE" -eq 0 ]]; then
  echo "❌ No supported images found in: $SRC_DIR"
  exit 1
fi

# ======================
# COPY + RENAME + NORMALIZE
# ======================
COUNT=1

for FILE in "${SOURCE_IMAGES[@]}"; do
  INDEX=$(printf "%05d" "$COUNT")
  DEST="$OUT_DIR/frame_${INDEX}.png"

  echo "   → $(basename "$FILE") -> $(basename "$DEST")"

  ffmpeg -hide_banner -loglevel error -y \
    -i "$FILE" \
    -vf "scale=${MAX_SIZE}:${MAX_SIZE}:force_original_aspect_ratio=decrease" \
    "$DEST"

  if [[ ! -f "$DEST" ]]; then
    echo "❌ Failed to create output image: $DEST"
    exit 1
  fi

  COUNT=$((COUNT + 1))
done

# ======================
# VALIDATION
# ======================
TOTAL=$(find "$OUT_DIR" -maxdepth 1 -type f -name 'frame_*.png' | wc -l | tr -d ' ')

if [[ "$TOTAL" -eq 0 ]]; then
  echo "❌ No output images were created"
  exit 1
fi

if [[ "$TOTAL" -ne "$TOTAL_SOURCE" ]]; then
  echo "❌ Output count mismatch"
  echo "   source images : $TOTAL_SOURCE"
  echo "   output images : $TOTAL"
  exit 1
fi

echo "✅ Prepared $TOTAL images"

# ======================
# FINAL CLEANUP
# ======================
echo "🧼 Removing non-frame files from output directory..."

find "$OUT_DIR" -maxdepth 1 -type f ! -name "frame_*.png" -delete

EXTRA_FILES=$(find "$OUT_DIR" -maxdepth 1 -type f ! -name "frame_*.png" | wc -l | tr -d ' ')

if [[ "$EXTRA_FILES" -ne 0 ]]; then
  echo "❌ Cleanup failed: some non-frame files remain"
  find "$OUT_DIR" -maxdepth 1 -type f ! -name "frame_*.png"
  exit 1
fi

echo "✅ Output directory clean (only frame_*.png present)"
