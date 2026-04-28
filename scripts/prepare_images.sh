#!/bin/bash
set -e

# ======================
# LOAD CONFIG
# ======================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config/config.sh"

SRC_DIR="$1"
OUT_DIR="${2:-dataset/images}"

# ======================
# CHECKS
# ======================
if [ -z "$SRC_DIR" ]; then
  echo "❌ Usage: $0 <source_images_dir> [output_dir]"
  exit 1
fi

if [ ! -d "$SRC_DIR" ]; then
  echo "❌ Source directory not found: $SRC_DIR"
  exit 1
fi

# ======================
# PREPARE OUTPUT
# ======================

echo "🧹 Preparing output directory: $OUT_DIR"

if [ -d "$OUT_DIR" ]; then
  rm -rf "$OUT_DIR"
fi

mkdir -p "$OUT_DIR"

echo "🖼️ Preparing images from: $SRC_DIR"
echo "📁 Output: $OUT_DIR"

# ======================
# COPY + RENAME (SAFE LOOP)
# ======================

COUNT=1

while IFS= read -r FILE; do
  INDEX=$(printf "%05d" "$COUNT")

  # convert/normalize every image
  ffmpeg -hide_banner -loglevel error -y \
    -i "$FILE" \
    -vf "scale=1280:-1" \
    "$OUT_DIR/frame_${INDEX}.png"

  COUNT=$((COUNT + 1))

done < <(find "$SRC_DIR" -maxdepth 1 -type f \
  \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.tif" -o -iname "*.tiff" \) \
  | sort)

# ======================
# VALIDATION
# ======================
TOTAL=$(find "$OUT_DIR" -maxdepth 1 -name 'frame_*.png' | wc -l | tr -d ' ')

if [ "$TOTAL" -eq 0 ]; then
  echo "❌ No supported images found in: $SRC_DIR"
  exit 1
fi

echo "✅ Prepared $TOTAL images"

# ======================
# FINAL CLEANUP (STRICT)
# ======================

echo "🧼 Removing non-frame files from output directory..."

find "$OUT_DIR" -type f ! -name "frame_*.png" -delete

# safety check
EXTRA_FILES=$(find "$OUT_DIR" -type f ! -name "frame_*.png" | wc -l | tr -d ' ')

if [ "$EXTRA_FILES" -ne 0 ]; then
  echo "❌ Cleanup failed: some non-frame files remain"
  find "$OUT_DIR" -type f ! -name "frame_*.png"
  exit 1
fi

echo "✅ Output directory clean (only frame_*.png present)"
