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
mkdir -p "$OUT_DIR"

# Remove previous extracted/prepared frames
rm -f "$OUT_DIR"/frame_*.png

echo "🖼️ Preparing images from: $SRC_DIR"
echo "📁 Output: $OUT_DIR"

# ======================
# COPY + NORMALIZE
# ======================
COUNT=1
FOUND=0

find "$SRC_DIR" -maxdepth 1 -type f \
  \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) \
  | sort \
  | while read -r FILE; do
      FOUND=1
      printf -v INDEX "%05d" "$COUNT"

      ffmpeg -hide_banner -loglevel error -y \
        -i "$FILE" \
        -vf "scale=1280:-1" \
        "$OUT_DIR/frame_${INDEX}.png"

      COUNT=$((COUNT + 1))
    done

if [ -z "$(ls -A "$OUT_DIR"/frame_*.png 2>/dev/null)" ]; then
  echo "❌ No supported images found in: $SRC_DIR"
  exit 1
fi

TOTAL=$(find "$OUT_DIR" -maxdepth 1 -name 'frame_*.png' | wc -l | tr -d ' ')

echo "✅ Prepared $TOTAL images"
