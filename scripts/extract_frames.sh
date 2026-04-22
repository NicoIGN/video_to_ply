#!/bin/bash
set -e

# ======================
# LOAD CONFIG (CRITICAL)
# ======================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config/config.sh"

VIDEO=$1
FPS=${2:-10}
OUT_DIR=${3:-dataset/images}

# ======================
# CHECKS
# ======================
if [ -z "$VIDEO" ]; then
  echo "❌ Usage: $0 video.mp4 [fps] [output_dir]"
  exit 1
fi

if [ ! -f "$VIDEO" ]; then
  echo "❌ Video not found: $VIDEO"
  exit 1
fi

# ======================
# PREPARE OUTPUT
# ======================
mkdir -p "$OUT_DIR"

echo "🎬 Extracting frames from: $VIDEO"
echo "⚙️ FPS: $FPS"
echo "📁 Output: $OUT_DIR"

# ======================
# FFMEG PIPELINE
# ======================
ffmpeg -hide_banner -loglevel error -stats \
  -i "$VIDEO" \
  -vf "fps=$FPS,scale=1280:-1" \
  "$OUT_DIR/frame_%04d.png"

echo "✅ Frames extracted"
