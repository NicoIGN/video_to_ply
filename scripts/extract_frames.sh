#!/bin/bash
set -euo pipefail

# ======================
# LOAD CONFIG (CRITICAL)
# ======================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config/config.sh"

# ======================
# REQUIRED VARIABLES
# ======================
if [[ -z "${VIDEO:-}" ]]; then
    echo "❌ VIDEO variable is not set"
    exit 1
fi

# ======================
# DEFAULTS
# ======================
NUM_FRAMES="${NUM_FRAMES:-150}"
IMAGE_DIR="${IMAGE_DIR:-dataset/images}"
IMAGE_WIDTH="${IMAGE_WIDTH:-1280}"

# ======================
# CHECKS
# ======================
if [[ ! -f "$VIDEO" ]]; then
    echo "❌ Video not found: $VIDEO"
    exit 1
fi

if ! [[ "$NUM_FRAMES" =~ ^[0-9]+$ ]] || [[ "$NUM_FRAMES" -le 0 ]]; then
    echo "❌ NUM_FRAMES must be a positive integer"
    exit 1
fi

# ======================
# PREPARE OUTPUT
# ======================
mkdir -p "$IMAGE_DIR"
rm -f "$IMAGE_DIR"/frame_*.png

echo "🎬 Video:        $VIDEO"
echo "🖼️  Frames:       $NUM_FRAMES"
echo "📏 Width:         $IMAGE_WIDTH px"
echo "📁 Output:        $IMAGE_DIR"

# ======================
# GET VIDEO DURATION
# ======================
DURATION=$(ffprobe -v error \
    -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 \
    "$VIDEO")

if [[ -z "$DURATION" ]]; then
    echo "❌ Failed to read video duration"
    exit 1
fi

echo "⏱️  Duration:      ${DURATION}s"

# ======================
# COMPUTE INTERVAL
# ======================
INTERVAL=$(python3 - <<EOF
duration = float("$DURATION")
count = int("$NUM_FRAMES")
print(duration / count)
EOF
)

echo "📐 Interval:      ${INTERVAL}s"

# ======================
# EXTRACT SHARP FRAMES
# ======================
ffmpeg -hide_banner -loglevel error -stats \
    -i "$VIDEO" \
    -vf "fps=1/${INTERVAL},thumbnail=15,scale=${IMAGE_WIDTH}:-1" \
    -frames:v "$NUM_FRAMES" \
    "$IMAGE_DIR/frame_%05d.png"

EXTRACTED=$(find "$IMAGE_DIR" -maxdepth 1 -name 'frame_*.png' | wc -l | tr -d ' ')

echo "✅ Extracted $EXTRACTED frames"

if [[ "$EXTRACTED" -ne "$NUM_FRAMES" ]]; then
    echo "⚠️ Requested $NUM_FRAMES frames, got $EXTRACTED"
fi
