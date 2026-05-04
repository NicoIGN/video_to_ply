#!/bin/bash
set -euo pipefail

# ======================
# REQUIRED VARIABLES
# ======================
if [[ -z "${VIDEO:-}" ]]; then
    echo "❌ VIDEO variable is not set"
    exit 1
fi

# ======================
# CONFIGURATION
# ======================
IMAGE_DIR="${IMAGE_DIR:-dataset/images}"
TMP_DIR="${TMP_DIR:-$(dirname "$IMAGE_DIR")/tmp_frames}"

IMAGE_WIDTH="${IMAGE_WIDTH:-0}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-0}"

FPS="${FPS:-}"
NUM_FRAMES="${NUM_FRAMES:-}"

# ======================
# SCALE LOGIC
# ======================
if [[ "$IMAGE_WIDTH" -ne 0 && "$IMAGE_HEIGHT" -ne 0 ]]; then
    SCALE_FILTER="scale=${IMAGE_WIDTH}:${IMAGE_HEIGHT},setsar=1,format=rgb24"
else
    SCALE_FILTER="setsar=1,format=rgb24"
fi

# ======================
# VALIDATION
# ======================
if [[ ! -f "$VIDEO" ]]; then
    echo "❌ Video not found: $VIDEO"
    exit 1
fi

if [[ -n "$FPS" && -n "$NUM_FRAMES" ]]; then
    echo "❌ Define either FPS or NUM_FRAMES, not both"
    exit 1
fi

if [[ -z "$FPS" && -z "$NUM_FRAMES" ]]; then
    echo "❌ Define FPS or NUM_FRAMES"
    exit 1
fi

mkdir -p "$IMAGE_DIR"
mkdir -p "$TMP_DIR"
rm -f "$IMAGE_DIR"/frame_*.png
rm -f "$TMP_DIR"/frame_*.png

echo "🎬 Video: $VIDEO"
echo "📁 Output: $IMAGE_DIR"

# ======================
# MODE FPS (UNCHANGED)
# ======================
if [[ -n "$FPS" ]]; then
    echo "⚙️ Mode: FPS ($FPS)"

    ffmpeg -hide_banner -loglevel error -stats \
        -i "$VIDEO" \
        -vf "fps=$FPS,${SCALE_FILTER}" \
        "$IMAGE_DIR/frame_%05d.png"

# ======================
# MODE SMART
# ======================
else
    echo "⚙️ Mode: SMART selection ($NUM_FRAMES)"

    DURATION=$(ffprobe -v error \
        -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 \
        "$VIDEO")

    TARGET_TMP=$((NUM_FRAMES * 3))

    INTERVAL=$(python3 - <<EOF
duration = float("$DURATION")
count = int("$TARGET_TMP")
print(duration / count)
EOF
)

    echo "📐 Oversampling: $TARGET_TMP frames"
    echo "⏱️ Interval: $INTERVAL s"

    # extract dense frames
    ffmpeg -hide_banner -loglevel error -stats \
        -i "$VIDEO" \
        -vf "fps=1/${INTERVAL},${SCALE_FILTER}" \
        "$TMP_DIR/frame_%05d.png"

    echo "🔎 Adaptive filtering..."


  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    python3 $SCRIPT_DIR/filter_frames.py \
        --tmp_dir "$TMP_DIR" \
        --out_dir "$IMAGE_DIR" \
        --num_frames "$NUM_FRAMES"

fi

rm -rf "$TMP_DIR"

# ======================
# COLMAP SAFETY CHECK
# ======================
echo "🔎 Checking image geometry consistency..."

python3 - <<EOF
import cv2, glob, sys
from collections import Counter

files = glob.glob("$IMAGE_DIR/frame_*.png")
sizes = []

for f in files:
    img = cv2.imread(f)
    if img is None:
        print("❌ corrupted image:", f)
        sys.exit(1)
    sizes.append((img.shape[1], img.shape[0]))

c = Counter(sizes)

if len(c) != 1:
    print("❌ FATAL: inconsistent image sizes:")
    for k, v in c.items():
        print(k, v)
    sys.exit(1)

print(f"✅ All images have consistent size: {list(c.keys())[0]}")
EOF

COUNT=$(find "$IMAGE_DIR" -name "frame_*.png" | wc -l | tr -d ' ')
echo "✅ Final frames: $COUNT"
