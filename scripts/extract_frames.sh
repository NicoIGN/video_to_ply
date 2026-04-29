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
IMAGE_WIDTH="${IMAGE_WIDTH:-1280}"

# ======================
# VALIDATION
# ======================
if [[ ! -f "$VIDEO" ]]; then
    echo "❌ Video not found: $VIDEO"
    exit 1
fi

echo "FPS1: $FPS"
echo "NUM_FRAMES1: $NUM_FRAMES"
FPS="${FPS:-}"
NUM_FRAMES="${NUM_FRAMES:-}"

echo "FPS2: $FPS"
echo "NUM_FRAMES2: $NUM_FRAMES"

if [[ -n "$FPS" && -n "$NUM_FRAMES" ]]; then
    echo "❌ Please define either FPS ($FPS) or NUM_FRAMES ($NUM_FRAMES), but not both"
    exit 1
fi

if [[ -z "$FPS" && -z "$NUM_FRAMES" ]]; then
    echo "❌ Please define either FPS or NUM_FRAMES"
    exit 1
fi

if [[ -n "$FPS" ]]; then
    if ! [[ "$FPS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "❌ FPS must be a positive number"
        exit 1
    fi
fi

if [[ -n "$NUM_FRAMES" ]]; then
    if ! [[ "$NUM_FRAMES" =~ ^[0-9]+$ ]] || [[ "$NUM_FRAMES" -le 0 ]]; then
        echo "❌ NUM_FRAMES must be a positive integer"
        exit 1
    fi
fi

# ======================
# PREPARE OUTPUT
# ======================
mkdir -p "$IMAGE_DIR"
rm -f "$IMAGE_DIR"/frame_*.png

echo "🎬 Video:   $VIDEO"
echo "📏 Width:   ${IMAGE_WIDTH}px"
echo "📁 Output:  $IMAGE_DIR"

# ======================
# EXTRACTION MODE: FPS
# ======================
if [[ -n "$FPS" ]]; then
    echo "⚙️ Mode:    Fixed FPS"
    echo "🎞️ FPS:     $FPS"

    ffmpeg -hide_banner -loglevel error -stats \
        -i "$VIDEO" \
        -vf "fps=$FPS,scale=${IMAGE_WIDTH}:-1" \
        "$IMAGE_DIR/frame_%05d.png"

# ======================
# EXTRACTION MODE: SHARP FRAMES
# ======================
else
    echo "⚙️ Mode:    Sharp frame selection"
    echo "🖼️ Frames:  $NUM_FRAMES"

    DURATION=$(ffprobe -v error \
        -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 \
        "$VIDEO")

    if [[ -z "$DURATION" ]]; then
        echo "❌ Failed to read video duration"
        exit 1
    fi

    INTERVAL=$(python3 - <<EOF
duration = float("$DURATION")
count = int("$NUM_FRAMES")
print(duration / count)
EOF
)

    echo "⏱️ Duration: ${DURATION}s"
    echo "📐 Interval: ${INTERVAL}s"

    ffmpeg -hide_banner -loglevel error -stats \
        -i "$VIDEO" \
        -vf "fps=1/${INTERVAL},thumbnail=4,scale=${IMAGE_WIDTH}:-1" \
        -frames:v "$NUM_FRAMES" \
        "$IMAGE_DIR/frame_%05d.png"
fi

# ======================
# SUMMARY
# ======================
EXTRACTED=$(find "$IMAGE_DIR" -maxdepth 1 -name 'frame_*.png' | wc -l | tr -d ' ')

echo "✅ Extracted $EXTRACTED frames"

if [[ -n "$NUM_FRAMES" && "$EXTRACTED" -ne "$NUM_FRAMES" ]]; then
    echo "⚠️ Requested $NUM_FRAMES frames, got $EXTRACTED"
fi
