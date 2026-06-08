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

# Optional video crop (seconds)
VIDEO_START="${VIDEO_START:-}"
VIDEO_END="${VIDEO_END:-}"

START_INDEX="${START_INDEX:-0}"
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

if [[ -n "$FPS" && ! "$FPS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "❌ FPS must be a positive number"
    exit 1
fi

if [[ -n "$NUM_FRAMES" && ! "$NUM_FRAMES" =~ ^[0-9]+$ ]]; then
    echo "❌ NUM_FRAMES must be a positive integer"
    exit 1
fi

if [[ -n "$VIDEO_START" && ! "$VIDEO_START" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "❌ VIDEO_START must be a positive number"
    exit 1
fi

if [[ -n "$VIDEO_END" && ! "$VIDEO_END" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "❌ VIDEO_END must be a positive number"
    exit 1
fi

if [[ -n "$VIDEO_START" && -n "$VIDEO_END" ]]; then
    python3 - <<EOF
start = float("$VIDEO_START")
end = float("$VIDEO_END")

if end <= start:
    raise SystemExit("❌ VIDEO_END must be greater than VIDEO_START")
EOF
fi

if ! [[ "$START_INDEX" =~ ^[0-9]+$ ]]; then
    echo "❌ START_INDEX must be a positive integer"
    exit 1
fi

mkdir -p "$IMAGE_DIR"
mkdir -p "$TMP_DIR"

if [[ "${START_INDEX:-0}" -eq 0 ]]; then
    echo "🧹 Cleaning existing frames"
    rm -f "$IMAGE_DIR"/frame_*.png
else
    echo "⏩ START_INDEX=$START_INDEX, keeping existing frames"
fi

rm -f "$TMP_DIR"/frame_*.png

echo "🎬 Video: $VIDEO"
echo "📁 Output (IMAGE_DIR): $IMAGE_DIR"

if [[ -n "$VIDEO_START" ]]; then
    echo "⏩ VIDEO_START: ${VIDEO_START}s"
fi

if [[ -n "$VIDEO_END" ]]; then
    echo "⏹️ VIDEO_END: ${VIDEO_END}s"
fi

echo "🔢 START_INDEX: $START_INDEX"

# ======================
# COMPUTE EFFECTIVE DURATION
# ======================
FULL_DURATION=$(ffprobe -v error \
    -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 \
    "$VIDEO")

EFFECTIVE_DURATION=$(python3 - <<EOF
full = float("$FULL_DURATION")

start = float("${VIDEO_START:-0}")

if "${VIDEO_END:-}" != "":
    end = float("$VIDEO_END")
else:
    end = full

duration = end - start

if duration <= 0:
    raise SystemExit("❌ Invalid effective duration")

print(duration)
EOF
)

# ======================
# BUILD FFMPEG TRIM ARGS
# ======================
FFMPEG_INPUT_ARGS=()

if [[ -n "$VIDEO_START" ]]; then
    FFMPEG_INPUT_ARGS+=(-ss "$VIDEO_START")
fi

FFMPEG_INPUT_ARGS+=(-i "$VIDEO")

if [[ -n "$VIDEO_END" ]]; then

    if [[ -n "$VIDEO_START" ]]; then
        TRIM_DURATION=$(python3 - <<EOF
start = float("$VIDEO_START")
end = float("$VIDEO_END")
print(end - start)
EOF
)
    else
        TRIM_DURATION="$VIDEO_END"
    fi

    FFMPEG_INPUT_ARGS+=(-t "$TRIM_DURATION")
fi

# ======================
# MODE FPS
# ======================
if [[ -n "$FPS" ]]; then
    echo "⚙️ Mode: FPS ($FPS)"
    echo "⏱️ Effective duration: $EFFECTIVE_DURATION s"

    ffmpeg -hide_banner -loglevel error -stats \
        "${FFMPEG_INPUT_ARGS[@]}" \
        -vf "fps=$FPS,${SCALE_FILTER}" \
        "$IMAGE_DIR/frame_%05d.png"

# ======================
# MODE SMART
# ======================
else
    echo "⚙️ Mode: SMART selection ($NUM_FRAMES)"

    TARGET_TMP=$((NUM_FRAMES * 3))

    INTERVAL=$(python3 - <<EOF
duration = float("$EFFECTIVE_DURATION")
count = int("$TARGET_TMP")
print(duration / count)
EOF
)

    echo "📐 Oversampling: $TARGET_TMP frames"
    echo "⏱️ Effective duration: $EFFECTIVE_DURATION s"
    echo "⏱️ Interval: $INTERVAL s"

    # extract dense frames
    ffmpeg -hide_banner -loglevel error -stats \
        "${FFMPEG_INPUT_ARGS[@]}" \
        -vf "fps=1/${INTERVAL},${SCALE_FILTER}" \
        "$TMP_DIR/frame_%05d.png"

    echo "🔎 Adaptive filtering..."

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    python3 "$SCRIPT_DIR/filter_frames.py" \
        --tmp_dir "$TMP_DIR" \
        --out_dir "$IMAGE_DIR" \
        --start_index "$START_INDEX" \
        --num_frames "$NUM_FRAMES"

fi


rm -rf "$TMP_DIR"

# ======================
# COLMAP SAFETY CHECK
# ======================
echo "🔎 Checking image geometry consistency..."

python3 - <<EOF
import cv2
import glob
import sys
from collections import Counter

files = glob.glob("$IMAGE_DIR/frame_*.png")

if not files:
    print("❌ No frames extracted")
    sys.exit(1)

sizes = []

for f in files:
    img = cv2.imread(f)

    if img is None:
        print("❌ Corrupted image:", f)
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
