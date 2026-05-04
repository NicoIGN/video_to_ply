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
TMP_DIR="${TMP_DIR:-dataset/tmp_frames}"
IMAGE_WIDTH="${IMAGE_WIDTH:-1280}"

FPS="${FPS:-}"
NUM_FRAMES="${NUM_FRAMES:-}"

BLUR_THRESHOLD="${BLUR_THRESHOLD:-120}"
DIFF_THRESHOLD="${DIFF_THRESHOLD:-5}" # % difference (ImageMagick)

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
# MODE FPS (simple)
# ======================
if [[ -n "$FPS" ]]; then
    echo "⚙️ Mode: FPS ($FPS)"

    ffmpeg -hide_banner -loglevel error -stats \
        -i "$VIDEO" \
        -vf "fps=$FPS,scale=${IMAGE_WIDTH}:-1" \
        "$IMAGE_DIR/frame_%05d.png"

# ======================
# MODE SMART NUM_FRAMES
# ======================
else
    echo "⚙️ Mode: SMART selection ($NUM_FRAMES)"

    # --- duration
    DURATION=$(ffprobe -v error \
        -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 \
        "$VIDEO")

    # --- oversample x2
    TARGET_TMP=$(python3 - <<EOF
print(int($NUM_FRAMES * 3))
EOF
)

    INTERVAL=$(python3 - <<EOF
duration = float("$DURATION")
count = int("$TARGET_TMP")
print(duration / count)
EOF
)

    echo "📐 Oversampling: $TARGET_TMP frames"
    echo "⏱️ Interval: $INTERVAL s"

    # --- extraction dense
    ffmpeg -hide_banner -loglevel error -stats \
        -i "$VIDEO" \
        -vf "fps=1/${INTERVAL},scale=${IMAGE_WIDTH}:-1" \
        "$TMP_DIR/frame_%05d.png"

    echo "🔎 Filtering..."

    python3 - <<EOF
import cv2
import os
from glob import glob

tmp_dir = "$TMP_DIR"
out_dir = "$IMAGE_DIR"

blur_threshold = float("$BLUR_THRESHOLD")

files = sorted(glob(os.path.join(tmp_dir, "*.png")))

selected = []
last_img = None

def sharpness(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

def diff(img1, img2):
    return cv2.absdiff(img1, img2).mean()

for f in files:
    img = cv2.imread(f)
    if img is None:
        continue

    s = sharpness(img)
    if s < blur_threshold:
        continue

    if last_img is not None:
        d = diff(img, last_img)
        if d < 2.0:
            continue

    selected.append((f, s))
    last_img = img

# sort by sharpness
selected.sort(key=lambda x: -x[1])

# keep best N
N = int("$NUM_FRAMES")
selected = selected[:N]

# restore chronological order
selected = sorted(selected, key=lambda x: x[0])

for i, (f, _) in enumerate(selected):
    out = os.path.join(out_dir, f"frame_{i:05d}.png")
    os.rename(f, out)

print(f"✅ Selected {len(selected)} frames")
EOF
fi

rm -rf $TMP_DIR

# ======================
# SUMMARY
# ======================
COUNT=$(find "$IMAGE_DIR" -name "frame_*.png" | wc -l | tr -d ' ')
echo "✅ Final frames: $COUNT"
