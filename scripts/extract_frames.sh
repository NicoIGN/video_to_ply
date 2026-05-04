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
DIFF_THRESHOLD="${DIFF_THRESHOLD:-5}"

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
# MODE FPS
# ======================
if [[ -n "$FPS" ]]; then
    echo "⚙️ Mode: FPS ($FPS)"

    ffmpeg -hide_banner -loglevel error -stats \
        -i "$VIDEO" \
        -vf "fps=$FPS,scale=${IMAGE_WIDTH}:-1" \
        "$IMAGE_DIR/frame_%05d.png"

# ======================
# MODE SMART (COLMAP SAFE)
# ======================
else
    echo "⚙️ Mode: SMART selection ($NUM_FRAMES)"

    DURATION=$(ffprobe -v error \
        -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 \
        "$VIDEO")

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

    # ----------------------
    # 1. DENSE EXTRACTION
    # ----------------------
    ffmpeg -hide_banner -loglevel error -stats \
        -i "$VIDEO" \
        -vf "fps=1/${INTERVAL},scale=${IMAGE_WIDTH}:-1" \
        "$TMP_DIR/frame_%06d.png"

    echo "🔎 Filtering (blur + redundancy)..."

    python3 - <<EOF
import os
import cv2
import shutil
from glob import glob

tmp_dir = "$TMP_DIR"
out_dir = "$IMAGE_DIR"

N = int("$NUM_FRAMES")
blur_threshold = float("$BLUR_THRESHOLD")
diff_threshold = float("$DIFF_THRESHOLD")

files = sorted(glob(os.path.join(tmp_dir, "*.png")))

def sharpness(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

def diff(a, b):
    return cv2.absdiff(a, b).mean()

candidates = []
last = None

for f in files:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    s = sharpness(img)
    if s < blur_threshold:
        continue

    if last is not None:
        if diff(img, last) < diff_threshold:
            continue

    candidates.append(f)
    last = img

print(f"📊 Candidates after filtering: {len(candidates)}")

# ----------------------
# IMPORTANT FIX:
# uniform temporal sampling (preserves geometry)
# ----------------------
if len(candidates) > N:
    step = len(candidates) / N
    selected = [candidates[int(i * step)] for i in range(N)]
else:
    selected = candidates[:N]

# ----------------------
# COPY (NOT RENAME)
# keeps stable indexing for HLOC / COLMAP
# ----------------------
for i, f in enumerate(selected):
    dst = os.path.join(out_dir, f"frame_{i:06d}.png")
    shutil.copy2(f, dst)

print(f"✅ Final selected frames: {len(selected)} (geometry preserved)")
EOF
fi

rm -rf "$TMP_DIR"

# ======================
# SUMMARY
# ======================
COUNT=$(find "$IMAGE_DIR" -name "frame_*.png" | wc -l | tr -d ' ')
echo "✅ Final frames: $COUNT"
