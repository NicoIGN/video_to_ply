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

# 0 = keep original video resolution
IMAGE_WIDTH="${IMAGE_WIDTH:-0}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-0}"

FPS="${FPS:-}"
NUM_FRAMES="${NUM_FRAMES:-}"

BLUR_THRESHOLD="${BLUR_THRESHOLD:-120}"
DIFF_THRESHOLD="${DIFF_THRESHOLD:-5}"

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
# MODE FPS
# ======================
if [[ -n "$FPS" ]]; then
    echo "⚙️ Mode: FPS ($FPS)"

    ffmpeg -hide_banner -loglevel error -stats \
        -i "$VIDEO" \
        -vf "fps=$FPS,${SCALE_FILTER}" \
        "$IMAGE_DIR/frame_%06d.png"

# ======================
# MODE SMART
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
    # Dense extraction
    # ----------------------
    ffmpeg -hide_banner -loglevel error -stats \
        -i "$VIDEO" \
        -vf "fps=1/${INTERVAL},${SCALE_FILTER}" \
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

    if sharpness(img) < blur_threshold:
        continue

    if last is not None and diff(img, last) < diff_threshold:
        continue

    candidates.append(f)
    last = img

print(f"📊 Candidates: {len(candidates)}")

if len(candidates) > N:
    step = len(candidates) / N
    selected = [candidates[int(i * step)] for i in range(N)]
else:
    selected = candidates[:N]

for i, f in enumerate(selected):
    dst = os.path.join(out_dir, f"frame_{i:06d}.png")
    shutil.copy2(f, dst)

print(f"✅ Selected frames: {len(selected)}")
EOF
fi

rm -rf "$TMP_DIR"

# ======================
# 🔥 FATAL SIZE CHECK (COLMAP SAFE)
# ======================
echo "🔎 Checking image geometry consistency..."

python3 - <<EOF
import cv2
import glob
import sys
from collections import Counter

files = glob.glob("$IMAGE_DIR/frame_*.png")
sizes = []

for f in files:
    img = cv2.imread(f)
    if img is None:
        print("❌ corrupted image:", f)
        sys.exit(1)
    sizes.append((img.shape[1], img.shape[0]))  # (W, H)

c = Counter(sizes)

if len(c) != 1:
    print("❌ FATAL: inconsistent image sizes detected:")
    for k, v in c.items():
        print(f"  size {k}: {v} images")
    sys.exit(1)

print(f"✅ All images have consistent size: {list(c.keys())[0]}")
EOF

# ======================
# SUMMARY
# ======================
COUNT=$(find "$IMAGE_DIR" -name "frame_*.png" | wc -l | tr -d ' ')
echo "✅ Final frames: $COUNT"
