#!/bin/bash
set -e

# ======================
# LOAD CONFIG
# ======================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config/config.sh"

# ======================
# INPUTS
# ======================
DATA_DIR=${1:-dataset/images}
OUTPUT_DIR=${2:-dataset/ori}
DEVICE=${DEVICE:-cpu}   # cpu | gpu

SKIP_NS=true

# ======================
# FLAGS
# ======================
for arg in "$@"; do
  if [[ "$arg" == "--skip-ns" ]]; then
    SKIP_NS=true
  fi
done

if [ ! -d "$DATA_DIR" ]; then
  echo "❌ Data dir not found: $DATA_DIR"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

LOG_FILE="/tmp/ns_process.log"
rm -f "$LOG_FILE"

echo "📁 Input : $DATA_DIR"
echo "📁 Output: $OUTPUT_DIR"
echo "⚙️ Device: $DEVICE"
echo "🚦 Skip NS: $SKIP_NS"

# ======================
# ENV SETUP
# ======================
export CAMERA_TYPE="simple_pinhole"

if [[ "$DEVICE" == "gpu" ]]; then
  echo "🚀 GPU MODE"
  unset CUDA_VISIBLE_DEVICES
  export QT_QPA_PLATFORM=offscreen
  export MPLBACKEND=Agg
else
  echo "🧠 CPU MODE (safe)"
fi

# ======================
# FALLBACK SCRIPT
# ======================
FALLBACK_SCRIPT="$(dirname "$0")/colmap_to_transforms.py"

# ======================
# SKIP MODE (DIRECT COLMAP → TRANSFORMS)
# ======================
if [[ "$SKIP_NS" == "true" ]]; then
  echo "⚡ SKIP MODE ENABLED"
  echo "🚀 Running COLMAP → transforms directly"

  python3 "$FALLBACK_SCRIPT" "$OUTPUT_DIR"

  STATUS=$?

  if [ "$STATUS" -eq 0 ]; then
    echo "✅ transforms.json generated (skip mode)"
    exit 0
  else
    echo "💀 Direct COLMAP conversion failed"
    exit 1
  fi
fi

# ======================
# RUN NS PIPELINE
# ======================
set +e

ns-process-data images \
  --data "$DATA_DIR" \
  --sfm_tool colmap \
  --output-dir "$OUTPUT_DIR" \
  --camera-type $CAMERA_TYPE \
  --matching-method sequential \
  --num-downscales 1 \
  2> "$LOG_FILE"

STATUS=$?

set -e

# ======================
# CHECK
# ======================
echo "STATUS: $STATUS"

COLMAP_DIR="$OUTPUT_DIR/colmap/sparse/0"

if [ "$STATUS" -ne 0 ]; then
  echo "❌ PIPELINE FAILED"
  echo "📄 Last logs:"
  tail -n 40 "$LOG_FILE"

  # fallback automatique
  if [ -f "$COLMAP_DIR/cameras.bin" ] || [ -f "$COLMAP_DIR/images.bin" ]; then
    echo "⚠️ COLMAP output detected → fallback"

    python3 "$FALLBACK_SCRIPT" "$OUTPUT_DIR"

    FALLBACK_STATUS=$?

    if [ "$FALLBACK_STATUS" -eq 0 ]; then
      echo "✅ Fallback transforms.json generated successfully"
      exit 0
    else
      echo "💀 Fallback also failed"
      exit 1
    fi
  else
    echo "💀 No COLMAP output found → cannot recover"
    exit 1
  fi
fi

echo "✅ ns-process-data SUCCESS"
