#!/bin/bash
set -e


-# ======================
-# LOAD CONFIG
-# ======================
-SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
-source "$SCRIPT_DIR/../config/config.sh"


# ======================
# INPUTS
# ======================
DATA_DIR=${1:-dataset/images}
OUTPUT_DIR=${2:-dataset/ori}
DEVICE=${DEVICE:-cpu}   # cpu | gpu

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

# ======================
# ENV SETUP
# ======================

if [[ "$DEVICE" == "gpu" ]]; then
  echo "🚀 GPU MODE"

  # laisse CUDA actif
  unset CUDA_VISIBLE_DEVICES

  # évite bugs GUI inutiles
  export QT_QPA_PLATFORM=offscreen
  export MPLBACKEND=Agg

else
  echo "🧠 CPU MODE (safe)"

  # 🔴 force CPU partout
  export CUDA_VISIBLE_DEVICES=""
  export COLMAP_USE_GPU=0
  export COLMAP_NO_GPU=1

  # 🔴 fix OpenGL (crash classique Mac / headless)
  export LIBGL_ALWAYS_SOFTWARE=1
  export QT_QPA_PLATFORM=offscreen
  export DISPLAY=

  # 🔴 stabilité
  export MPLBACKEND=Agg
  export OPENCV_LOG_LEVEL=ERROR
  export XDG_RUNTIME_DIR=/tmp/runtime-root
  export OMP_NUM_THREADS=4

fi

# ======================
# RUN
# ======================
set +e

ns-process-data images \
  --data "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --camera-type perspective \
  --matching-method sequential \
  --num-downscales 1 \
  2> "$LOG_FILE"

STATUS=$?

set -e

# ======================
# CHECK
# ======================
echo "STATUS: $STATUS"

if [ "$STATUS" -ne 0 ]; then
  echo "❌ PIPELINE FAILED"
  echo "📄 Last logs:"
  tail -n 40 "$LOG_FILE"
  exit 1
fi

echo "✅ ns-process-data SUCCESS"
