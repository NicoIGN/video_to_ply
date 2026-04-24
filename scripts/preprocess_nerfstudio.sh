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
DEVICE=${DEVICE:-cpu}

SKIP_NS=false

# ======================
# FLAGS
# ======================
for arg in "$@"; do
  if [[ "$arg" == "--skip-ns" ]]; then
    SKIP_NS=true
  fi
done

# ======================
# CHECK INPUT
# ======================
if [ ! -d "$DATA_DIR" ]; then
  echo "❌ Data dir not found: $DATA_DIR"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

LOG_FILE="/tmp/ns_process.log"
#CAMERA_TYPE="simple_pinhole"
CAMERA_TYPE="perspective"
VERBOSE="" #--verbose

rm -f "$LOG_FILE"

echo "────────────────────────────────────"
echo "📁 INPUT        : $DATA_DIR"
echo "📁 OUTPUT       : $OUTPUT_DIR"
echo "⚙️ DEVICE       : $DEVICE"
echo "🚦 SKIP_NS      : $SKIP_NS"
echo "📄 LOG FILE        : $LOG_FILE"
echo "📷 CAMERA TYPE     : $CAMERA_TYPE"
echo "────────────────────────────────────"

# ======================
# ENV
# ======================

if [[ "$DEVICE" == "gpu" ]]; then
  echo "🚀 GPU MODE"
  unset CUDA_VISIBLE_DEVICES
  export QT_QPA_PLATFORM=offscreen
  export MPLBACKEND=Agg
else
  echo "🧠 CPU MODE"
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export COLMAP_USE_GPU=0
  export QT_QPA_PLATFORM=offscreen
  export MPLBACKEND=Agg
  export OPENCV_LOG_LEVEL=ERROR
  export XDG_RUNTIME_DIR=/tmp/runtime-root
  export CUDA_VISIBLE_DEVICES=""
  export LIBGL_ALWAYS_SOFTWARE=1

  export NS_DEBUG=1
  export NS_LOG_LEVEL=debug
fi

# ======================
# PATHS
# ======================
COLMAP_DIR="$OUTPUT_DIR/colmap/sparse/0"
FALLBACK_SCRIPT="$(dirname "$0")/colmap_to_transforms.py"
TRANSFORMS="$OUTPUT_DIR/transforms.json"

# ======================
# SKIP MODE
# ======================
if [[ "$SKIP_NS" == "true" ]]; then
  echo "⚡ SKIP MODE ENABLED"
  echo "🚀 Running COLMAP → transforms directly"

  # ======================
  # CHECK COLMAP FIRST
  # ======================
  echo "🔍 Checking COLMAP outputs..."

  if [ ! -d "$COLMAP_DIR" ]; then
    echo "❌ COLMAP folder missing:"
    echo "   $COLMAP_DIR"
    echo ""
    echo "💡 You must run COLMAP or ns-process-data first"
    exit 1
  fi

  if [ ! -f "$COLMAP_DIR/cameras.bin" ] || [ ! -f "$COLMAP_DIR/images.bin" ]; then
    echo "❌ COLMAP incomplete data"
    echo "   Missing files in: $COLMAP_DIR"
    echo ""
    echo "📦 Expected:"
    echo "   - cameras.bin"
    echo "   - images.bin"
    echo ""
    echo "💡 Fix: run COLMAP preprocessing first"
    exit 1
  fi

  echo "✅ COLMAP data found"
  echo "📦 Running transforms export..."

  python3 "$FALLBACK_SCRIPT" "$OUTPUT_DIR"
  STATUS=$?

  if [ "$STATUS" -eq 0 ] && [ -f "$TRANSFORMS" ]; then
    echo "✅ transforms.json generated successfully"
    exit 0
  else
    echo "💀 Transforms generation failed"
    exit 1
  fi
fi

# ======================
# NS PIPELINE
# ======================
echo "🚀 Running ns-process-data..."

set +e

COLMAP_BIN=$(which colmap || true)

if [ -z "$COLMAP_BIN" ]; then
  echo "❌ COLMAP not found in PATH"
  exit 1
else
  echo "COLMAP found in $COLMAP_BIN"
fi

ns-process-data images \
  --data "$DATA_DIR" \
  --sfm_tool colmap \
  --output-dir "$OUTPUT_DIR" \
  --camera-type $CAMERA_TYPE \
  --matching-method sequential \
  --num-downscales 1 $VERBOSE
  
  2> "$LOG_FILE"

STATUS=$?
set -e

# ======================
# FALLBACK IF FAIL
# ======================
if [ "$STATUS" -ne 0 ]; then
  echo "❌ PIPELINE FAILED"
  echo "📄 Last logs:"
  tail -n 40 "$LOG_FILE"

  if [ -f "$COLMAP_DIR/cameras.bin" ] || [ -f "$COLMAP_DIR/images.bin" ]; then
    echo "⚠️ COLMAP detected → fallback transforms"

    python3 "$FALLBACK_SCRIPT" "$OUTPUT_DIR"

    FALLBACK_STATUS=$?

    if [ "$FALLBACK_STATUS" -eq 0 ]; then
      echo "✅ fallback OK"
      exit 0
    else
      echo "💀 fallback failed"
      exit 1
    fi
  else
    echo "💀 No COLMAP output → cannot recover"
    exit 1
  fi
fi

echo "✅ ns-process-data SUCCESS"
