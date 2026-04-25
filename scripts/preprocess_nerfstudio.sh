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
PYCOLMAP_MODE=true   # 🔥 NEW: force safe backend

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
CAMERA_TYPE="perspective"
VERBOSE=""

rm -f "$LOG_FILE"

echo "────────────────────────────────────"
echo "📁 INPUT   : $DATA_DIR"
echo "📁 OUTPUT  : $OUTPUT_DIR"
echo "⚙️ DEVICE  : $DEVICE"
echo "🧠 PYCOLMAP: $PYCOLMAP_MODE"
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
  export CUDA_VISIBLE_DEVICES=""
  export LIBGL_ALWAYS_SOFTWARE=1
  export QT_QPA_PLATFORM=offscreen
  export MPLBACKEND=Agg
  export COLMAP_SIFT_NO_GPU=1
  export NS_DISABLE_GPU=1
fi

# ======================
# 🔥 FORCE PYCOLMAP MODE (IMPORTANT FIX)
# ======================
if [[ "$PYCOLMAP_MODE" == "true" ]]; then
  echo "🚀 Using PYCOLMAP backend (NO COLMAP CLI)"

  export NERFSTUDIO_SFM_BACKEND=pycolmap
  export NS_USE_PYCOLMAP=1
  echo colmap: `which which colmap`
fi

# ======================
# PATHS
# ======================
COLMAP_DIR="$OUTPUT_DIR/colmap/sparse/0"
FALLBACK_SCRIPT="$(dirname "$0")/colmap_to_transforms.py"
TRANSFORMS="$OUTPUT_DIR/transforms.json"

# ======================
# RUN PIPELINE
# ======================
echo "🚀 Running ns-process-data..."

set +e

ns-process-data images \
  --data "$DATA_DIR" \
  --sfm_tool colmap \
  --output-dir "$OUTPUT_DIR" \
  --camera-type $CAMERA_TYPE \
  --matching-method sequential \
  --num-downscales 1 \
  > "$LOG_FILE" 2>&1

STATUS=$?
set -e

# ======================
# STRICT VALIDATION
# ======================
if [ "$STATUS" -ne 0 ]; then
  echo "❌ PIPELINE FAILED"
  tail -n 50 "$LOG_FILE"
  exit 1
fi

# 🔥 CRITICAL CHECK
if [ ! -f "$TRANSFORMS" ]; then
  echo "💀 SUCCESS BUT NO transforms.json"
  tail -n 50 "$LOG_FILE"
  exit 1
fi

echo "✅ ns-process-data SUCCESS (PYCOLMAP MODE)"
