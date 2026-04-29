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
  case "$arg" in
    --skip-ns)
      SKIP_NS=true
      ;;
  esac
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

rm -f "$LOG_FILE"

echo "────────────────────────────────────"
echo "📁 INPUT            : $DATA_DIR"
echo "📁 OUTPUT           : $OUTPUT_DIR"
echo "⚙️ DEVICE           : $DEVICE"
echo "📷 CAMERA TYPE      : $CAMERA_TYPE"
echo "🔀 MATCHING METHOD  : $MATCHING_METHOD"
echo "📉 NUM DOWNSCALES   : $NUM_DOWNSCALES"
echo "────────────────────────────────────"

# ======================
# ENV
# ======================


if [[ "$DEVICE" == "cpu" ]]; then
  echo "🧠 CPU MODE"
  export LIBGL_ALWAYS_SOFTWARE=1
  export QT_QPA_PLATFORM=offscreen
  export MPLBACKEND=Agg
  export CUDA_VISIBLE_DEVICES=""
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
fi

# ======================
# RUN PIPELINE
# ======================
echo "🚀 Running ns-process-data..."

set +e

if [[ "$DEVICE" == "cpu" ]]; then
  GPU_FLAG="--no-gpu"
else
  GPU_FLAG=""
fi

ns-process-data images \
  $GPU_FLAG \
  --data "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --camera-type "$CAMERA_TYPE" \
  --matching-method "$MATCHING_METHOD" \
  --num-downscales $NUM_DOWNSCALES \
  > "$LOG_FILE" 2>&1

STATUS=$?
set -e

# ======================
# LOG CHECK
# ======================
if [ "$STATUS" -ne 0 ]; then
  echo "❌ PIPELINE FAILED"
  tail -n 80 "$LOG_FILE"
  exit 1
fi

TRANSFORMS="$OUTPUT_DIR/transforms.json"

# ======================
# VALIDATION
# ======================
if [ ! -f "$TRANSFORMS" ]; then
  echo "💀 SUCCESS BUT NO transforms.json"
  tail -n 80 "$LOG_FILE"
  exit 1
fi

echo "✅ ns-process-data SUCCESS"
echo "📦 OUTPUT READY: $OUTPUT_DIR"
