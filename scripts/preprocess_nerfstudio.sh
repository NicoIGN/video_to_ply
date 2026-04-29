#!/bin/bash
set -e

# ======================
# LOAD CONFIG
# ======================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#source "$SCRIPT_DIR/../config/config.sh"

# ======================
# INPUTS
# ======================
: "${DATA_DIR:?❌ DATA_DIR env var is required}"
: "${OUTPUT_DIR:?❌ OUTPUT_DIR env var is required}"
: "${DEVICE:?❌ DEVICE env var is required (cpu|gpu)}"

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

rm -f "$LOG_FILE"

echo "────────────────────────────────────"
echo "📁 INPUT                 : $DATA_DIR"
echo "📁 OUTPUT                : $OUTPUT_DIR"
echo "⚙️ DEVICE                : $DEVICE"
echo "📷 CAMERA TYPE           : $CAMERA_TYPE"
echo "🔀 MATCHING METHOD       : $MATCHING_METHOD"
echo "🧠 SFMT TOOL             : $SFMT_TOOL"
echo "🧬 FEATURE TYPE          : $FEATURE_TYPE"
echo "🔗 MATCHER TYPE          : $MATCHER_TYPE"
echo "📉 NUM DOWNSCALES        : $NUM_DOWNSCALES"
echo "✂️ CROP FACTOR           : ${CROP_FACTOR:-none}"
echo "🎯 RADIUS CROP           : $PERCENT_RADIUS_CROP"
echo "📐 SCALE FACTOR          : $CAMERA_RES_SCALE_FACTOR"
echo "📷 SINGLE CAMERA MODE    : $USE_SINGLE_CAMERA_MODE"
echo "🔧 REFINE INTRINSICS     : $REFINE_INTRINSICS"
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

ARGS=()

ARGS+=($GPU_FLAG)
ARGS+=(--data "$DATA_DIR")
ARGS+=(--output-dir "$OUTPUT_DIR")
ARGS+=(--camera-type "$CAMERA_TYPE")
ARGS+=(--matching-method "$MATCHING_METHOD")
ARGS+=(--feature-type "$FEATURE_TYPE")
ARGS+=(--matcher-type "$MATCHER_TYPE")
ARGS+=(--num-downscales "$NUM_DOWNSCALES")
ARGS+=(--percent-radius-crop "$PERCENT_RADIUS_CROP")
ARGS+=(--refine-intrinsics)
ARGS+=(--use-single-camera-mode)
ARGS+=(--sfm-tool "$SFMT_TOOL")

# SAFE crop-factor handling
if [[ -n "$CROP_FACTOR" ]]; then
  ARGS+=(--crop-factor $CROP_FACTOR)
fi

ns-process-data images "${ARGS[@]}" > "$LOG_FILE" 2>&1

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
