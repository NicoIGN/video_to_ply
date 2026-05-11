#!/bin/bash
set -e

# ======================
# INPUTS
# ======================
: "${DATA_DIR:?❌ DATA_DIR env var is required}"
: "${OUTPUT_DIR:?❌ OUTPUT_DIR env var is required}"
: "${DEVICE:?❌ DEVICE env var is required (cpu|gpu)}"

SKIP_NS=false

for arg in "$@"; do
  case "$arg" in
    --skip-ns)
      SKIP_NS=true
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


# ======================
# CHECK INPUT
# ======================
if [ ! -d "$DATA_DIR" ]; then
  echo "❌ Data dir not found: $DATA_DIR"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ======================
# LOG SETUP (NEW)
# ======================
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

PROCESS_LOG="$LOG_DIR/ns_process.log"
HEARTBEAT_LOG="$LOG_DIR/ns_process_heartbeat.log"

rm -f "$PROCESS_LOG" "$HEARTBEAT_LOG"

echo "📝 Process log: $PROCESS_LOG"
echo "💓 Heartbeat log: $HEARTBEAT_LOG"

# heartbeat (détection freeze)
(
  while true; do
    sleep 60
    echo "$(date '+%F %T') ns-process-data still running" >> "$HEARTBEAT_LOG"
  done
) &
HEARTBEAT_PID=$!

# ======================
# SUMMARY
# ======================

echo "────────────────────────────────────"

echo "📁 INPUT                 : $DATA_DIR"
echo "📁 OUTPUT                : $OUTPUT_DIR"

echo "⚙️ DEVICE                : $DEVICE"
echo "🔇 VERBOSE               : $VERBOSE"

echo "📷 CAMERA TYPE           : $CAMERA_TYPE"
echo "🧩 SAME DIMENSIONS       : $SAME_DIMENSIONS"

echo "🔀 MATCHING METHOD       : $MATCHING_METHOD"
echo "🧠 SFM TOOL              : $SFMT_TOOL"

echo "🧬 FEATURE TYPE          : $FEATURE_TYPE"
echo "🔗 MATCHER TYPE          : $MATCHER_TYPE"

echo "📉 NUM DOWNSCALES        : $NUM_DOWNSCALES"

echo "✂️ CROP FACTOR           : ${CROP_FACTOR:-none}"
echo "🎯 RADIUS CROP           : ${PERCENT_RADIUS_CROP:-1.0}"

echo "📷 SINGLE CAMERA MODE    : $USE_SINGLE_CAMERA_MODE"

echo "🔧 REFINE INTRINSICS     : $REFINE_INTRINSICS"
echo "🔬 REFINE PIXSFM         : $REFINE_PIXSFM"

echo "📦 USE SFM DEPTH         : $USE_SFM_DEPTH"
echo "🐞 DEPTH DEBUG           : $INCLUDE_DEPTH_DEBUG"

echo "⏭️ SKIP COLMAP           : $SKIP_COLMAP"
echo "⚡ SKIP IMAGE PROCESSING : $SKIP_IMAGE_PROCESSING"

echo "🗂️ COLMAP MODEL PATH     : ${COLMAP_MODEL_PATH:-default}"
echo "🛠️ COLMAP CMD           : ${COLMAP_CMD:-colmap}"

echo "────────────────────────────────────"
# ======================
# ENV DEBUG (NEW)
# ======================
export LOGLEVEL=DEBUG
export COLMAP_LOG_LEVEL=2

if [[ "$DEVICE" == "cpu" ]]; then
  echo "🧠 CPU MODE"
  export LIBGL_ALWAYS_SOFTWARE=1
  export QT_QPA_PLATFORM=offscreen
  export MPLBACKEND=Agg
  export CUDA_VISIBLE_DEVICES=""
  export OMP_NUM_THREADS=1
else
  export OMP_NUM_THREADS=3
  export MKL_NUM_THREADS=3
  export TORCH_NUM_THREADS=3

  # HLOC / DataLoader workers (matching pairs)
  export DATASET_WORKERS=3

  # CUDA memory (important pour SuperGlue + Nerfstudio)
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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

# ======================
# REQUIRED
# ======================

ARGS+=(--data "$DATA_DIR")
ARGS+=(--output-dir "$OUTPUT_DIR")

# ======================
# OPTIONAL SCALAR PARAMS
# ======================

[[ -n "$CAMERA_TYPE" ]] && \
  ARGS+=(--camera-type "$CAMERA_TYPE")

[[ -n "$MATCHING_METHOD" ]] && \
  ARGS+=(--matching-method "$MATCHING_METHOD")

[[ -n "$FEATURE_TYPE" ]] && \
  ARGS+=(--feature-type "$FEATURE_TYPE")

[[ -n "$MATCHER_TYPE" ]] && \
  ARGS+=(--matcher-type "$MATCHER_TYPE")

[[ -n "$NUM_DOWNSCALES" ]] && \
  ARGS+=(--num-downscales "$NUM_DOWNSCALES")

[[ -n "$SFMT_TOOL" ]] && \
  ARGS+=(--sfm-tool "$SFMT_TOOL")

[[ -n "$PERCENT_RADIUS_CROP" ]] && \
  ARGS+=(--percent-radius-crop "$PERCENT_RADIUS_CROP")

[[ -n "$COLMAP_MODEL_PATH" ]] && \
  ARGS+=(--colmap-model-path "$COLMAP_MODEL_PATH")

[[ -n "$COLMAP_CMD" ]] && \
  ARGS+=(--colmap-cmd "$COLMAP_CMD")

[[ -n "$IMAGES_PER_EQUIRECT" ]] && \
  ARGS+=(--images-per-equirect "$IMAGES_PER_EQUIRECT")

[[ -n "$CROP_BOTTOM" ]] && \
  ARGS+=(--crop-bottom "$CROP_BOTTOM")

# ======================
# BOOLEAN FLAGS
# ======================

[[ "$REFINE_INTRINSICS" == "true" ]] && \
  ARGS+=(--refine-intrinsics)

[[ "$REFINE_PIXSFM" == "true" ]] && \
  ARGS+=(--refine-pixsfm)

[[ "$USE_SINGLE_CAMERA_MODE" == "true" ]] && \
  ARGS+=(--use-single-camera-mode)

[[ "$SKIP_COLMAP" == "true" ]] && \
  ARGS+=(--skip-colmap)

[[ "$SKIP_IMAGE_PROCESSING" == "true" ]] && \
  ARGS+=(--skip-image-processing)

[[ "$USE_SFM_DEPTH" == "true" ]] && \
  ARGS+=(--use-sfm-depth)

[[ "$INCLUDE_DEPTH_DEBUG" == "true" ]] && \
  ARGS+=(--include-depth-debug)

[[ "$VERBOSE" == "true" ]] && \
  ARGS+=(--verbose)

# default = true → only add flag if explicitly false
[[ "$SAME_DIMENSIONS" == "false" ]] && \
  ARGS+=(--no-same-dimensions)

# ======================
# VECTOR PARAMS
# ======================

if [[ -n "$CROP_FACTOR" ]]; then
  read -r TOP BOTTOM LEFT RIGHT <<< "$CROP_FACTOR"

  ARGS+=(
    --crop-factor
    "$TOP"
    "$BOTTOM"
    "$LEFT"
    "$RIGHT"
  )
fi
# ======================
# EXEC WITH FULL STREAM LOGGING
# ======================

ns-process-data images "${ARGS[@]}" \
  > >(tee -a "$PROCESS_LOG") \
  2> >(tee -a "$PROCESS_LOG" >&2)

STATUS=$?

kill $HEARTBEAT_PID 2>/dev/null || true

set -e

# ======================
# FAILURE HANDLING
# ======================
if [ "$STATUS" -ne 0 ]; then
  echo "❌ PIPELINE FAILED (exit code: $STATUS)"
  tail -n 80 "$PROCESS_LOG"
  exit "$STATUS"
fi

# ======================
# VALIDATION
# ======================
TRANSFORMS="$OUTPUT_DIR/transforms.json"

if [ ! -f "$TRANSFORMS" ]; then
  echo "💀 SUCCESS BUT NO transforms.json"
  tail -n 80 "$PROCESS_LOG"
  exit 1
fi

# ======================
# COLMAP COVERAGE CHECK
# ======================
if [ -f "$PROCESS_LOG" ]; then

  COLMAP_PERCENT=""

  # ======================
  # Cas 1 : ancien format partiel
  # ======================
  COLMAP_PERCENT=$(grep "COLMAP only found poses" "$PROCESS_LOG" \
    | grep -oE '[0-9]+(\.[0-9]+)?' \
    | tail -n 1)

  # ======================
  # Cas 2 : all images
  # ======================
  if grep -q "COLMAP found poses for all images" "$PROCESS_LOG"; then
    COLMAP_PERCENT="100"
  fi

  # ======================
  # Cas 3 : percentage of poses
  # ex: COLMAP found poses for 98.67% of the images
  # ======================
  if [ -z "$COLMAP_PERCENT" ]; then
    COLMAP_PERCENT=$(grep -Eo "COLMAP found poses for [0-9]+(\.[0-9]+)?%" "$PROCESS_LOG" \
      | grep -Eo "[0-9]+(\.[0-9]+)?" \
      | tail -n 1)
  fi

  # ======================
  # VALIDATION
  # ======================
  if [ -n "$COLMAP_PERCENT" ]; then
    echo "📊 COLMAP pose coverage: ${COLMAP_PERCENT}%"

    if (( $(echo "$COLMAP_PERCENT < 70" | bc -l) )); then
      echo "❌ STOP: COLMAP coverage too low (<70%)"
      echo "📉 Failing pipeline to avoid bad reconstruction"
      tail -n 80 "$PROCESS_LOG"
      exit 1
    fi
  else
    echo "⚠️ Could not detect COLMAP coverage percentage"
  fi
fi

# ======================
# SILENT FAILURE DETECTION (NEW)
# ======================
LAST_LOG=$(tail -n 30 "$PROCESS_LOG")

if ! echo "$LAST_LOG" | grep -q -E "Finished|Done|Writing"; then
  echo "⚠️ Possible silent failure detected"
  tail -n 80 "$PROCESS_LOG"
fi


echo "✅ ns-process-data SUCCESS"
echo "📦 OUTPUT READY: $OUTPUT_DIR"
