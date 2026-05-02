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
echo "📷 CAMERA TYPE           : $CAMERA_TYPE"
echo "🔀 MATCHING METHOD       : $MATCHING_METHOD"
echo "🧠 SFM TOOL             : $SFMT_TOOL"
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
ARGS+=(--data "$DATA_DIR")
ARGS+=(--output-dir "$OUTPUT_DIR")
ARGS+=(--camera-type "$CAMERA_TYPE")
ARGS+=(--matching-method "$MATCHING_METHOD")
ARGS+=(--feature-type "$FEATURE_TYPE")
ARGS+=(--matcher-type "$MATCHER_TYPE")
ARGS+=(--num-downscales "$NUM_DOWNSCALES")
ARGS+=(--percent-radius-crop "$PERCENT_RADIUS_CROP")
ARGS+=(--refine-intrinsics)
ARGS+=(--sfm-tool "$SFMT_TOOL")

if [[ "$SFMT_TOOL" == "colmap" && "$USE_SINGLE_CAMERA_MODE" == "true" ]]; then
  ARGS+=(--use-single-camera-mode)
fi

if [[ -n "$CROP_FACTOR" ]]; then
  ARGS+=(--crop-factor $CROP_FACTOR)
fi

# ======================
# EXEC WITH FULL STREAM LOGGING (NEW)
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
# COLMAP COVERAGE CHECK (NEW)
# ======================
if [ -f "$PROCESS_LOG" ]; then
  COLMAP_PERCENT=$(grep "COLMAP only found poses" "$PROCESS_LOG" \
    | grep -oE '[0-9]+(\.[0-9]+)?' \
    | tail -n 1)

  if [ -n "$COLMAP_PERCENT" ]; then
    echo "📊 COLMAP pose coverage: $COLMAP_PERCENT%"

    # comparaison float-safe
    LOW=$(echo "$COLMAP_PERCENT < 70" | bc -l)

    if [ "$LOW" -eq 1 ]; then
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
