#!/bin/bash
set -e

# ======================
# LOAD CONFIG (CRITICAL)
# ======================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config/config.sh"


# ======================
# SAFETY CHECKS
# ======================
if [ -z "$DATA" ]; then
  echo "❌ DATA is empty (check run.sh)"
  exit 1
fi

if [ ! -f "$DATA/transforms.json" ]; then
  echo "❌ Missing transforms.json in $DATA"
  exit 1
fi


# ======================
# DEFAULT VIS MODE FALLBACK
# ======================
TRAIN_VIS_MODE=${TRAIN_VIS_MODE:-tensorboard}

export MACHINE_DEVICE_TYPE=""

if [ "$DEVICE" = "gpu" ]; then
    MACHINE_DEVICE_TYPE="cuda"

elif [ "$DEVICE" = "cpu" ]; then
    MACHINE_DEVICE_TYPE="cpu"
    export TORCHDYNAMO_DISABLE=1
    export OMP_NUM_THREADS=1

else
    echo "❌ CONFIGURATION ERROR: DEVICE unknown, should be cpu or gpu"
    exit 1
fi

export MODEL_IMPLEMENTATION
export MACHINE_DEVICE_TYPE
export MAX_JOBS
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS

# ======================
# CUDA ARCH AUTO-DETECTION
# ======================

if command -v python3 >/dev/null 2>&1; then
  if python3 -c "import torch" >/dev/null 2>&1; then

    export TORCH_CUDA_ARCH_LIST=$(python3 - << 'EOF'
import torch

if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print(f"{cap[0]}.{cap[1]}")
EOF
)

    echo "⚙️ TORCH_CUDA_ARCH_LIST auto-set to: $TORCH_CUDA_ARCH_LIST"

  else
    echo "⚠️ torch not available in python, skipping TORCH_CUDA_ARCH_LIST"
  fi
else
  echo "⚠️ python3 not found, skipping TORCH_CUDA_ARCH_LIST"
fi

# ======================
# SUMMARY
# ======================
echo "────────────────────────────────────────────"
echo "🚀 TRAINING CONFIG SUMMARY"
echo "────────────────────────────────────────────"

echo "📁 DATA                     : $DATA"
echo "📁 OUTPUTDIR                : $OUTPUTDIR"
echo "🧪 MODEL                    : $MODEL"
echo "🧪 MODEL_IMPLEMENTATION     : $MODEL_IMPLEMENTATION"
echo "🧪 EXPERIMENT_NAME          : $EXPERIMENT_NAME"
echo "⚙️ DEVICE                   : $DEVICE"
echo "🔁 MAX ITERATIONS           : $MAX_ITER"
echo "🔁 MAX JOBS                 : $MAX_JOBS"
echo "📊 VIS MODE                 : $TRAIN_VIS_MODE"


echo "────────────────────────────────────────────"
echo "🧠 DATA PIPELINE"
echo "  - train rays/batch       : $TRAIN_RAYS_PER_BATCH"
echo "  - camera res scale       : $CAMERA_RES_SCALE_FACTOR"


echo "────────────────────────────────────────────"
echo "🧠 MODEL CONFIG"
echo "  - nerf samples/ray       : $NUM_NERF_SAMPLES_PER_RAY"
echo "  - proposal samples/ray   : $NUM_PROPOSAL_SAMPLES_PER_RAY"
echo "  - max resolution         : $MAX_RES"
echo "  - implementation         : $MODEL_IMPLEMENTATION"


echo "────────────────────────────────────────────"
echo "🔥 STARTING TRAINING..."
echo "────────────────────────────────────────────"


# ======================
# CHECKPOINT AUTO-RESUME
# ======================
LOAD_DIR=""

if [ -d "$OUTPUTDIR/nerfstudio_models" ]; then
    LOAD_DIR="$OUTPUTDIR/nerfstudio_models"
fi

if [ -d "$OUTPUTDIR" ]; then
    LAST_RUN=$(ls -td "$OUTPUTDIR"/*/nerfstudio_models 2>/dev/null | head -n 1)
    if [ ! -z "$LAST_RUN" ]; then
        LOAD_DIR="$LAST_RUN"
    fi
fi

if [ ! -z "$LOAD_DIR" ]; then
    echo "♻️ CHECKPOINT FOUND → RESUMING TRAINING"
    echo "📦 LOAD_DIR: $LOAD_DIR"
else
    echo "🆕 NO CHECKPOINT FOUND → TRAINING FROM SCRATCH"
fi


# ======================
# COMMON ARGS
# ======================
COMMON_ARGS=(
  --output-dir "$OUTPUTDIR"
  --experiment-name "$EXPERIMENT_NAME"
  --machine.device-type "$MACHINE_DEVICE_TYPE"
  --max-num-iterations "$MAX_ITER"
  --steps-per-save 500
  --steps-per-eval-all-images 500
  --save-only-latest-checkpoint True
  --vis "$TRAIN_VIS_MODE"
  --logging.local-writer.enable True
  --logging.steps-per-log 10
  --viewer.quit-on-train-completion True
)


# ======================
# DEVICE-SPECIFIC ARGS
# ======================
if [[ "$DEVICE" == "gpu" ]]; then

  DEVICE_ARGS=(
    --pipeline.datamanager.camera-res-scale-factor "$CAMERA_RES_SCALE_FACTOR"
  )

elif [[ "$DEVICE" == "cpu" ]]; then

  DEVICE_ARGS=(
    --pipeline.datamanager.camera-res-scale-factor "$CAMERA_RES_SCALE_FACTOR"
    --pipeline.model.implementation "$MODEL_IMPLEMENTATION"
    --pipeline.model.num-nerf-samples-per-ray "$NUM_NERF_SAMPLES_PER_RAY"
    --pipeline.model.num-proposal-samples-per-ray $NUM_PROPOSAL_SAMPLES_PER_RAY
    --pipeline.model.max-res "$MAX_RES"
    --pipeline.model.predict-normals True
  )

else
  echo "❌ DEVICE must be cpu or gpu"
  exit 1
fi


# ======================
# ADD ROBUST LOGGING + SILENT FAILURE DETECTION
# Place BEFORE "RUN TRAINING"
# ======================

LOG_DIR="$OUTPUTDIR/logs"
mkdir -p "$LOG_DIR"

TRAIN_LOG="$LOG_DIR/ns_train.log"
HEARTBEAT_LOG="$LOG_DIR/ns_train_heartbeat.log"

echo "📝 Full training log: $TRAIN_LOG"
echo "💓 Heartbeat log: $HEARTBEAT_LOG"

(
  while true; do
    sleep 60
    echo "$(date '+%F %T') ns-train still active" >> "$HEARTBEAT_LOG"
  done
) &
HEARTBEAT_PID=$!


# ======================
# RUN TRAINING
# ======================

export LOGLEVEL=DEBUG
export TORCH_SHOW_CPP_STACKTRACES=1

set +e

if [ ! -z "$LOAD_DIR" ]; then
  ns-train \
    "$MODEL" \
    "${COMMON_ARGS[@]}" \
    "${DEVICE_ARGS[@]}" \
    --load-dir "$LOAD_DIR" \
    nerfstudio-data \
    --data "$DATA" \
    > >(tee -a "$TRAIN_LOG") \
    2> >(tee -a "$TRAIN_LOG" >&2)
else
  ns-train \
    "$MODEL" \
    "${COMMON_ARGS[@]}" \
    "${DEVICE_ARGS[@]}" \
    nerfstudio-data \
    --data "$DATA" \
    > >(tee -a "$TRAIN_LOG") \
    2> >(tee -a "$TRAIN_LOG" >&2)
fi

STATUS=$?

kill $HEARTBEAT_PID 2>/dev/null || true

set -e


# ======================
# CHECK RESULT
# ======================
if [[ "$STATUS" -ne 0 ]]; then
  echo "❌ ns-train crashed (exit code: $STATUS)"
  tail -50 "$TRAIN_LOG"
  exit "$STATUS"
fi


# ======================
# EXTRA SILENT STOP DETECTION
# ======================
LAST_LOG_LINE=$(tail -n 20 "$TRAIN_LOG")

if ! echo "$LAST_LOG_LINE" | grep -q "Training Finished"; then
  echo "⚠️ ns-train may have stopped unexpectedly before proper completion"
  tail -50 "$TRAIN_LOG"
fi


# ======================
# CHECKPOINT VALIDATION
# ======================
CKPT_DIR=$(find "$OUTPUTDIR" -type d -name "nerfstudio_models" 2>/dev/null | head -n 1)

if [[ -z "$CKPT_DIR" ]]; then
  echo "⚠️ Training finished but no checkpoint found"
  exit 1
fi

echo "✅ TRAINING COMPLETE"
echo "📦 Checkpoint directory: $CKPT_DIR"
