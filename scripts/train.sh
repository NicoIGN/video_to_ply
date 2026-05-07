#!/bin/bash
set -e

# ======================
# LOAD CONFIG (CRITICAL)
# ======================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/../config/config.sh"

# ======================
# UTILS
# ======================

add_arg() {
  local array_name="$1"
  local flag="$2"
  local value="$3"

  if [[ -n "${value:-}" ]]; then
    eval "$array_name+=(\"\$flag\" \"\$value\")"
  fi
}

add_bool_arg() {
  local array_name="$1"
  local flag="$2"
  local value="${3-}"

  [[ -z "$value" ]] && return

  case "$value" in
    True|true|1)   value="True" ;;
    False|false|0) value="False" ;;
    *) echo "⚠️ Invalid boolean value for $flag: $value"; return ;;
  esac

  eval "$array_name+=(\"\$flag\" \"\$value\")"
}

add_multi_arg() {
  local array_name="$1"
  local flag="$2"
  shift 2

  if [[ $# -gt 0 ]]; then
    eval "$array_name+=(\"\$flag\")"

    for arg in "$@"; do
      eval "$array_name+=(\"\$arg\")"
    done
  fi
}

# ======================
# SAFETY CHECKS
# ======================

if [[ -z "${DATA:-}" ]]; then
  echo "❌ DATA is empty (check run.sh)"
  exit 1
fi

if [[ ! -f "$DATA/transforms.json" ]]; then
  echo "❌ Missing transforms.json in $DATA"
  exit 1
fi

# ======================
# DEFAULTS
# ======================

TRAIN_VIS_MODE=${TRAIN_VIS_MODE:-tensorboard}

STEPS_PER_SAVE=${STEPS_PER_SAVE:-2000}

STEPS_PER_EVAL_ALL_IMAGES=${STEPS_PER_EVAL_ALL_IMAGES:-2000}

REFINE_EVERY=${REFINE_EVERY:-500}

STEPS_PER_LOG=${STEPS_PER_LOG:-250}

# ======================
# DEVICE CONFIG
# ======================

if [[ "$DEVICE" == "gpu" ]]; then

  MACHINE_DEVICE_TYPE="cuda"

elif [[ "$DEVICE" == "cpu" ]]; then

  MACHINE_DEVICE_TYPE="cpu"

  export TORCHDYNAMO_DISABLE=1
  export OMP_NUM_THREADS=1

else
  echo "❌ CONFIGURATION ERROR: DEVICE must be cpu or gpu"
  exit 1
fi

export MACHINE_DEVICE_TYPE
export MODEL_IMPLEMENTATION
export MAX_JOBS

export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"

export TORCH_DISABLE_ADDR2LINE=1
export TORCHINDUCTOR_DISABLE=1
export TORCH_COMPILE_DISABLE=1

# ======================
# CUDA ARCH AUTO-DETECTION
# ======================

if command -v python3 >/dev/null 2>&1; then

  if python3 -c "import torch" >/dev/null 2>&1; then

    export TORCH_CUDA_ARCH_LIST=$(
      python3 - << 'EOF'
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
echo "💾 STEPS PER SAVE           : $STEPS_PER_SAVE"
echo "🖼️ STEPS PER EVAL ALL IMG   : $STEPS_PER_EVAL_ALL_IMAGES"
echo "🧪 REFINE EVERY             : $REFINE_EVERY"

echo "────────────────────────────────────────────"
echo "🧠 DATA PIPELINE"

echo "  - train rays per batch    : $TRAIN_RAYS_PER_BATCH"
echo "  - camera resolution scale : $CAMERA_RES_SCALE_FACTOR"

echo "────────────────────────────────────────────"
echo "🧠 MODEL CONFIG"

echo "  - NeRF samples per ray    : $NUM_NERF_SAMPLES_PER_RAY"
echo "  - proposal samples/ray    : $NUM_PROPOSAL_SAMPLES_PER_RAY"
echo "  - max resolution          : $MAX_RES"
echo "  - implementation          : $MODEL_IMPLEMENTATION"

echo "────────────────────────────────────────────"
echo "✨ GAUSSIAN SPLATTING"

echo "  - densify grad threshold  : $DENSIFY_GRAD_THRESH"
echo "  - cull alpha threshold    : $CULL_ALPHA_THRESH"
echo "  - cull screen size        : $CULL_SCREEN_SIZE"
echo "  - split screen size       : $SPLIT_SCREEN_SIZE"

echo "────────────────────────────────────────────"
echo "🧱 COLLIDER"

echo "  - near plane              : $COLLIDER_NEAR"
echo "  - far plane               : $COLLIDER_FAR"
echo "  - enable collider         : $ENABLE_COLLIDER"

echo "────────────────────────────────────────────"
echo "🔥 STARTING TRAINING..."
echo "────────────────────────────────────────────"

# ======================
# CHECKPOINT AUTO-RESUME
# ======================

echo "────────────────────────────────────────────"
echo "🔍 CHECKPOINT AUTO-RESUME"
echo "────────────────────────────────────────────"

LOAD_DIR=""

BASE_DIR="$OUTPUTDIR/$EXPERIMENT_NAME/$MODEL"

echo "📂 BASE_DIR : $BASE_DIR"

if [[ -d "$BASE_DIR/nerfstudio_models" ]]; then

  LOAD_DIR="$BASE_DIR/nerfstudio_models"

  echo "✅ Direct checkpoint found"
  echo "📦 $LOAD_DIR"

fi

if [[ -d "$BASE_DIR" ]]; then

  LAST_RUN=$(ls -td "$BASE_DIR"/*/nerfstudio_models 2>/dev/null | head -n 1)

  if [[ -n "$LAST_RUN" ]]; then

    LOAD_DIR="$LAST_RUN"

    echo "✅ Latest checkpoint found"
    echo "📦 $LOAD_DIR"

  fi
fi

# disable resume
LOAD_DIR=""

# ======================
# RUN TIMESTAMP
# ======================

if [[ -n "$LOAD_DIR" ]]; then

  RUN_TIMESTAMP="$(basename "$(dirname "$LOAD_DIR")")"

  echo "♻️ Resuming existing run"
  echo "🕒 TIMESTAMP : $RUN_TIMESTAMP"

else

  RUN_TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"

  echo "🆕 Starting new run"
  echo "🕒 TIMESTAMP : $RUN_TIMESTAMP"

fi

# ======================
# COMMON ARGS
# ======================

COMMON_ARGS=()

add_arg COMMON_ARGS --output-dir "$OUTPUTDIR"
add_arg COMMON_ARGS --experiment-name "$EXPERIMENT_NAME"
add_arg COMMON_ARGS --machine.device-type "$MACHINE_DEVICE_TYPE"
add_arg COMMON_ARGS --max-num-iterations "$MAX_ITER"
add_arg COMMON_ARGS --steps-per-save "$STEPS_PER_SAVE"
add_arg COMMON_ARGS --steps-per-eval-all-images "$STEPS_PER_EVAL_ALL_IMAGES"
add_arg COMMON_ARGS --vis "$TRAIN_VIS_MODE"
add_arg COMMON_ARGS --logging.steps-per-log "$STEPS_PER_LOG"

add_bool_arg COMMON_ARGS --save-only-latest-checkpoint True
add_bool_arg COMMON_ARGS --logging.local-writer.enable True
add_bool_arg COMMON_ARGS --viewer.quit-on-train-completion True
add_bool_arg COMMON_ARGS --mixed-precision True
add_bool_arg COMMON_ARGS --use-grad-scaler True

add_arg COMMON_ARGS --load-dir "$LOAD_DIR"

# ======================
# DEVICE ARGS
# ======================

DEVICE_ARGS=()

unset DENSIFY_GRAD_THRESH
unset CULL_ALPHA_THRESH
unset CULL_SCREEN_SIZE
unset SPLIT_SCREEN_SIZE
  
if [[ "$DEVICE" == "gpu" ]]; then

  add_arg DEVICE_ARGS       --pipeline.datamanager.camera-res-scale-factor "$CAMERA_RES_SCALE_FACTOR"
  add_arg DEVICE_ARGS       --pipeline.datamanager.cache-images gpu
  add_bool_arg DEVICE_ARGS  --pipeline.datamanager.images-on-gpu True
  add_bool_arg DEVICE_ARGS  --pipeline.datamanager.masks-on-gpu False
  add_arg DEVICE_ARGS       --pipeline.model.densify-grad-thresh "$DENSIFY_GRAD_THRESH"
  add_arg DEVICE_ARGS       --pipeline.model.cull-alpha-thresh "$CULL_ALPHA_THRESH"
  add_arg DEVICE_ARGS       --pipeline.model.cull-screen-size "$CULL_SCREEN_SIZE"
  add_arg DEVICE_ARGS       --pipeline.model.split-screen-size "$SPLIT_SCREEN_SIZE"
  add_arg DEVICE_ARGS       --pipeline.model.refine-every "$REFINE_EVERY"
  add_bool_arg DEVICE_ARGS  --pipeline.model.use-bilateral-grid "$USE_BILATERAL_GRID"
  add_bool_arg DEVICE_ARGS  --pipeline.model.use-scale-regularization "$USE_SCALE_REGULARIZATION"
  add_arg DEVICE_ARGS       --pipeline.model.max-gauss-ratio "$MAX_GAUSS_RATIO"
  add_arg DEVICE_ARGS       --pipeline.model.stop-split-at "$STOP_SPLIT_AT"
  add_arg DEVICE_ARGS       --pipeline.model.cull-scale-thresh "$CULL_SCALE_THRESH"
  add_arg DEVICE_ARGS       --pipeline.model.reset-alpha-every "$RESET_ALPHA_EVERY"
  add_arg DEVICE_ARGS       --pipeline.model.ssim-lambda "$SSIM_LAMBDA"
  add_bool_arg DEVICE_ARGS  --pipeline.model.enable-collider "$ENABLE_COLLIDER"

  if [[ "$ENABLE_COLLIDER" == "True" ]]; then
    if [[ -n "${COLLIDER_NEAR:-}" ]]; then
      add_multi_arg DEVICE_ARGS --pipeline.model.collider-params near_plane   "$COLLIDER_NEAR"
    fi

    if [[ -n "${COLLIDER_FAR:-}" ]]; then
      add_multi_arg DEVICE_ARGS --pipeline.model.collider-params far_plane    "$COLLIDER_FAR"
    fi
  fi

elif [[ "$DEVICE" == "cpu" ]]; then
  add_arg DEVICE_ARGS --pipeline.datamanager.camera-res-scale-factor "$CAMERA_RES_SCALE_FACTOR"
  add_arg DEVICE_ARGS --pipeline.model.implementation "$MODEL_IMPLEMENTATION"
  add_arg DEVICE_ARGS --pipeline.model.num-nerf-samples-per-ray "$NUM_NERF_SAMPLES_PER_RAY"
  add_arg DEVICE_ARGS --pipeline.model.max-res "$MAX_RES"
  add_bool_arg DEVICE_ARGS --pipeline.model.predict-normals True

  if [[ -n "${NUM_PROPOSAL_SAMPLES_PER_RAY:-}" ]]; then
    add_multi_arg DEVICE_ARGS --pipeline.model.num-proposal-samples-per-ray $NUM_PROPOSAL_SAMPLES_PER_RAY
  fi
fi

echo "DEVICE_ARGS: $DEVICE_ARGS"

# ======================
# LOGGING
# ======================

LOG_DIR="$OUTPUTDIR/logs"

mkdir -p "$LOG_DIR"

TRAIN_LOG="$LOG_DIR/ns_train.log"

HEARTBEAT_LOG="$LOG_DIR/ns_train_heartbeat.log"

echo "📝 Full training log : $TRAIN_LOG"
echo "💓 Heartbeat log     : $HEARTBEAT_LOG"

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

ns-train \
  "$MODEL" \
  "${COMMON_ARGS[@]}" \
  "${DEVICE_ARGS[@]}" \
  nerfstudio-data \
  --data "$DATA" \
  > >(tee -a "$TRAIN_LOG") \
  2> >(tee -a "$TRAIN_LOG" >&2)

STATUS=$?

kill "$HEARTBEAT_PID" 2>/dev/null || true

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
# SILENT FAILURE DETECTION
# ======================

LAST_LOG_LINE=$(tail -n 20 "$TRAIN_LOG")

if ! echo "$LAST_LOG_LINE" | grep -q "Training Finished"; then

  echo "⚠️ ns-train may have stopped unexpectedly"

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
