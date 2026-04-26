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

export MODEL_IMPLEMENTATION=""
export MACHINE_DEVICE_TYPE=""

if [ "$DEVICE" = "gpu" ]; then
    MODEL_IMPLEMENTATION="tcnn"
    MACHINE_DEVICE_TYPE="cuda"
    export MAX_JOBS=2
elif [ "$DEVICE" = "cpu" ]; then
    MODEL_IMPLEMENTATION="torch"
    MACHINE_DEVICE_TYPE="cpu"
    export TORCHDYNAMO_DISABLE=1
    export OMP_NUM_THREADS=1

else
    echo "❌ CONFIGURATION ERROR: DEVICE unknown, should be cpu or gpu"
    exit 1
fi

export MODEL_IMPLEMENTATION
export MACHINE_DEVICE_TYPE

export EXPERIMENT_NAME="model3d"


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
echo "🔁 MAX ITERATIONS          : $MAX_ITER"
echo "📊 VIS MODE                : $TRAIN_VIS_MODE"

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
# COMMON ARGS
# (must come BEFORE nerfstudio-data)
# ======================
COMMON_ARGS=(
  --output-dir "$OUTPUTDIR"
  --experiment-name "$EXPERIMENT_NAME"
  --machine.device-type "$MACHINE_DEVICE_TYPE"
  --vis "$TRAIN_VIS_MODE"
  --max-num-iterations "$MAX_ITER"
  --steps-per-save 250
  --steps-per-eval-all-images 250
  --save-only-latest-checkpoint True
)

# ======================
# DEVICE-SPECIFIC ARGS
# (must come BEFORE nerfstudio-data)
# ======================
if [[ "$DEVICE" == "gpu" ]]; then

  DEVICE_ARGS=(
    --pipeline.datamanager.camera-res-scale-factor "$CAMERA_RES_SCALE_FACTOR"
  )

elif [[ "$DEVICE" == "cpu" ]]; then

  DEVICE_ARGS=(
    --pipeline.datamanager.train-num-rays-per-batch "$TRAIN_RAYS_PER_BATCH"
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
# RUN (CRASH SAFE)
# IMPORTANT: nerfstudio-data MUST be LAST
# ======================
set +e

ns-train \
  "$MODEL" \
  "${COMMON_ARGS[@]}" \
  "${DEVICE_ARGS[@]}" \
  nerfstudio-data \
  --data "$DATA"

STATUS=$?

set -e

if [[ "$STATUS" -ne 0 ]]; then
  echo "❌ ns-train crashed (exit code: $STATUS)"
  exit "$STATUS"
fi

echo "✅ TRAINING COMPLETE"
