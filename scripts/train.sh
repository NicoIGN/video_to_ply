#!/bin/bash
set -e

# ======================
# LOAD CONFIG (CRITICAL)
# ======================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config/config.sh"


# ======================
# ENV FLAGS
# ======================
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1


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
# évite crash si non défini dans config
TRAIN_VIS_MODE=${TRAIN_VIS_MODE:-tensorboard}

export MODEL_IMPLEMENTATION=""
export MACHINE_DEVICE_TYPE=""
if [ "$DEVICE" = "gpu" ]; then
    MODEL_IMPLEMENTATION="tcnn"
    MACHINE_DEVICE_TYPE="cuda"
elif [ "$DEVICE" = "cpu" ]; then
    MODEL_IMPLEMENTATION="torch"
        MACHINE_DEVICE_TYPE="cpu"
else
    echo "❌ CONFIGURATION ERROR: DEVICE unknown, should be cpu or gpu"
    exit 1
fi

export MODEL_IMPLEMENTATION

#EXPERIMENT_NAME="$(basename "$OUTPUTDIR")"
export EXPERIMENT_NAME="model3d"

# ======================
# SUMMARY (IMPORTANT)
# ======================
echo "────────────────────────────────────────────"
echo "🚀 TRAINING CONFIG SUMMARY"
echo "────────────────────────────────────────────"

echo "📁 DATA                     : $DATA"
echo "📁 OUTPUTDIR                : $OUTPUTDIR"
echo "🧪 MODEL                    : $MODEL"
echo "🧪 MODEL_IMPLEMENTATION     : $MODEL_IMPLEMENTATION"
echo "🧪 EXPERIMENT_NAME         : $EXPERIMENT_NAME"
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

echo "💾 CHECKPOINTING"
echo "  - steps per save         : 50"
echo "  - steps per eval images  : 50"
echo "  - keep only latest       : true"

echo "────────────────────────────────────────────"
echo "🔥 STARTING TRAINING..."
echo "────────────────────────────────────────────"

# ======================
# TRAIN
# ======================

ns-train "$MODEL" \
    --output-dir "$OUTPUTDIR" \
    --experiment-name "$EXPERIMENT_NAME" \
    --machine.device-type "$MACHINE_DEVICE_TYPE" \
    --vis "$TRAIN_VIS_MODE" \
    --max-num-iterations "$MAX_ITER" \
    \
    --pipeline.datamanager.train-num-rays-per-batch "$TRAIN_RAYS_PER_BATCH" \
    --pipeline.datamanager.camera-res-scale-factor "$CAMERA_RES_SCALE_FACTOR" \
    \
    --pipeline.model.num-nerf-samples-per-ray "$NUM_NERF_SAMPLES_PER_RAY" \
    --pipeline.model.num-proposal-samples-per-ray $NUM_PROPOSAL_SAMPLES_PER_RAY \
    --pipeline.model.max-res "$MAX_RES" \
    --pipeline.model.predict-normals True \
    --pipeline.model.implementation "$MODEL_IMPLEMENTATION" \
    \
    --steps-per-save 50 \
    --steps-per-eval-all-images 50 \
    --save-only-latest-checkpoint True \
    \
    nerfstudio-data \
    --data "$DATA"
