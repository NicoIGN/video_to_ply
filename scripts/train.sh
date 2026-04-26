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


# ======================
# SUMMARY (IMPORTANT)
# ======================
echo "────────────────────────────────────────────"
echo "🚀 TRAINING CONFIG SUMMARY"
echo "────────────────────────────────────────────"

echo "📁 DATA                     : $DATA"
echo "📁 OUTPUTDIR                : $OUTPUTDIR"
echo "🧪 MODEL                    : $MODEL"
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
#EXPERIMENT_NAME="$(basename "$OUTPUTDIR")"
EXPERIMENT_NAME="outputs"

ns-train "$MODEL" \
  --data "$DATA" \
  --output-dir "$OUTPUTDIR" \
  --machine.device-type "$DEVICE" \
  --max-num-iterations "$MAX_ITER" \
  --experiment-name "$EXPERIMENT_NAME" \
  \
  --vis "$TRAIN_VIS_MODE" \
  \
  --pipeline.datamanager.train-num-rays-per-batch "$TRAIN_RAYS_PER_BATCH" \
  --pipeline.datamanager.camera-res-scale-factor "$CAMERA_RES_SCALE_FACTOR" \
  \
  --pipeline.model.num-nerf-samples-per-ray "$NUM_NERF_SAMPLES_PER_RAY" \
  --pipeline.model.num-proposal-samples-per-ray $NUM_PROPOSAL_SAMPLES_PER_RAY \
  --pipeline.model.max-res "$MAX_RES" \
  --pipeline.model.predict-normals True \
  \
  --pipeline.model.implementation "$MODEL_IMPLEMENTATION" \
  \
  --steps-per-save 50 \
  --steps-per-eval-all-images 50 \
  --save-only-latest-checkpoint True
