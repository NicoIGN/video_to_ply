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
TRAIN_VIS_MODE=${TRAIN_VIS_MODE:-none}


# ======================
# TRAIN
# ======================
ns-train "$MODEL" \
  --data "$DATA" \
  --machine.device-type "$DEVICE" \
  --max-num-iterations "$MAX_ITER" \
  --experiment-name "$(basename "$OUTPUT")" \
  \
  --vis "$TRAIN_VIS_MODE" \
  \
  --pipeline.datamanager.train-num-rays-per-batch "$TRAIN_RAYS_PER_BATCH" \
  --pipeline.datamanager.camera-res-scale-factor "$CAMERA_RES_SCALE_FACTOR" \
  \
  --pipeline.model.num-nerf-samples-per-ray "$NUM_NERF_SAMPLES_PER_RAY" \
  --pipeline.model.num-proposal-samples-per-ray $NUM_PROPOSAL_SAMPLES_PER_RAY \
  --pipeline.model.max-res "$MAX_RES" \
  \
  --pipeline.model.implementation "$MODEL_IMPLEMENTATION"
