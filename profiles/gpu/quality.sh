########################################
# PERFORMANCE PROFILE - QUALITY ONLY
########################################

TRAINING_PROFILE="gpu/quality"

DEVICE="gpu"
MODEL="splatfacto"
MODEL_IMPLEMENTATION="tcnn"
TRAIN_VIS_MODE="tensorboard"

########################################
# IMAGE / PREPROCESSING
########################################

CAMERA_RES_SCALE_FACTOR=1.0
MAX_RES=1024
NUM_DOWNSCALES=1
SKIP_IMAGE_PROCESSING=true
MAX_JOBS=2

########################################
# TRAINING
########################################

MAX_ITER=6000
TRAIN_RAYS_PER_BATCH=512

NUM_NERF_SAMPLES_PER_RAY=64
NUM_PROPOSAL_SAMPLES_PER_RAY="128 64"

########################################
# GAUSSIAN SPLATTING
########################################

# densification (plus stable en high-res training)
DENSIFY_GRAD_THRESH=0.001

# keep fine structures longer (better thin geometry / edges)
CULL_ALPHA_THRESH=0.05

# slightly more aggressive pruning of oversized splats
CULL_SCREEN_SIZE=0.25

# earlier splitting for higher geometric precision
SPLIT_SCREEN_SIZE=0.015

########################################
# EXPORT QUALITY
########################################

EXPORT_NUM_POINTS=2500000
EXPORT_DOWNSAMPLE=1
