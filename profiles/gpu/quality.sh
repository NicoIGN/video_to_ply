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

# Gaussian densification threshold
# Lower = more detail, more VRAM, slower training
# Default: 0.0008
DENSIFY_GRAD_THRESH=0.0004

# Alpha culling threshold
# Lower = preserves faint/small gaussians longer
# Default: 0.1
CULL_ALPHA_THRESH=0.05

# Large splat culling threshold
# Higher = removes oversized gaussians more aggressively
# Default: 0.15
CULL_SCREEN_SIZE=0.3

# Large splat split threshold
# Lower = splits oversized gaussians earlier
# Default: 0.05
SPLIT_SCREEN_SIZE=0.02

########################################
# EXPORT QUALITY
########################################

EXPORT_NUM_POINTS=2000000
EXPORT_DOWNSAMPLE=1

