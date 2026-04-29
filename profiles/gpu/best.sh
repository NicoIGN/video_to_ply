########################################
# PERFORMANCE PROFILE - QUALITY+
########################################

TRAINING_PROFILE="gpu/quality_plus"

DEVICE="gpu"
MODEL="splatfacto"
MODEL_IMPLEMENTATION="tcnn"
TRAIN_VIS_MODE="tensorboard"

########################################
# IMAGE / PREPROCESSING
########################################

CAMERA_RES_SCALE_FACTOR=1.0
MAX_RES=1536
NUM_DOWNSCALES=0
SKIP_IMAGE_PROCESSING=true
MAX_JOBS=2

########################################
# TRAINING
########################################

MAX_ITER=12000
TRAIN_RAYS_PER_BATCH=1024

NUM_NERF_SAMPLES_PER_RAY=96
NUM_PROPOSAL_SAMPLES_PER_RAY="256 128"

DENSIFY_GRAD_THRESH=0.0004
CULL_ALPHA_THRESH=0.05
CULL_SCREEN_SIZE=0.3
SPLIT_SCREEN_SIZE=0.02


########################################
# EXPORT QUALITY
########################################

EXPORT_NUM_POINTS=5000000
EXPORT_DOWNSAMPLE=1
