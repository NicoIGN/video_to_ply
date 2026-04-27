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

########################################
# TRAINING
########################################

MAX_ITER=6000
TRAIN_RAYS_PER_BATCH=512

NUM_NERF_SAMPLES_PER_RAY=64
NUM_PROPOSAL_SAMPLES_PER_RAY="128 64"

########################################
# EXPORT QUALITY
########################################

EXPORT_NUM_POINTS=2000000
EXPORT_DOWNSAMPLE=1

MAX_JOBS=4
