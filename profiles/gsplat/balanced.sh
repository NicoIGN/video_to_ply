########################################
# PERFORMANCE PROFILE
########################################

TRAINING_PROFILE="gpu/balanced"

DEVICE="gpu"
MODEL="splatfacto"
MODEL_IMPLEMENTATION="tcnn"
TRAIN_VIS_MODE="tensorboard"

########################################
# IMAGE / PREPROCESSING
########################################

CAMERA_RES_SCALE_FACTOR=0.5
MAX_RES=512
NUM_DOWNSCALES=1
SKIP_IMAGE_PROCESSING=true
MAX_JOBS=2

########################################
# TRAINING
########################################

MAX_ITER=3000
TRAIN_RAYS_PER_BATCH=256

NUM_NERF_SAMPLES_PER_RAY=24
NUM_PROPOSAL_SAMPLES_PER_RAY="48 24"

########################################
# GAUSSIAN SPLATTING
########################################

DENSIFY_GRAD_THRESH=0.0005
CULL_ALPHA_THRESH=0.05
CULL_SCREEN_SIZE=0.3
SPLIT_SCREEN_SIZE=0.02

########################################
# EXPORT BALANCED
########################################

EXPORT_NUM_POINTS=400000
EXPORT_DOWNSAMPLE=2
