########################################
# PERFORMANCE PROFILE 
########################################

TRAINING_PROFILE="balanced"

########################################
# IMAGE / PREPROCESSING
########################################

CAMERA_RES_SCALE_FACTOR=0.75
MAX_RES=512
NUM_DOWNSCALES=1
SKIP_IMAGE_PROCESSING=true

########################################
# TRAINING
########################################

MAX_ITER=3000
TRAIN_RAYS_PER_BATCH=256

NUM_NERF_SAMPLES_PER_RAY=32
NUM_PROPOSAL_SAMPLES_PER_RAY="64 32"

########################################
# EXPORT BALANCED
########################################

EXPORT_NUM_POINTS=500000
EXPORT_DOWNSAMPLE=1
EXPORT_REMOVE_OUTLIERS=true

