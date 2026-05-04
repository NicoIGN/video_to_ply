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

MAX_ITER=20000
TRAIN_RAYS_PER_BATCH=768

NUM_NERF_SAMPLES_PER_RAY=96
NUM_PROPOSAL_SAMPLES_PER_RAY="192 96"

########################################
# GAUSSIAN SPLATTING
########################################

# légèrement plus stable à haute densité
DENSIFY_GRAD_THRESH=0.00035

# meilleure conservation des micro-détails sans bruit excessif
CULL_ALPHA_THRESH=0.045

# évite accumulation de splats trop larges en haute résolution
CULL_SCREEN_SIZE=0.28

# split un peu plus agressif pour mieux capter les détails fins
SPLIT_SCREEN_SIZE=0.018

########################################
# EXPORT QUALITY
########################################

EXPORT_NUM_POINTS=6000000
EXPORT_DOWNSAMPLE=1
