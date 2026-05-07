########################################
# PERFORMANCE PROFILE - QUALITY ONLY (FIXED)
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
MAX_RES=1536
NUM_DOWNSCALES=0
SKIP_IMAGE_PROCESSING=true
MAX_JOBS=2

########################################
# TRAINING
########################################

MAX_ITER=16000
TRAIN_RAYS_PER_BATCH=1024

NUM_NERF_SAMPLES_PER_RAY=96
NUM_PROPOSAL_SAMPLES_PER_RAY="128 96"

########################################
# GAUSSIAN SPLATTING (STABLE + CLEAN)
########################################

# densification (réduit pour éviter explosion de splats)
DENSIFY_GRAD_THRESH=0.0025

# meilleure suppression des floaters
CULL_ALPHA_THRESH=0.04

# moins de blobs étirés (réduit ghosting)
CULL_SCREEN_SIZE=0.15

# split plus fin mais moins agressif globalement
SPLIT_SCREEN_SIZE=0.01

# meilleure convergence géométrique
REFINE_EVERY=300
STOP_SPLIT_AT=9000

# stabilité globale
CULL_SCALE_THRESH=0.35
RESET_ALPHA_EVERY=50

########################################
# QUALITY / REGULARIZATION
########################################

USE_BILATERAL_GRID=true
USE_SCALE_REGULARIZATION=true

MAX_GAUSS_RATIO=10
SSIM_LAMBDA=0.32

########################################
# EXPORT QUALITY
########################################

EXPORT_NUM_POINTS=4000000
EXPORT_DOWNSAMPLE=1
