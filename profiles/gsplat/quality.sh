########################################
# PERFORMANCE PROFILE - QUALITY FIXED
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
NUM_DOWNSCALES=1
SKIP_IMAGE_PROCESSING=true
MAX_JOBS=2

########################################
# TRAINING
########################################

MAX_ITER=16000
TRAIN_RAYS_PER_BATCH=1024

NUM_NERF_SAMPLES_PER_RAY=64        # (stable run = 64, plus stable que 96)
NUM_PROPOSAL_SAMPLES_PER_RAY="128 64"

########################################
# GAUSSIAN SPLATTING (STABLE + HIGH QUALITY)
########################################

# densification (stable run = 0.001, meilleur équilibre)
DENSIFY_GRAD_THRESH=0.001

# suppression floaters (stable)
CULL_ALPHA_THRESH=0.05

# important: valeur stable (pas trop agressif)
CULL_SCREEN_SIZE=0.25

# split (stable run = 0.015 → plus cohérent que 0.01)
SPLIT_SCREEN_SIZE=0.015

# refine plus fréquent (meilleur tracking géométrique)
REFINE_EVERY=100

STOP_SPLIT_AT=8000

# stabilité globale (identique stable run)
CULL_SCALE_THRESH=0.5
RESET_ALPHA_EVERY=30

########################################
# QUALITY / REGULARIZATION
########################################

USE_BILATERAL_GRID=True
USE_SCALE_REGULARIZATION=True

MAX_GAUSS_RATIO=5
SSIM_LAMBDA=0.2

########################################
# EXPORT QUALITY
########################################

EXPORT_NUM_POINTS=4000000
EXPORT_DOWNSAMPLE=1
