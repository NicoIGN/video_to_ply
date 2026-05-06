########################################
# PERFORMANCE PROFILE - QUALITY STABLE
########################################

TRAINING_PROFILE="gpu/quality_stable_4Mcap"

DEVICE="gpu"
MODEL="splatfacto"
MODEL_IMPLEMENTATION="tcnn"
TRAIN_VIS_MODE="tensorboard"

########################################
# IMAGE / PREPROCESSING
########################################

CAMERA_RES_SCALE_FACTOR=1.0
MAX_RES=2048
NUM_DOWNSCALES=0
SKIP_IMAGE_PROCESSING=true
MAX_JOBS=2

########################################
# TRAINING
########################################

MAX_ITER=18000
REFINE_EVERY=200
TRAIN_RAYS_PER_BATCH=512

NUM_NERF_SAMPLES_PER_RAY=64
NUM_PROPOSAL_SAMPLES_PER_RAY="128 64"

########################################
# GAUSSIAN SPLATTING - STABILITY FIRST
########################################

# densification plus stricte → évite explosion
DENSIFY_GRAD_THRESH=0.002

# supprime plus agressivement les faibles contributions
CULL_ALPHA_THRESH=0.12

# évite accumulation de petits splats visibles
CULL_SCREEN_SIZE=0.15

# split plus contrôlé (évite cascade)
SPLIT_SCREEN_SIZE=0.025

# stop split plus tôt → stabilise la structure
STOP_SPLIT_AT=10000

########################################
# HARD STABILITY LIMITS (IMPORTANT)
########################################

# contrôle indirect de la croissance
MAX_GAUSS_RATIO=5

# évite formes trop extrêmes
CULL_SCALE_THRESH=0.5

# empêche explosion tardive
RESET_ALPHA_EVERY=30

########################################
# QUALITY / REGULARIZATION
########################################

SSIM_LAMBDA=0.2
USE_SCALE_REGULARIZATION=true
MAX_GAUSS_RATIO=5

########################################
# EXPORT QUALITY
########################################

EXPORT_NUM_POINTS=3500000
EXPORT_DOWNSAMPLE=1
