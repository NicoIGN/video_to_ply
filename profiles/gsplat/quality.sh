########################################
# PERFORMANCE PROFILE - QUALITY STABLE
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

# + long mais pas inutilement
if [ -z "${MAX_ITER+x}" ]; then
  MAX_ITER=7000
fi

# compromis bruit / stabilité
TRAIN_RAYS_PER_BATCH=512

# meilleur signal géométrique
NUM_NERF_SAMPLES_PER_RAY=48
NUM_PROPOSAL_SAMPLES_PER_RAY="96 48"

########################################
# GAUSSIAN SPLATTING (QUALITY CONTROLLED)
########################################

# 🔥 DENSIFICATION (équilibrée)
DENSIFY_GRAD_THRESH=0.0007
# (plus bas que ton quality → permet split utile, sans explosion)

# 🧹 CLEANING (garde le détail fin)
CULL_ALPHA_THRESH=0.06

# 📏 SPATIAL CONTROL (évite blobs)
CULL_SCREEN_SIZE=0.22
SPLIT_SCREEN_SIZE=0.018

# ⚡ FREQUENCY (clé pour qualité propre)
REFINE_EVERY=200

# 🛑 STOP split avant bruit
STOP_SPLIT_AT=6500

# 🧠 STABILISATION
RESET_ALPHA_EVERY=35
CULL_SCALE_THRESH=0.45

########################################
# QUALITY / REGULARIZATION
########################################

USE_BILATERAL_GRID=true
USE_SCALE_REGULARIZATION=true

MAX_GAUSS_RATIO=3.5
SSIM_LAMBDA=0.3

########################################
# EXPORT QUALITY
########################################

EXPORT_NUM_POINTS=1500000
EXPORT_DOWNSAMPLE=1
