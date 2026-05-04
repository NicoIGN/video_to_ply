########################################
# PERFORMANCE PROFILE - QUALITY+ (STABLE)
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
# GAUSSIAN SPLATTING (BALANCED)
########################################

# ⚖️ moins permissif → évite explosion de splats
DENSIFY_GRAD_THRESH=0.0007

# 🧹 prune plus efficacement le bruit invisible
CULL_ALPHA_THRESH=0.065

# 🧽 supprime davantage les gros splats inutiles
CULL_SCREEN_SIZE=0.34

# 🎯 split plus contrôlé → garde détail sans surdensifier
SPLIT_SCREEN_SIZE=0.012

########################################
# EXPORT QUALITY
########################################

# ⚠️ aligné avec un budget réaliste de splats
EXPORT_NUM_POINTS=2000000
EXPORT_DOWNSAMPLE=1
