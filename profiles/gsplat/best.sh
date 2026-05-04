########################################
# PERFORMANCE PROFILE - QUALITY+ (CONTROLLED)
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
# GAUSSIAN SPLATTING (CONTROLLED GROWTH)
########################################

# 🧠 ralentit la surdensification précoce
DENSIFY_GRAD_THRESH=0.00085

# 🧹 pruning un peu plus strict pour compenser la densification
CULL_ALPHA_THRESH=0.07

# 🧽 nettoie mieux les splats larges (stabilité visuelle)
CULL_SCREEN_SIZE=0.36

# 🎯 split légèrement plus conservateur (évite duplication inutile)
SPLIT_SCREEN_SIZE=0.011

########################################
# EXPORT QUALITY
########################################

EXPORT_NUM_POINTS=1500000
EXPORT_DOWNSAMPLE=1
