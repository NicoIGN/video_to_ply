########################################
# PERFORMANCE PROFILE
########################################

TRAINING_PROFILE="gpu/quality"

DEVICE="gpu"
MODEL="splatfacto"
MODEL_IMPLEMENTATION="tcnn"
TRAIN_VIS_MODE="tensorboard"

########################################
# IMAGE / PREPROCESSING
########################################

# ⚠️ CRITIQUE pour débloquer la densification
CAMERA_RES_SCALE_FACTOR=0.9
MAX_RES=1280

NUM_DOWNSCALES=0
SKIP_IMAGE_PROCESSING=true

########################################
# TRAINING
########################################

# ⚠️ CRITIQUE (temps de densification)
MAX_ITER=9000

# 🛑 STOP SPLIT
#STOP_SPLIT_AT=$MAX_ITER

# ⚠️ CRITIQUE (qualité du gradient)
TRAIN_RAYS_PER_BATCH=1024

# 🧠 Meilleur signal pour split
NUM_NERF_SAMPLES_PER_RAY=48
NUM_PROPOSAL_SAMPLES_PER_RAY="128 128"

########################################
# GAUSSIAN SPLATTING (REDUCED SPLATS)
########################################

# 🔥 DENSIFICATION
DENSIFY_GRAD_THRESH=0.0004

# 🧹 CLEANING
CULL_ALPHA_THRESH=0.1        # ↑ supprime les splats faibles

# 📏 SPATIAL CONTROL (réduction explosion)
CULL_SCREEN_SIZE=0.25         # ↑ plus agressif en screen-space
SPLIT_SCREEN_SIZE=0.03

# ⚡ DENSIFICATION FREQUENCY (moins de croissance)
REFINE_EVERY=300              # ↑ réduit création de nouveaux splats


# 🧠 STABILISATION (évite accumulation de bruit)
RESET_ALPHA_EVERY=200
CULL_SCALE_THRESH=0.5         # ↓ supprime petits clusters instables

########################################
# QUALITY / REGULARIZATION
########################################


USE_BILATERAL_GRID=false
USE_SCALE_REGULARIZATION=true

MAX_GAUSS_RATIO=4.0          # ↓ limite taille splats
SSIM_LAMBDA=0.25             # léger boost stabilité image (optionnel)

########################################
# EXPORT BALANCED
########################################

# adapté au nouveau volume
EXPORT_NUM_POINTS=600000
EXPORT_DOWNSAMPLE=1
