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

# ⚠️ CRITIQUE pour débloquer la densification
CAMERA_RES_SCALE_FACTOR=0.75
MAX_RES=1280

NUM_DOWNSCALES=1
SKIP_IMAGE_PROCESSING=true
MAX_JOBS=2

########################################
# TRAINING
########################################

# ⚠️ CRITIQUE (temps de densification)
MAX_ITER=10000
# 🛑 STOP SPLIT PLUS TÔT
STOP_SPLIT_AT=6000


# ⚠️ CRITIQUE (qualité du gradient)
TRAIN_RAYS_PER_BATCH=512

# 🧠 Meilleur signal pour split
NUM_NERF_SAMPLES_PER_RAY=32
NUM_PROPOSAL_SAMPLES_PER_RAY="64 32"

########################################
# GAUSSIAN SPLATTING (REDUCED SPLATS)
########################################

# 🔥 DENSIFICATION (plus contrôlée sur long run)
DENSIFY_GRAD_THRESH=0.0005   # ↑ un peu plus strict pour éviter sur-split tardif

# 🧹 CLEANING (plus agressif en fin de training)
CULL_ALPHA_THRESH=0.14       # ↑ élimine plus tôt les splats instables

# 📏 SPATIAL CONTROL (anti-explosion bords)
CULL_SCREEN_SIZE=0.28        # ↑ réduit les micro-splats périphériques
SPLIT_SCREEN_SIZE=0.025      # ↑ évite sur-fragmentation fine

# ⚡ DENSIFICATION FREQUENCY (moins de croissance)
REFINE_EVERY=200              # ↑ réduit création de nouveaux splats


# 🧠 STABILISATION (évite accumulation de bruit)
RESET_ALPHA_EVERY=40          # ↑ nettoyage plus fréquent
CULL_SCALE_THRESH=0.5         # ↓ supprime petits clusters instables

########################################
# QUALITY / REGULARIZATION
########################################


USE_BILATERAL_GRID=true
USE_SCALE_REGULARIZATION=true

MAX_GAUSS_RATIO=4.0          # ↓ limite taille splats
SSIM_LAMBDA=0.25             # léger boost stabilité image (optionnel)

########################################
# EXPORT BALANCED
########################################

# adapté au nouveau volume
EXPORT_NUM_POINTS=600000
EXPORT_DOWNSAMPLE=1
