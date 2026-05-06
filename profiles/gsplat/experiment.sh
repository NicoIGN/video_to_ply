########################################
# PERFORMANCE PROFILE
########################################

TRAINING_PROFILE="gpu/experiment"

DEVICE="gpu"
MODEL="splatfacto"
MODEL_IMPLEMENTATION="tcnn"
TRAIN_VIS_MODE="tensorboard"

########################################
# IMAGE / PREPROCESSING
########################################

# ⚠️ CRITIQUE pour débloquer la densification
CAMERA_RES_SCALE_FACTOR=0.75
MAX_RES=1024

NUM_DOWNSCALES=1
SKIP_IMAGE_PROCESSING=true
MAX_JOBS=2

########################################
# TRAINING
########################################

# ⚠️ CRITIQUE (temps de densification)
MAX_ITER=12000

# ⚠️ CRITIQUE (qualité du gradient)
TRAIN_RAYS_PER_BATCH=512

# 🧠 Meilleur signal pour split
NUM_NERF_SAMPLES_PER_RAY=32
NUM_PROPOSAL_SAMPLES_PER_RAY="64 32"

########################################
# GAUSSIAN SPLATTING
########################################

# 🔥 DENSIFICATION (plus agressif mais contrôlé)
DENSIFY_GRAD_THRESH=0.0003

# 🧹 Nettoyage (équilibre densification)
CULL_ALPHA_THRESH=0.08

# 📏 Contrôle spatial
CULL_SCREEN_SIZE=0.2
SPLIT_SCREEN_SIZE=0.015

# ⚡ Fréquence de raffinement (clé)
REFINE_EVERY=200

# 🛑 Stop explosion tardive
STOP_SPLIT_AT=8000

# 🧠 Stabilisation
RESET_ALPHA_EVERY=50
CULL_SCALE_THRESH=0.6

########################################
# QUALITY / REGULARIZATION
########################################

USE_BILATERAL_GRID=true
USE_SCALE_REGULARIZATION=true
MAX_GAUSS_RATIO=5.0

# 🎨 Meilleure qualité perceptuelle
SSIM_LAMBDA=0.2

########################################
# EXPORT BALANCED
########################################

# adapté au nouveau volume
EXPORT_NUM_POINTS=600000
EXPORT_DOWNSAMPLE=1
