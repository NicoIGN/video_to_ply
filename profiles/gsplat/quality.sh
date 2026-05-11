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

# garde le sweet spot trouvé
CAMERA_RES_SCALE_FACTOR=0.75
MAX_RES=1152

NUM_DOWNSCALES=1
SKIP_IMAGE_PROCESSING=true
MAX_JOBS=2

########################################
# TRAINING
########################################

# longue phase d’optimisation
MAX_ITER=15000

# IMPORTANT :
# on arrête la croissance géométrique tôt
# puis on laisse optimiser couleurs/opacités/SH
#STOP_SPLIT_AT=5000

TRAIN_RAYS_PER_BATCH=512

# ton sweet spot actuel
NUM_NERF_SAMPLES_PER_RAY=32
NUM_PROPOSAL_SAMPLES_PER_RAY="64 32"

########################################
# GAUSSIAN SPLATTING
########################################

# garde le comportement stable du balanced
DENSIFY_GRAD_THRESH=0.00045

# légèrement moins agressif pour préserver les détails
CULL_ALPHA_THRESH=0.11

# évite les bavures périphériques
CULL_SCREEN_SIZE=0.25

# ton meilleur compromis actuel
SPLIT_SCREEN_SIZE=0.02

# stabilité
RESET_ALPHA_EVERY=40
CULL_SCALE_THRESH=0.5

########################################
# QUALITY / REGULARIZATION
########################################

USE_BILATERAL_GRID=true
USE_SCALE_REGULARIZATION=true

# évite les splats étirés
MAX_GAUSS_RATIO=4.0

# bon équilibre texture / stabilité
SSIM_LAMBDA=0.25

########################################
# EXPORT
########################################

EXPORT_NUM_POINTS=700000
EXPORT_DOWNSAMPLE=1
