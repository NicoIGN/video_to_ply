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

# Stable full-res training
CAMERA_RES_SCALE_FACTOR=1.0

# Garde une résolution raisonnable
MAX_RES=1280

NUM_DOWNSCALES=1
SKIP_IMAGE_PROCESSING=true

########################################
# TRAINING
########################################

# Stable long training
MAX_ITER=15000

# IMPORTANT :
# évite la densification tardive explosive
STOP_SPLIT_AT=9000

# Stable gradients
TRAIN_RAYS_PER_BATCH=1024

# Bon compromis qualité/stabilité
NUM_NERF_SAMPLES_PER_RAY=48
NUM_PROPOSAL_SAMPLES_PER_RAY="128 128"

########################################
# GAUSSIAN SPLATTING
########################################

# Densification plus conservative
DENSIFY_GRAD_THRESH=0.00045

########################################
# CLEANING
########################################

# Nettoyage alpha un peu plus agressif
CULL_ALPHA_THRESH=0.12

# Évite gros splats écran
CULL_SCREEN_SIZE=0.25
SPLIT_SCREEN_SIZE=0.02

########################################
# DENSIFICATION CONTROL
########################################

# Beaucoup plus stable à long terme
REFINE_EVERY=300

# PARAMÈTRE CRITIQUE
# évite saturation alpha / écran blanc
RESET_ALPHA_EVERY=40

# Supprime davantage de gros splats instables
CULL_SCALE_THRESH=0.5

########################################
# QUALITY / REGULARIZATION
########################################

# Très utile pour stabilité visuelle
USE_BILATERAL_GRID=true

USE_SCALE_REGULARIZATION=true

# Évite covariance géante
MAX_GAUSS_RATIO=4.0

# Stable sans trop lisser
SSIM_LAMBDA=0.2

########################################
# EXPORT
########################################

EXPORT_NUM_POINTS=600000
EXPORT_DOWNSAMPLE=1
