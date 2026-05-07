########################################
# PERFORMANCE PROFILE - QUALITY SAFE
########################################

TRAINING_PROFILE="gpu/quality"

DEVICE="gpu"
MODEL="splatfacto"
MODEL_IMPLEMENTATION="tcnn"
TRAIN_VIS_MODE="tensorboard"

########################################
# IMAGE / PREPROCESSING
########################################

# montée progressive seulement
CAMERA_RES_SCALE_FACTOR=1
MAX_RES=1280

# IMPORTANT :
# garde le downscale stabilisateur
NUM_DOWNSCALES=0

SKIP_IMAGE_PROCESSING=true
MAX_JOBS=2

########################################
# TRAINING
########################################

# plus long mais pas extrême
MAX_ITER=10000

# améliore gradients sans explosion
TRAIN_RAYS_PER_BATCH=768

# augmentation modérée
NUM_NERF_SAMPLES_PER_RAY=48
NUM_PROPOSAL_SAMPLES_PER_RAY="128 64"

########################################
# GAUSSIAN SPLATTING
########################################

# légèrement plus permissif
DENSIFY_GRAD_THRESH=0.00038

# garde davantage de détails
CULL_ALPHA_THRESH=0.10

# moins agressif mais stable
CULL_SCREEN_SIZE=0.2

# split un peu plus fin
SPLIT_SCREEN_SIZE=0.015

########################################
# DENSIFICATION CONTROL
########################################

# densification un peu plus active
REFINE_EVERY=200

# laisse vivre les splits plus longtemps
STOP_SPLIT_AT=10000

# stabilisation
RESET_ALPHA_EVERY=40

# nettoyage modéré
CULL_SCALE_THRESH=0.48

########################################
# QUALITY / REGULARIZATION
########################################

USE_BILATERAL_GRID=true
USE_SCALE_REGULARIZATION=true

MAX_GAUSS_RATIO=10.0

SSIM_LAMBDA=0.25

########################################
# EXPORT QUALITY
########################################

EXPORT_NUM_POINTS=1500000
EXPORT_DOWNSAMPLE=1
