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
CAMERA_RES_SCALE_FACTOR=0.9
MAX_RES=1280

# IMPORTANT :
# garde le downscale stabilisateur
NUM_DOWNSCALES=1

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
NUM_PROPOSAL_SAMPLES_PER_RAY="96 48"

########################################
# GAUSSIAN SPLATTING
########################################

# légèrement plus permissif
DENSIFY_GRAD_THRESH=0.00038

# garde davantage de détails
CULL_ALPHA_THRESH=0.10

# moins agressif mais stable
CULL_SCREEN_SIZE=0.22

# split un peu plus fin
SPLIT_SCREEN_SIZE=0.018

########################################
# DENSIFICATION CONTROL
########################################

# densification un peu plus active
REFINE_EVERY=250

# laisse vivre les splits plus longtemps
STOP_SPLIT_AT=8500

# stabilisation
RESET_ALPHA_EVERY=40

# nettoyage modéré
CULL_SCALE_THRESH=0.48

########################################
# QUALITY / REGULARIZATION
########################################

USE_BILATERAL_GRID=true
USE_SCALE_REGULARIZATION=true

MAX_GAUSS_RATIO=4.0

SSIM_LAMBDA=0.26

########################################
# EXPORT QUALITY
########################################

EXPORT_NUM_POINTS=1500000
EXPORT_DOWNSAMPLE=1
