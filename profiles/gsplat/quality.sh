########################################
# PERFORMANCE PROFILE - QUALITY
########################################

TRAINING_PROFILE="gpu/quality"

DEVICE="gpu"
MODEL="splatfacto"
MODEL_IMPLEMENTATION="tcnn"
TRAIN_VIS_MODE="tensorboard"

########################################
# IMAGE / PREPROCESSING
########################################

# vraie résolution de travail
CAMERA_RES_SCALE_FACTOR=1.0

# bon compromis qualité / stabilité
MAX_RES=1536

# conserve tous les détails
NUM_DOWNSCALES=0

SKIP_IMAGE_PROCESSING=true
MAX_JOBS=2

########################################
# TRAINING
########################################

# convergence correcte
MAX_ITER=16000

# très important pour stabilité haute qualité
TRAIN_RAYS_PER_BATCH=1024

# meilleur signal géométrique
NUM_NERF_SAMPLES_PER_RAY=64
NUM_PROPOSAL_SAMPLES_PER_RAY="128 64"

########################################
# GAUSSIAN SPLATTING
########################################

# plus de détails / plus de splats
DENSIFY_GRAD_THRESH=0.0004

# conserve détails fins sans explosion
CULL_ALPHA_THRESH=0.08

# évite gros blobs visibles
CULL_SCREEN_SIZE=0.18

# split plus fin pour géométrie détaillée
SPLIT_SCREEN_SIZE=0.018

########################################
# DENSIFICATION CONTROL
########################################

# ralentit légèrement la croissance
REFINE_EVERY=200

# laisse la scène densifier longtemps
STOP_SPLIT_AT=12000

# stabilise les alphas
RESET_ALPHA_EVERY=35

########################################
# REGULARIZATION / STABILITY
########################################

USE_BILATERAL_GRID=true
USE_SCALE_REGULARIZATION=true

# évite splats géants dégénérés
MAX_GAUSS_RATIO=4.5

# nettoie petits clusters instables
CULL_SCALE_THRESH=0.45

# meilleur rendu perceptuel
SSIM_LAMBDA=0.28

########################################
# EXPORT QUALITY
########################################

EXPORT_NUM_POINTS=2500000
EXPORT_DOWNSAMPLE=1
