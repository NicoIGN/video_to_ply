########################################
# PERFORMANCE PROFILE - QUALITY PLUS
########################################

TRAINING_PROFILE="gpu/quality_plus"

DEVICE="gpu"
MODEL="splatfacto"
MODEL_IMPLEMENTATION="tcnn"
TRAIN_VIS_MODE="tensorboard"

########################################
# IMAGE / PREPROCESSING
########################################

# pleine résolution utile
CAMERA_RES_SCALE_FACTOR=1.0

# qualité élevée sans instabilité extrême
MAX_RES=2048

# conserve tous les détails source
NUM_DOWNSCALES=0

SKIP_IMAGE_PROCESSING=true
MAX_JOBS=2

########################################
# TRAINING
########################################

# convergence longue
MAX_ITER=22000

# critique pour stabilité à haute densité
TRAIN_RAYS_PER_BATCH=1024

# meilleur signal géométrique
NUM_NERF_SAMPLES_PER_RAY=64
NUM_PROPOSAL_SAMPLES_PER_RAY="128 64"

########################################
# GAUSSIAN SPLATTING
########################################

# densification fine et agressive
DENSIFY_GRAD_THRESH=0.0003

# garde davantage de petits détails
CULL_ALPHA_THRESH=0.07

# évite blobs géants
CULL_SCREEN_SIZE=0.16

# split plus fin
SPLIT_SCREEN_SIZE=0.016

########################################
# DENSIFICATION CONTROL
########################################

# densification soutenue mais stable
REFINE_EVERY=180

# laisse vivre la densification longtemps
STOP_SPLIT_AT=17000

# évite dérive alpha / saturation
RESET_ALPHA_EVERY=30

########################################
# REGULARIZATION / STABILITY
########################################

USE_BILATERAL_GRID=true
USE_SCALE_REGULARIZATION=true

# contrôle anisotropie
MAX_GAUSS_RATIO=4.0

# nettoie clusters dégénérés
CULL_SCALE_THRESH=0.45

# meilleur rendu perceptuel
SSIM_LAMBDA=0.30

########################################
# EXPORT QUALITY
########################################

EXPORT_NUM_POINTS=4000000
EXPORT_DOWNSAMPLE=1
