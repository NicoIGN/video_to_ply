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

# meilleur équilibre stabilité / détail
CAMERA_RES_SCALE_FACTOR=1.0
MAX_RES=1536

# conserve pleine résolution source
NUM_DOWNSCALES=0

SKIP_IMAGE_PROCESSING=true
MAX_JOBS=2

########################################
# TRAINING
########################################

# temps suffisant pour convergence propre
MAX_ITER=16000

# IMPORTANT pour stabilité en haute résolution
TRAIN_RAYS_PER_BATCH=1024

# meilleur signal géométrique
NUM_NERF_SAMPLES_PER_RAY=64
NUM_PROPOSAL_SAMPLES_PER_RAY="128 64"

########################################
# GAUSSIAN SPLATTING
########################################

# densification fine mais stable
DENSIFY_GRAD_THRESH=0.00055

# évite explosion de splats
REFINE_EVERY=250

# stop densification avant phase finale
STOP_SPLIT_AT=16000

########################################
# CLEANING / STABILITY
########################################

# nettoie les splats faibles avant dérive
CULL_ALPHA_THRESH=0.10

# évite gros blobs écran
CULL_SCREEN_SIZE=0.22

# split suffisamment fin pour détails
SPLIT_SCREEN_SIZE=0.018

# supprime clusters instables
CULL_SCALE_THRESH=0.45

# reset périodique pour éviter saturation alpha
RESET_ALPHA_EVERY=40

########################################
# REGULARIZATION / QUALITY
########################################

USE_BILATERAL_GRID=true
USE_SCALE_REGULARIZATION=true

# limite taille extrême des splats
MAX_GAUSS_RATIO=4.0

# améliore netteté perçue
SSIM_LAMBDA=0.30

########################################
# EXPORT
########################################

# largement suffisant pour qualité élevée
EXPORT_NUM_POINTS=1500000
EXPORT_DOWNSAMPLE=1
