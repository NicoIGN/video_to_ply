########################################
# PERFORMANCE PROFILE - QUALITY PRO
########################################

TRAINING_PROFILE="gpu/quality_plus"

DEVICE="gpu"
MODEL="splatfacto"
MODEL_IMPLEMENTATION="tcnn"
TRAIN_VIS_MODE="tensorboard"

########################################
# IMAGE / PREPROCESSING
########################################

CAMERA_RES_SCALE_FACTOR=1.0
MAX_RES=2048

NUM_DOWNSCALES=0
SKIP_IMAGE_PROCESSING=true
MAX_JOBS=2

########################################
# TRAINING
########################################

MAX_ITER=16000

# ⚡ plus fréquent au début → meilleur détail
REFINE_EVERY=150

TRAIN_RAYS_PER_BATCH=512

NUM_NERF_SAMPLES_PER_RAY=64
NUM_PROPOSAL_SAMPLES_PER_RAY="128 64"

########################################
# GAUSSIAN SPLATTING (DETAIL FIRST)
########################################

# 🔥 clé : ton 0.002 est trop haut
DENSIFY_GRAD_THRESH=0.0009

# garde structures fines plus longtemps
CULL_ALPHA_THRESH=0.07

# évite blobs mais sans tuer détails
CULL_SCREEN_SIZE=0.20

# split plus fin (important pour 2K)
SPLIT_SCREEN_SIZE=0.018

# laisse vivre la géométrie plus longtemps
STOP_SPLIT_AT=14000

########################################
# STABILITY CONTROL (SMART)
########################################

# réduit dérive sans bloquer
RESET_ALPHA_EVERY=40

# moins agressif que ton 0.5
CULL_SCALE_THRESH=0.45

# IMPORTANT : ton double MAX_GAUSS_RATIO était inutile
MAX_GAUSS_RATIO=4.0

########################################
# QUALITY / REGULARIZATION
########################################

USE_SCALE_REGULARIZATION=true

# un peu plus fort → améliore sharpness perçue
SSIM_LAMBDA=0.3

########################################
# EXPORT QUALITY
########################################

EXPORT_NUM_POINTS=3000000
EXPORT_DOWNSAMPLE=1
