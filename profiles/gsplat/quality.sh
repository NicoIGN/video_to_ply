########################################
# PERFORMANCE PROFILE
########################################

TRAINING_PROFILE="gpu/high-quality"

DEVICE="gpu"
MODEL="splatfacto"
MODEL_IMPLEMENTATION="tcnn"
TRAIN_VIS_MODE="tensorboard"

########################################
# IMAGE / PREPROCESSING
########################################

# ton vrai sweet spot qualité/stabilité
CAMERA_RES_SCALE_FACTOR=0.75
MAX_RES=1152

NUM_DOWNSCALES=1
SKIP_IMAGE_PROCESSING=true
MAX_JOBS=2

########################################
# TRAINING
########################################

MAX_ITER=15000

# IMPORTANT :
# ne pas override le comportement naturel
# du scheduler Splatfacto
# STOP_SPLIT_AT supprimé volontairement

TRAIN_RAYS_PER_BATCH=512

# reste sur ton meilleur équilibre actuel
NUM_NERF_SAMPLES_PER_RAY=32
NUM_PROPOSAL_SAMPLES_PER_RAY="64 32"

########################################
# GAUSSIAN SPLATTING
########################################

# légèrement conservateur pour éviter
# l'explosion tardive des splats
DENSIFY_GRAD_THRESH=0.00048

# nettoyage modéré
# surtout ne pas trop monter
CULL_ALPHA_THRESH=0.11

# garde-fou contre les bavures bords
CULL_SCREEN_SIZE=0.26

# très important :
# 0.02 semble être ton vrai sweet spot
SPLIT_SCREEN_SIZE=0.02

# stabilité générale
RESET_ALPHA_EVERY=40
CULL_SCALE_THRESH=0.5

########################################
# QUALITY / REGULARIZATION
########################################

USE_BILATERAL_GRID=true
USE_SCALE_REGULARIZATION=true

# évite les splats géants / étirés
MAX_GAUSS_RATIO=4.0

# meilleur compromis observé chez toi
SSIM_LAMBDA=0.25

########################################
# EXPORT
########################################

EXPORT_NUM_POINTS=800000
EXPORT_DOWNSAMPLE=1
