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

# légère hausse qualité sans explosion mémoire
CAMERA_RES_SCALE_FACTOR=1
MAX_RES=1280

NUM_DOWNSCALES=1
SKIP_IMAGE_PROCESSING=true
MAX_JOBS=2

########################################
# TRAINING
########################################

# un peu plus long pour converger proprement
MAX_ITER=10000

# gradients plus stables
TRAIN_RAYS_PER_BATCH=768

# meilleur signal géométrique
NUM_NERF_SAMPLES_PER_RAY=48
NUM_PROPOSAL_SAMPLES_PER_RAY="96 48"

########################################
# GAUSSIAN SPLATTING
########################################

# densification légèrement plus sensible
# sans redevenir agressive
DENSIFY_GRAD_THRESH=0.0004

# garde plus de micro-structure utile
# sans laisser trop de poussière
CULL_ALPHA_THRESH=0.10

# bon équilibre nettoyage/stabilité
CULL_SCREEN_SIZE=0.22

# réduit le sur-splitting fin
SPLIT_SCREEN_SIZE=0.03

# raffinement modéré
REFINE_EVERY=250

# on arrête le split avant la fin
# pour stabiliser la géométrie
STOP_SPLIT_AT=10000

# moins agressif que 40
# laisse converger les opacités
RESET_ALPHA_EVERY=60

# garde le nettoyage des clusters instables
CULL_SCALE_THRESH=0.5

########################################
# QUALITY / REGULARIZATION
########################################

USE_BILATERAL_GRID=true
USE_SCALE_REGULARIZATION=true

# un peu moins contraint
# aide les détails fins
MAX_GAUSS_RATIO=5.0

# bon compromis détail/stabilité
SSIM_LAMBDA=0.22

########################################
# EXPORT
########################################

EXPORT_NUM_POINTS=1000000
EXPORT_DOWNSAMPLE=1
