########################################
# TRAINING
########################################

MAX_ITER=9000
STOP_SPLIT_AT=7500

TRAIN_RAYS_PER_BATCH=512

NUM_NERF_SAMPLES_PER_RAY=32
NUM_PROPOSAL_SAMPLES_PER_RAY="64 32"

########################################
# GAUSSIAN SPLATTING (BALANCED QUALITY)
########################################

# densification légèrement plus permissif que ton 12k
DENSIFY_GRAD_THRESH=0.00042

# pruning moins agressif (important pour éviter perte de détail)
CULL_ALPHA_THRESH=0.10

# spatial control équilibré
CULL_SCREEN_SIZE=0.23
SPLIT_SCREEN_SIZE=0.018

# densification plus fréquente (meilleure reconstruction locale)
REFINE_EVERY=200

# stabilité sans sur-cleaning
RESET_ALPHA_EVERY=50
CULL_SCALE_THRESH=0.55

########################################
# QUALITY
########################################

MAX_GAUSS_RATIO=4.5
SSIM_LAMBDA=0.20

########################################
# IMAGE
########################################

CAMERA_RES_SCALE_FACTOR=0.75
MAX_RES=1280
