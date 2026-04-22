############################
# PIPELINE CONTROL (SKIP FLAGS)
############################
SKIP_CONDA_UPDATE=true
SKIP_FRAME_EXTRACTION=true
SKIP_COLMAP=false
SKIP_TRAINING=false
SKIP_EXPORT=false


############################
# PIPELINE EXECUTION ENV
############################

ROOT_DIR="runs/default"

# derived paths (will be set in run.sh)
INPUT_DIR=""
FRAME_DIR=""
ORI_DIR=""
OUTPUT_DIR=""
EXPORT_DIR=""


############################
# DEVICE MODE
############################

DEVICE="cpu"   # cpu | gpu


############################
# MODEL CONFIG
############################

MODEL="nerfacto"
MODEL_IMPLEMENTATION="torch"


############################
# VIDEO PIPELINE PARAMS
############################

VIDEO_NAME="video.mov"
FPS=10


############################
# DATA PREPROCESS (COLMAP)
############################

SFMT_TOOL="colmap"
MATCHING_METHOD="sequential"
SKIP_IMAGE_PROCESSING=true
NUM_DOWNSCALES=1


############################
# IMAGE PREPROCESSING
############################

CAMERA_RES_SCALE_FACTOR=1.0   # set 0.5 for FAST MODE (~4x speedup)


############################
# TRAINING PARAMETERS
############################

MAX_ITER=2000
RAYS=1024
TRAIN_RAYS_PER_BATCH=1024


############################
# MODEL QUALITY / SPEED TRADEOFF
############################

NUM_NERF_SAMPLES_PER_RAY=32
NUM_PROPOSAL_SAMPLES_PER_RAY="160 64"
MAX_RES=1024


############################
# NERF ADVANCED SETTINGS
############################

CAMERA_MODE="off"


############################
# EXPORT CONFIG
############################

NORMAL_METHOD="open3d"


############################
# PERFORMANCE FLAGS (CPU SAFE DEFAULTS)
############################

OMP_NUM_THREADS=1
TORCHDYNAMO_DISABLE=1
PYTORCH_ENABLE_MPS_FALLBACK=1


############################
# CONDA / ENVIRONMENT
############################

CONDA_ENV_FILE="environment/conda_macosx.yml"
CONDA_ENV_NAME="gsplat2"


############################
# PROXY (optional)
############################

HTTP_PROXY="http://proxy.ign.fr:3128"
HTTPS_PROXY="http://proxy.ign.fr:3128"
