############################
# PIPELINE CONTROL (SKIP FLAGS)
############################

SKIP_CONDA_UPDATE=true
SKIP_FRAME_EXTRACTION=false
SKIP_COLMAP=false
SKIP_TRAINING=false
SKIP_EXPORT=false

NO_PROXY=true

############################
# PIPELINE EXECUTION ENV
############################

ROOT_DIR="runs/default"

INPUT_DIR=""
FRAME_DIR=""
ORI_DIR=""
OUTPUT_DIR=""
EXPORT_DIR=""


############################
# DEVICE MODE
############################

# If DEVICE is already defined in environment, keep it
if [ -z "${DEVICE+x}" ]; then
  DEVICE="cpu"   # cpu | gpu
fi

export DEVICE

# GPU REQUIREMENTS (important)
# - NVIDIA GPU + CUDA drivers
# - PyTorch compiled with CUDA
# - required for:
#   - splatfacto / splatfacto-w
#   - instant-ngp
#   - zip-nerf
#   - pynerf
#   - feature-splatting
#   - tetra-nerf (partially GPU)

# CPU MODE LIMITATIONS
# - no CUDA kernels
# - slower dataloading + training
# - recommended models:
#   - nerfacto (BEST CPU CHOICE)
#   - nerf
#   - kplanes (slow)
#   - tensorf (slow but works)


############################
# MODEL CONFIG
############################

if [ -z "${MODEL+x}" ]; then
  MODEL="nerfacto"
fi



# Nerfstudio backend implementation
MODEL_IMPLEMENTATION="torch"
# torch  -> CPU / safe fallback
# tcnn   -> GPU ONLY (tiny-cuda-nn required)


############################
# VIDEO PIPELINE
############################

VIDEO_NAME="video.mov"
FPS=10


############################
# COLMAP / PREPROCESS
############################

SFMT_TOOL="colmap"
MATCHING_METHOD="sequential"
NUM_DOWNSCALES=1
SKIP_IMAGE_PROCESSING=true

WITH_NERFSTUDIO="true"

############################
# IMAGE PREPROCESSING
############################

#CAMERA_RES_SCALE_FACTOR=1.0
CAMERA_RES_SCALE_FACTOR=0.25  # 0.5 = FAST MODE (~4x speedup)

############################
# TRAINING PARAMETERS (NERF CORE)
############################

# Nombre total d’itérations d’entraînement
# → 1 itération = optimisation sur un batch de rayons
# ↑ augmente la qualité mais augmente le temps de calcul
if [ -z "${MAX_ITER+x}" ]; then
  MAX_ITER=2000
fi

# Nombre de rayons (pixels simulés) traités par batch
# → contrôle la stabilité et la mémoire utilisée
# ↑ plus grand = plus stable mais plus lent
# TRAIN_RAYS_PER_BATCH=1024
TRAIN_RAYS_PER_BATCH=256


############################
# SAMPLING (RECONSTRUCTION 3D)
############################

# Nombre de points échantillonnés par rayon caméra
# → chaque rayon est "découpé" en 3D pour estimer couleur + densité
# ↑ plus élevé = détails plus fins mais calcul plus lourd
NUM_NERF_SAMPLES_PER_RAY=32

# Échantillonnage en 2 étapes (proposal network)
# 1er nombre : exploration grossière (zones importantes)
# 2e nombre : raffinement des zones sélectionnées
# → améliore qualité et efficacité du rendu
# NUM_PROPOSAL_SAMPLES_PER_RAY="160 64"
NUM_PROPOSAL_SAMPLES_PER_RAY="64 32"


############################
# IMAGE / DATA RESOLUTION
############################

# Résolution maximale utilisée pendant l’entraînement
# → les images peuvent être downscalées automatiquement
# ↑ plus élevé = plus de détails mais plus lent et plus gourmand
# MAX_RES=1024
MAX_RES=256


############################
# NERF ADVANCED SETTINGS
############################

CAMERA_MODE="off"



############################
# TRAINING MODE (VISUALISATION)
############################

# Contrôle l’interface de visualisation pendant le training
# options possibles :
#   viewer               → interface web interactive (par défaut)
#   tensorboard          → logs only (RECOMMANDÉ pour pipeline automatisé)
#   comet                → tracking expérimental
#   wandb                → tracking cloud
#   viewer+tensorboard   → hybride (debug + logs)
#   viewer+wandb         → hybride
#   viewer+comet        → hybride
#   viewer_beta         → version expérimentale viewer
#
# ⚠️ IMPORTANT :
# - "viewer" pour voir le resultat dans Nerfstudio
# - "tensorboard" pour mode headless réel (batch / scripts)
TRAIN_VIS_MODE="tensorboard"
#TRAIN_VIS_MODE="viewer"

if [ "$DEVICE" == "gpu" ]; then
    TRAIN_VIS_MODE="tensorboard"
fi

############################
# PERFORMANCE FLAGS
############################

OMP_NUM_THREADS=1
TORCHDYNAMO_DISABLE=1
PYTORCH_ENABLE_MPS_FALLBACK=1



############################
# EXPORT CONFIG
############################

NORMAL_METHOD="open3d"

# MODE GLOBAL D'EXPORT
# - fast     → très rapide, preview / debug
# - balanced → compromis qualité/vitesse (recommandé)
# - quality  → export complet haute qualité (lent)
EXPORT_MODE="balanced"

# PARAMÈTRES DÉRIVÉS (utilisés par export.sh)

# nombre de points exportés
EXPORT_NUM_POINTS_FAST=200000
EXPORT_NUM_POINTS_BALANCED=500000
EXPORT_NUM_POINTS_QUALITY=2000000

# normales
EXPORT_NORMALS_FAST="open3d"
EXPORT_NORMALS_BALANCED="open3d"
EXPORT_NORMALS_QUALITY="open3d"

# nettoyage
EXPORT_REMOVE_OUTLIERS_FAST=True
EXPORT_REMOVE_OUTLIERS_BALANCED=True
EXPORT_REMOVE_OUTLIERS_QUALITY=True

# downsampling global
EXPORT_DOWNSAMPLE_FAST=2
EXPORT_DOWNSAMPLE_BALANCED=1
EXPORT_DOWNSAMPLE_QUALITY=1


############################
# CONDA
############################

CONDA_ENV_FILE="environment/conda_colab.yml"
CONDA_ENV_NAME="gsplat"


############################
# PROXY
############################

if [ "$NO_PROXY" != true ]; then
  export HTTP_PROXY="http://proxy.ign.fr:3128"
  export HTTPS_PROXY="http://proxy.ign.fr:3128"
  export http_proxy="$HTTP_PROXY"
  export https_proxy="$HTTPS_PROXY"

  echo "🌐 Proxy enabled (3)"
else
  echo "🚫 Proxy disabled (NO_PROXY=true)"
  unset HTTP_PROXY
  unset HTTPS_PROXY
  unset http_proxy
  unset https_proxy
fi
