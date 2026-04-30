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


############################
# PIPELINE EXECUTION ENV
############################

ROOT_DIR="runs/default"

export SCENE_NAME="scene3d"



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
EXPERIMENT_NAME="model3d"

if [ -z "${MODEL+x}" ]; then
  MODEL="nerfacto"
fi



# Nerfstudio backend implementation
if [ -z "${MODEL_IMPLEMENTATION+x}" ]; then
    MODEL_IMPLEMENTATION="torch"
# torch  -> CPU / safe fallback
# tcnn   -> GPU ONLY (tiny-cuda-nn required)
fi

############################
# VIDEO PIPELINE
############################

VIDEO_NAME="video.mov"
if [ -z "${FPS+x}" ]; then
    FPS=10
fi

############################
# COLMAP / PREPROCESS
############################

SFMT_TOOL="colmap" # colmap | hloc | any
# → hloc = meilleur pour scènes difficiles / moins d’artefacts

MATCHING_METHOD="exhaustive"
# sequential | vocab_tree | exhaustive
# → exhaustive = plus précis mais lent
# → vocab_tree = bon compromis
# → sequential = vidéo uniquement

NUM_DOWNSCALES=1
# 0 | 1 | 2 | 3
# → + haut = moins de détails mais plus stable

SKIP_IMAGE_PROCESSING=true
# true | false
# → évite resize/copie images si déjà préparées

CAMERA_TYPE="perspective"
# perspective | pinhole | fisheye | equirectangular
# → mauvais choix = déformations COLMAP

############################
# FEATURE / MATCHING (COLMAP / HLOC)
############################

FEATURE_TYPE="any"
# sift | superpoint | superpoint_aachen | disk | r2d2 | any
# → superpoint/disk = meilleur pour scènes complexes
# disk pas compatible colmap

MATCHER_TYPE="any"
# NN | NN-mutual | superglue | superglue-fast | lightglue | disk+lightglue | any
# → superglue/lightglue = réduction artefacts + meilleurs matches -> incompatible colmap, utiliser hloc

############################
# CAMERA / STRUCTURE OPTIONS
############################

USE_SFM_DEPTH=false
# true | false
# → depth SfM utile pour densification mais plus lourd

REFINE_INTRINSICS=true
# true | false
# → améliore calibration caméra (bundle adjustment)

USE_SINGLE_CAMERA_MODE=true
# true | false
# → true si une seule caméra (sinon artefacts possibles)

############################
# CROPPING / FILTERING INPUT
############################

PERCENT_RADIUS_CROP=0.9
# 0.0 → 1.0
# → supprime bords image (réduit clusters parasites COLMAP)

CROP_FACTOR="0.02 0.05 0.02 0.02"
# top bottom left right (0-1)
# → crop manuel zones instables

############################
# IMAGE PREPROCESSING
############################

if [ -z "${CAMERA_RES_SCALE_FACTOR+x}" ]; then
    CAMERA_RES_SCALE_FACTOR=1.0
    #CAMERA_RES_SCALE_FACTOR=0.75  # 0.5 = FAST MODE (~4x speedup)
fi

############################
# TRAINING PARAMETERS (NERF CORE)
############################

# Nombre total d’itérations d’entraînement
# → 1 itération = optimisation sur un batch de rayons
# ↑ augmente la qualité mais augmente le temps de calcul
if [ -z "${MAX_ITER+x}" ]; then
  MAX_ITER=3000
fi

if [ -z "${MAX_JOBS+x}" ]; then
  MAX_JOBS=2
fi


# Nombre de rayons (pixels simulés) traités par batch
# → contrôle la stabilité et la mémoire utilisée
# ↑ plus grand = plus stable mais plus lent
if [ -z "${TRAIN_RAYS_PER_BATCH+x}" ]; then
    # TRAIN_RAYS_PER_BATCH=1024
    # TRAIN_RAYS_PER_BATCH=512
    TRAIN_RAYS_PER_BATCH=256
fi


############################
# SAMPLING (RECONSTRUCTION 3D)
############################

# Nombre de points échantillonnés par rayon caméra
# → chaque rayon est "découpé" en 3D pour estimer couleur + densité
# ↑ plus élevé = détails plus fins mais calcul plus lourd
if [ -z "${NUM_NERF_SAMPLES_PER_RAY+x}" ]; then
    NUM_NERF_SAMPLES_PER_RAY=32
fi

# Échantillonnage en 2 étapes (proposal network)
# 1er nombre : exploration grossière (zones importantes)
# 2e nombre : raffinement des zones sélectionnées
# → améliore qualité et efficacité du rendu
if [ -z "${NUM_PROPOSAL_SAMPLES_PER_RAY+x}" ]; then
    # NUM_PROPOSAL_SAMPLES_PER_RAY="160 64"
    NUM_PROPOSAL_SAMPLES_PER_RAY="64 32"
fi


############################
# GAUSSIAN SPLATTING (DENSIFICATION & PRUNING)
############################

# Seuil de gradient pour la densification des gaussiennes
# → contrôle quand de nouvelles gaussiennes sont ajoutées
# ↑ plus bas = plus de détails, mais plus de bruit et mémoire
if [ -z "${DENSIFY_GRAD_THRESH+x}" ]; then
  DENSIFY_GRAD_THRESH=0.0004
fi

# Seuil alpha pour supprimer les gaussiennes faibles
# → enlève les éléments peu visibles / inutiles
# ↑ plus haut = scène plus propre mais perte de détails fins
if [ -z "${CULL_ALPHA_THRESH+x}" ]; then
  CULL_ALPHA_THRESH=0.05
fi

# Taille écran pour culling (élimination des petites contributions)
# → supprime les splats trop petits à l’écran
# ↑ plus grand = plus agressif, moins de détails éloignés
if [ -z "${CULL_SCREEN_SIZE+x}" ]; then
  CULL_SCREEN_SIZE=0.3
fi

# Taille écran pour split (division des gaussiennes)
# → contrôle quand une gaussienne est divisée en plusieurs
# ↑ plus bas = plus de précision locale, mais plus de splats
if [ -z "${SPLIT_SCREEN_SIZE+x}" ]; then
  SPLIT_SCREEN_SIZE=0.02
fi
############################
# IMAGE / DATA RESOLUTION
############################

# Résolution maximale utilisée pendant l’entraînement
# → les images peuvent être downscalées automatiquement
# ↑ plus élevé = plus de détails mais plus lent et plus gourmand
if [ -z "${MAX_RES+x}" ]; then
    # MAX_RES=1024
    MAX_RES=512
fi

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
if [ -z "${TRAIN_VIS_MODE+x}" ]; then
    TRAIN_VIS_MODE="tensorboard"
    #TRAIN_VIS_MODE="viewer"
fi

############################
# EXPORT CONFIG
############################
# normales

NORMAL_METHOD="open3d"
REMOVE_OUTLIERS=True
