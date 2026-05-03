
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

PERCENT_RADIUS_CROP=0.95
# 0.0 → 1.0
# → supprime bords image (réduit clusters parasites COLMAP)

CROP_FACTOR="0.02 0.05 0.02 0.02"
# top bottom left right (0-1)
# → crop manuel zones instables
