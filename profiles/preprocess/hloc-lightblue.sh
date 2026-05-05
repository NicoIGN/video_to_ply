
############################
# COLMAP / PREPROCESS
############################

SFMT_TOOL="hloc" # colmap | hloc | any
# → hloc = meilleur pour scènes difficiles / moins d’artefacts

MATCHING_METHOD="vocab_tree"
# sequential | vocab_tree | exhaustive
# → exhaustive = plus précis mais lent
# → vocab_tree = bon compromis
# → sequential = vidéo uniquement

NUM_DOWNSCALES=2
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

FEATURE_TYPE="superpoint"
# sift | superpoint | superpoint_aachen | disk | r2d2 | any
# → sift = COLMAP pur (robuste mais limité sur scènes peu texturées)
# → superpoint / disk = meilleurs pour scènes difficiles (indoor, faible texture)
# → disk souvent plus performant sur surfaces pauvres (table, objets)
# ⚠️ superpoint/disk nécessitent HLOC (pas compatibles COLMAP pur)

MATCHER_TYPE="lightglue"
# NN | NN-mutual | superglue | superglue-fast | lightglue | disk+lightglue | any
# → NN / NN-mutual = matching classique (COLMAP pur)
# → superglue = très précis mais plus lent (HLOC)
# → lightglue = plus rapide et souvent plus robuste (recommandé)
# → disk+lightglue = combo très performant pour scènes peu texturées
# ⚠️ superglue / lightglue nécessitent HLOC (pas compatibles COLMAP pur)

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
