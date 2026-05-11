# 🧠 Guide rapide — ns-process-data images (Nerfstudio)

Ce document résume les principaux paramètres disponibles pour `ns-process-data images`.

---

# 📁 Dataset I/O

## --data PATH (obligatoire)

Chemin vers les données d’entrée.

- vidéo
- dossier d’images

Exemple :

```bash
--data /content/video.mp4
```

---

## --output-dir PATH (obligatoire)

Dossier de sortie Nerfstudio.

Contient :
- images processées
- poses COLMAP / HLoc
- transforms.json

Exemple :

```bash
--output-dir /content/output
```

---

## --eval-data PATH | None

Données d’évaluation séparées.

- `None` → mêmes données utilisées pour train/eval
- sinon → dossier ou vidéo séparé

---

# 📷 Caméra

## --camera-type

```text
{perspective,fisheye,equirectangular,pinhole,simple_pinhole}
```

### Types

- `perspective`
  - standard smartphone / DSLR
  - recommandé dans la majorité des cas

- `fisheye`
  - ultra grand angle
  - GoPro / action cams

- `equirectangular`
  - caméras 360°

- `pinhole`
  - modèle caméra pinhole classique

- `simple_pinhole`
  - version simplifiée du modèle pinhole

---

# 🔀 Matching & SfM

## --matching-method

```text
{exhaustive,sequential,vocab_tree}
```

### Modes

- `vocab_tree`
  - recommandé
  - bon compromis vitesse/précision

- `exhaustive`
  - plus précis
  - beaucoup plus lent
  - coûteux sur gros datasets

- `sequential`
  - optimisé vidéo
  - rapide
  - suppose images temporellement proches

---

## --sfm-tool

```text
{any,colmap,hloc}
```

### Modes

- `any`
  - sélection automatique

- `colmap`
  - pipeline SIFT classique

- `hloc`
  - pipeline moderne
  - SuperPoint / SuperGlue / LightGlue

---

## --refine-pixsfm

Active Pixel Perfect SfM.

- améliore précision des poses
- plus lent
- seulement avec `hloc`

Très utile pour :
- Gaussian Splatting
- objets complexes
- scènes difficiles

---

## --refine-intrinsics

Active refinement des intrinsics caméra.

- bundle adjustment des paramètres caméra
- utile smartphone / vidéo
- seulement avec `colmap`

---

## --skip-colmap

Skip reconstruction COLMAP/HLoc si possible.

Utile pour :
- rerun rapide
- poses déjà calculées
- transforms.json déjà présent

---

## --colmap-model-path PATH

Chemin vers modèle COLMAP existant.

Utilisé avec :

```bash
--skip-colmap
```

Défaut :

```text
colmap/sparse/0
```

---

## --use-single-camera-mode

Suppose une seule calibration caméra.

- recommandé smartphone / vidéo
- réduit artefacts
- seulement avec `hloc`

Désactiver si :
- plusieurs caméras
- focales différentes
- dataset multi-device

---

# 🧠 Features & Matching

## --feature-type

```text
{any,sift,superpoint,superpoint_aachen,superpoint_max,superpoint_inloc,r2d2,d2net-ss,sosnet,disk}
```

### Features

- `sift`
  - classique COLMAP

- `superpoint`
  - features modernes

- `superpoint_aachen`
  - robuste
  - très bon compromis général

- `superpoint_max`
  - recall agressif

- `superpoint_inloc`
  - orienté indoor

- `r2d2`
  - robuste faible texture

- `d2net-ss`
  - dense local features

- `sosnet`
  - descripteur appris

- `disk`
  - excellent faible texture
  - parfois plus instable

---

## --matcher-type

```text
{any,NN,superglue,superglue-fast,NN-superpoint,NN-ratio,NN-mutual,adalam,disk+lightglue,superpoint+lightglue}
```

### Matchers

- `superglue`
  - précis
  - plus lent

- `superglue-fast`
  - version accélérée

- `superpoint+lightglue`
  - rapide
  - moderne
  - recommandé

- `disk+lightglue`
  - très performant faible texture

- `NN`
  - nearest neighbor simple

- `NN-ratio`
  - Lowe ratio test

- `NN-mutual`
  - matching mutuel

- `NN-superpoint`
  - matching spécifique SuperPoint

- `adalam`
  - matching robuste géométrique

---

# 📉 Downscaling

## --num-downscales INT

Nombre de niveaux de downscale générés.

Chaque niveau divise la résolution par 2.

### Exemples

- `0`
  - original uniquement

- `1`
  - + /2

- `2`
  - + /2 et /4

- `3`
  - + /2, /4 et /8

Défaut :

```text
3
```

---

## --same-dimensions

Suppose que toutes les images ont les mêmes dimensions.

Avantages :
- preprocessing plus rapide
- downscale optimisé
- moins de vérifications

Désactiver si :
- tailles différentes
- rotations EXIF variées
- dataset hétérogène

---

## --skip-image-processing

Skip :
- copie images
- resize/downscale

Lance seulement COLMAP/HLoc si possible.

Très utile pour :
- reruns rapides
- datasets déjà préparés
- pipelines cache

---

# ✂️ Crop

## --crop-factor FLOAT FLOAT FLOAT FLOAT

Crop global.

Format :

```text
top bottom left right
```

Toutes les valeurs doivent être dans `[0,1]`.

### Exemple

```bash
--crop-factor 0.02 0.05 0.02 0.02
```

=> crop :
- 2% haut
- 5% bas
- 2% gauche
- 2% droite

Très utile pour :
- rolling shutter
- bords flous
- watermark
- UI caméra
- aberrations optiques

---

## --crop-bottom FLOAT

Shortcut pour crop bas uniquement.

Equivalent à :

```bash
--crop-factor 0.0 X 0.0 0.0
```

---

## --percent-radius-crop FLOAT

Masque circulaire centré.

Le rayon correspond au pourcentage de la diagonale image.

### Exemple

```bash
--percent-radius-crop 0.95
```

Très utile pour :
- coins flous
- aberrations optiques
- bords parasites
- vidéos smartphone

Défaut :

```text
1.0
```

---

# 🧠 SfM Depth

## --use-sfm-depth

Exporte depth maps depuis les points SfM.

- plus lourd
- peut améliorer certaines pipelines
- utile pour supervision depth

---

## --include-depth-debug

Exporte images debug depth.

Seulement si :

```bash
--use-sfm-depth
```

est activé.

---

# ⚙️ GPU / CPU

## --gpu / --no-gpu

Force utilisation GPU ou CPU.

GPU fortement recommandé pour :
- HLoc
- SuperGlue
- LightGlue
- gros datasets

---

# 🔇 Logging

## --verbose

Active logs détaillés.

Défaut :

```text
False
```

---

# 🌍 Equirectangular

## --images-per-equirect

```text
{8,14}
```

Nombre d’images extraites depuis une image equirectangulaire.

Utilisé uniquement avec :

```bash
--camera-type equirectangular
```

- `8`
  - plus rapide

- `14`
  - meilleure couverture

---

# 🛠️ COLMAP

## --colmap-cmd STR

Commande utilisée pour lancer COLMAP.

Défaut :

```text
colmap
```

Utile si :
- binaire custom
- chemin spécifique
- environnement particulier

---

# ❓ Help

## -h / --help

Affiche l’aide complète CLI.
