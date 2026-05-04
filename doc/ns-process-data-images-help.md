# 🧠 Guide rapide — ns-process-data images (Nerfstudio)

Ce document résume les principaux paramètres disponibles pour `ns-process-data images`.

---

## 📁 Dataset I/O

### --data PATH (obligatoire)
Chemin vers les données d’entrée.

- Vidéo ou dossier d’images
- Exemple : `/content/video.mp4`

---

### --output-dir PATH (obligatoire)
Dossier de sortie Nerfstudio.

Contient :
- images processées
- poses COLMAP / HLoc
- transforms.json

---

### --eval-data PATH | None
Données d’évaluation.

- `None` → split automatique train/test
- sinon dossier/vidéo séparé

---

## 📷 Caméra

### --camera-type
{perspective, fisheye, equirectangular, pinhole, simple_pinhole}

- perspective → standard (smartphone / vidéo)
- fisheye → ultra grand angle
- equirectangular → 360°
- pinhole → modèle simple
- simple_pinhole → version simplifiée

---

## 🔀 Matching & SfM

### --matching-method
{exhaustive, sequential, vocab_tree}

- vocab_tree → recommandé (bon compromis)
- exhaustive → très précis mais lent
- sequential → vidéos uniquement

---

### --sfm-tool
{any, colmap, hloc}

- any → auto (recommandé)
- colmap → SIFT classique
- hloc → features modernes (SuperPoint / SuperGlue)

---

## 🧠 Features & Matching

### --feature-type
Types de features :

- sift
- superpoint
- superpoint_aachen / max / inloc
- r2d2, d2net-ss, sosnet, disk

👉 surtout utilisé avec HLoc

---

### --matcher-type
Algorithmes de matching :

- superglue (standard HLoc)
- superpoint+lightglue (rapide + moderne)
- disk+lightglue
- NN / ratio / mutual (classiques)

---

## 📉 Downscaling

### --num-downscales INT
Réduction de résolution.

Chaque niveau divise par 2 :

- 1 → original
- 2 → /2, /4
- 3 → /2, /4, /8

---

## ✂️ Crop

### --crop-factor FLOAT FLOAT FLOAT FLOAT
Crop global :

format :
