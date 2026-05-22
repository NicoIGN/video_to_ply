# Dataparser Configuration (`pipeline.datamanager.dataparser`)

Ce document regroupe les principaux paramètres du dataparser Nerfstudio utilisés pour charger, normaliser et préparer les données avant le training.

---

## 📁 Dataset Input

### --data PATH

Chemin vers le dataset ou vers un fichier `transforms.json`.

- **Type :** PATH
- **Valeur par défaut :** `.`

---

## 📏 Scene Scaling

### --scale-factor FLOAT

Facteur de scale appliqué aux positions des caméras.

- **Type :** FLOAT
- **Valeur par défaut :** `1.0`

---

### --scene-scale FLOAT

Facteur de scale de la région d’intérêt de la scène.

- **Type :** FLOAT
- **Valeur par défaut :** `1.0`

---

### --auto-scale-poses {True,False}

Normalise automatiquement les poses caméra dans une bounding box `[-1, 1]`.

- **Type :** bool
- **Valeur par défaut :** `True`

---

## 🖼 Image Resolution & Downscaling

### --downscale-factor INT | None

Facteur de réduction des images avant chargement.

- **Type :** INT | None
- **Valeur par défaut :** `None`

#### Comportement

Si `None`, Nerfstudio choisit automatiquement un facteur de downscale afin que :

- la plus grande dimension soit inférieure à ~1600 px

#### Exemples

| Valeur | Résultat |
|---|---|
| `1` | Full resolution |
| `2` | Largeur/hauteur ÷2 |
| `4` | Largeur/hauteur ÷4 |

#### Notes

Ce paramètre :
- agit avant chargement GPU
- réduit VRAM, IO disque et cache mémoire
- est différent de `camera-res-scale-factor`

---

## 🧭 Camera Orientation & Centering

### --orientation-method {pca,up,vertical,none}

Méthode utilisée pour orienter automatiquement la scène.

- **pca :** orientation basée PCA
- **up :** utilise l’axe vertical moyen
- **vertical :** tente de réaligner verticalement
- **none :** désactive la réorientation

- **Valeur par défaut :** `up`

---

### --center-method {poses,focus,none}

Méthode utilisée pour recentrer les poses caméra.

- **poses :** centre sur les positions caméra
- **focus :** centre sur le point de focus
- **none :** aucun recentrage

- **Valeur par défaut :** `poses`

---

## 🧪 Train / Eval Split

### --eval-mode {fraction,filename,interval,all}

Méthode de séparation train/eval.

- **fraction :** split par pourcentage
- **filename :** split basé sur les noms de fichiers
- **interval :** prend une image sur N pour eval
- **all :** utilise toutes les images partout

- **Valeur par défaut :** `fraction`

---

### --train-split-fraction FLOAT

Pourcentage des images utilisées pour le training.

- **Type :** FLOAT
- **Valeur par défaut :** `0.9`

---

### --eval-interval INT

Intervalle entre images d’évaluation en mode `interval`.

- **Type :** INT
- **Valeur par défaut :** `8`

---

## 📐 Depth & Masks

### --depth-unit-scale-factor FLOAT

Facteur de conversion des depth maps vers les mètres.

- **Type :** FLOAT
- **Valeur par défaut :** `0.001`

#### Exemple

- `0.001` :
  conversion millimètres → mètres

---

### --mask-color FLOAT FLOAT FLOAT | None

Couleur utilisée pour remplacer les pixels masqués/inconnus.

- **Type :** RGB FLOAT triplet | None
- **Valeur par défaut :** `None`

---

## ☁️ 3D Reconstruction

### --load-3D-points {True,False}

Charge les points 3D issus de COLMAP.

- **Type :** bool
- **Valeur par défaut :** `True`

#### Notes

Ces points servent généralement :
- à initialiser les gaussiens
- à améliorer le démarrage du training
- à stabiliser la reconstruction

---
