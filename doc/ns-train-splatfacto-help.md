# Pipeline Model Configuration (Gaussian / NeRF Training)

Ce document regroupe les principaux paramètres du `pipeline.model` ainsi que les optimizers associés.

---

## 🔧 Scene & Rendering Setup

### --pipeline.model.collider (bool)
Active la création d’un collider de scène pour filtrer les rayons.

- **Type :** bool
- **Valeur par défaut :** True

---

### --pipeline.model.collider-params

Paramètres utilisés pour initialiser le collider de scène.

- **Type :** None | [STR FLOAT ...]
- **Valeur par défaut :** near_plane 2.0 far_plane 6.0

---

### --pipeline.model.eval-num-rays-per-chunk INT

Nombre de rayons traités par chunk lors de l’évaluation.

- **Valeur par défaut :** 4096

---

### --pipeline.model.prompt STR | None

Prompt texte utilisé pour les modèles text-to-NeRF.

- **Valeur par défaut :** None

---

### --pipeline.model.background-color {random,black,white}

Définit la couleur de fond.

- **random :** couleur aléatoire
- **black / white :** fond fixe
- **Valeur par défaut :** random

---

## ⏱ Training Schedule

### --pipeline.model.warmup-length INT

Nombre d’itérations où la refinement est désactivée.

- **Valeur par défaut :** 500

---

### --pipeline.model.refine-every INT

Fréquence de densification / culling des gaussiens.

- **Valeur par défaut :** 100

---

### --pipeline.model.stop-split-at INT

Stoppe le split des gaussiens après cette itération.

- **Valeur par défaut :** 15000

---

### --pipeline.model.stop-split-at INT

Stoppe l’opération de split des gaussiens après un certain nombre d’itérations.

- **Type :** INT  
- **Description :** définit à partir de quelle étape l’algorithme arrête de diviser les gaussiens (split).
- **Comportement :** passé ce seuil, seules les autres opérations (comme le culling ou l’optimisation) continuent.
- **Valeur par défaut :** 15000  

---

### --pipeline.model.resolution-schedule INT

Contrôle la montée progressive de la résolution.

- Double la résolution toutes les N étapes
- **Valeur par défaut :** 3000

---

## 🎯 Gaussian Pruning & Densification

### --pipeline.model.cull-alpha-thresh FLOAT

Seuil d’opacité pour supprimer les gaussiens.

- **Valeur par défaut :** 0.1  
- **Conseil :** 0.005 pour meilleure qualité

---

### --pipeline.model.cull-scale-thresh FLOAT

Supprime les gaussiens trop grands.

- **Valeur par défaut :** 0.5

---

### --pipeline.model.cull-screen-size FLOAT

Supprime les gaussiens trop grands à l’écran.

- **Valeur par défaut :** 0.15

---

### --pipeline.model.split-screen-size FLOAT

Split des gaussiens trop grands à l’écran.

- **Valeur par défaut :** 0.05

---

### --pipeline.model.densify-grad-thresh FLOAT

Seuil de gradient pour densification.

- **Valeur par défaut :** 0.0008

---

### --pipeline.model.densify-size-thresh FLOAT

Seuil de taille pour split/duplication.

- **Valeur par défaut :** 0.01

---

### --pipeline.model.n-split-samples INT

Nombre de sous-échantillons lors d’un split.

- **Valeur par défaut :** 2

---

### --pipeline.model.reset-alpha-every INT

Réinitialise l’alpha périodiquement.

- **Valeur par défaut :** 30

---

## 🧠 SH (Spherical Harmonics)

### --pipeline.model.sh-degree INT

Degré maximal des harmoniques sphériques.

- **Valeur par défaut :** 3

---

### --pipeline.model.sh-degree-interval INT

Augmente progressivement le degré SH.

- **Valeur par défaut :** 1000

---

## 🎲 Initialization

### --pipeline.model.random-init bool

Initialisation aléatoire des gaussiens.

- **Valeur par défaut :** False

---

### --pipeline.model.num-random INT

Nombre de gaussiens aléatoires.

- **Valeur par défaut :** 50000

---

### --pipeline.model.random-scale FLOAT

Taille du cube d’initialisation.

- **Valeur par défaut :** 10.0

---

## 🎨 Loss & Quality

### --pipeline.model.ssim-lambda FLOAT

Poids de la loss SSIM.

- **Valeur par défaut :** 0.2

---

### --pipeline.model.use-scale-regularization bool

Active la régularisation des scales (PhysGaussian).

- **Valeur par défaut :** False

---

### --pipeline.model.max-gauss-ratio FLOAT

Seuil de ratio max/min des gaussiens.

- **Valeur par défaut :** 10.0

---

### --pipeline.model.output-depth-during-training bool

Sortie des depth maps pendant le training.

- **Valeur par défaut :** False

---

## 🖼 Rendering Mode

### --pipeline.model.rasterize-mode {classic,antialiased}

Mode de rasterisation.

- **classic :** standard, rapide
- **antialiased :** meilleure qualité mais incompatible avec certains viewers
- **Valeur par défaut :** classic

---

## 🎛 Advanced Rendering

### --pipeline.model.use-bilateral-grid bool

Active correction ISP via bilateral grid.

- **Référence :** Bilateral Guided Radiance Field Processing
- **Valeur par défaut :** False

---

### --pipeline.model.grid-shape INT INT INT

Shape du grid bilateral (X, Y, W)

- **Valeur par défaut :** 16 16 8

---

### --pipeline.model.color-corrected-metrics bool

Applique correction couleur avant métriques.

- **Valeur par défaut :** False

---

## ⚙️ Camera Optimizer

### --pipeline.model.camera-optimizer.mode {off,SO3xR3,SE3}

Optimisation des poses caméra.

- **off :** désactivé
- **SO3xR3 :** recommandé
- **SE3 :** complet
- **Valeur par défaut :** off

---

### --pipeline.model.camera-optimizer.trans-l2-penalty FLOAT

Régularisation translation.

- **Valeur par défaut :** 0.01

---

### --pipeline.model.camera-optimizer.rot-l2-penalty FLOAT

Régularisation rotation.

- **Valeur par défaut :** 0.001

---

## 📊 Loss Coefficients

### --pipeline.model.loss-coefficients.rgb-loss-coarse FLOAT

Poids loss RGB coarse.

- **Valeur par défaut :** 1.0

---

### --pipeline.model.loss-coefficients.rgb-loss-fine FLOAT

Poids loss RGB fine.

- **Valeur par défaut :** 1.0

---

## 🚀 Optimizers (Global Structure)

### Means Optimizer
- lr: 0.00016

### Features DC
- lr: 0.0025

### Opacities
- lr: 0.05

### Scales
- lr: 0.005

### Quats
- lr: 0.001

### Camera Optimizer
- lr: 0.0001

### Bilateral Grid
- lr: 0.002

---