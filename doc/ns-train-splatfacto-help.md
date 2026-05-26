# Nerfstudio `ns-train splatfacto` — Documentation des paramètres

Ce document présente les principaux paramètres de `ns-train splatfacto` par **thématiques fonctionnelles** :

- configuration générale de l’expérience
- machine et exécution
- logging et visualisation
- dataset et datamanager
- évaluation
- rendu et scène
- densification / pruning
- qualité / losses
- caméra
- optimizers

L’objectif est d’avoir une vue **plus pratique** que le `--help` brut, notamment pour le tuning mémoire / stabilité.

---

# 1. Configuration générale de l’expérience

## `--output-dir PATH`
Répertoire de sortie où seront stockés :
- checkpoints
- logs
- résultats d’entraînement

- **Valeur par défaut :** `outputs`

---

## `--method-name STR`
Nom de la méthode utilisée.

- **Valeur par défaut :** `splatfacto`

---

## `--experiment-name STR | None`
Nom de l’expérience.

- Si `None`, automatiquement dérivé du dataset.
- **Valeur par défaut :** `None`

---

## `--project-name STR | None`
Nom du projet.

- **Valeur par défaut :** `nerfstudio-project`

---

## `--timestamp STR`
Timestamp de l’expérience.

- **Valeur par défaut :** `{timestamp}`

---

## `--data PATH | None`
Alias de `--pipeline.datamanager.data`.

- **Valeur par défaut :** `None`

---

## `--prompt STR | None`
Alias de `--pipeline.model.prompt`.

- **Valeur par défaut :** `None`

---

## `--relative-model-dir PATH`
Sous-répertoire relatif pour stocker les checkpoints du modèle.

- **Valeur par défaut :** `nerfstudio_models`

---

## `--save-only-latest-checkpoint {True,False}`
Ne garde que le dernier checkpoint si `True`.

- **Valeur par défaut :** `True`

---

## `--load-scheduler {True,False}`
Charge l’état du scheduler si un checkpoint est restauré.

- **Valeur par défaut :** `True`

---

## `--load-dir PATH | None`
Répertoire modèle préentraîné à recharger.

- **Valeur par défaut :** `None`

---

## `--load-step INT | None`
Étape spécifique à recharger depuis `load-dir`.

- **Valeur par défaut :** `None`

---

## `--load-config PATH | None`
Chemin vers un fichier YAML de configuration à charger.

- **Valeur par défaut :** `None`

---

## `--load-checkpoint PATH | None`
Chemin vers un checkpoint spécifique.

- **Valeur par défaut :** `None`

---

# 2. Exécution et machine

## `--machine.seed INT`
Seed aléatoire.

- **Valeur par défaut :** `42`

---

## `--machine.num-devices INT`
Nombre total de devices disponibles.

- **Valeur par défaut :** `1`

---

## `--machine.num-machines INT`
Nombre total de machines pour entraînement distribué.

- **Valeur par défaut :** `1`

---

## `--machine.machine-rank INT`
Rang de la machine courante.

- **Valeur par défaut :** `0`

---

## `--machine.dist-url STR`
Point de connexion pour le distribué.

- **Valeur par défaut :** `auto`

---

## `--machine.device-type {cpu,cuda,mps}`
Type de device utilisé.

- **Valeur par défaut :** `cuda`

---

## `--max-num-iterations INT`
Nombre maximal d’itérations d’entraînement.

- **Valeur par défaut :** `30000`

---

## `--mixed-precision {True,False}`
Active l’entraînement en précision mixte.

- **Valeur par défaut :** `False`

**Remarque :**
- peut réduire l’usage mémoire GPU.

---

## `--use-grad-scaler {True,False}`
Active le gradient scaler, même si AMP est désactivé.

- **Valeur par défaut :** `False`

---

## `--gradient-accumulation-steps [STR INT ...]`
Accumulation de gradients par groupe de paramètres.

- **Valeur par défaut :** vide

**Remarque :**
- permet de simuler un batch plus grand sans tout charger d’un coup.

---

## `--start-paused {True,False}`
Démarre l’entraînement en pause.

- **Valeur par défaut :** `False`

---

# 3. Sauvegarde, logs et profiling

## `--steps-per-save INT`
Nombre d’itérations entre deux sauvegardes.

- **Valeur par défaut :** `2000`

---

## `--logging.relative-log-dir PATH`
Répertoire relatif des logs.

- **Valeur par défaut :** `.`

---

## `--logging.steps-per-log INT`
Fréquence de logging des statistiques.

- **Valeur par défaut :** `10`

---

## `--logging.max-buffer-size INT`
Taille max de l’historique utilisé pour les moyennes glissantes.

- **Valeur par défaut :** `20`

---

## `--logging.profiler {none,basic,pytorch}`
Choix du profiler.

- **Valeur par défaut :** `basic`

**Options :**
- `none`
- `basic`
- `pytorch`

---

## `--log-gradients {True,False}`
Active le logging des gradients.

- **Valeur par défaut :** `False`

---

## `--logging.local-writer.enable {True,False}`
Active l’affichage local des stats.

- **Valeur par défaut :** `True`

---

## `--logging.local-writer.stats-to-track [...]`
Liste des statistiques suivies localement.

- **Valeur par défaut :**
  - `ITER_TRAIN_TIME`
  - `TRAIN_RAYS_PER_SEC`
  - `CURR_TEST_PSNR`
  - `VIS_RAYS_PER_SEC`
  - `TEST_RAYS_PER_SEC`
  - `ETA`

---

## `--logging.local-writer.max-log-size INT`
Nombre maximal de lignes affichées avant wrapping.

- **Valeur par défaut :** `10`

---

# 4. Visualisation / Viewer

## `--vis {viewer,wandb,tensorboard,comet,viewer+wandb,viewer+tensorboard,viewer+comet,viewer_legacy}`
Choisit le backend de visualisation / logging.

- **Valeur par défaut :** `viewer`

**Options possibles :**
- `viewer`
- `wandb`
- `tensorboard`
- `comet`
- `viewer+wandb`
- `viewer+tensorboard`
- `viewer+comet`
- `viewer_legacy`

---

## `--viewer.relative-log-filename STR`
Nom du fichier log utilisé par le viewer.

- **Valeur par défaut :** `viewer_log_filename.txt`

---

## `--viewer.websocket-port INT | None`
Port websocket du viewer.

- **Valeur par défaut :** `None`

**Interprétation :**
- si `None`, un port libre est choisi automatiquement.

---

## `--viewer.websocket-port-default INT`
Port par défaut si aucun port n’est précisé.

- **Valeur par défaut :** `7007`

---

## `--viewer.websocket-host STR`
Adresse hôte du serveur websocket.

- **Valeur par défaut :** `0.0.0.0`

---

## `--viewer.num-rays-per-chunk INT`
Nombre de rayons rendus par chunk dans le viewer.

- **Valeur par défaut :** `32768`

---

## `--viewer.max-num-display-images INT`
Nombre maximal d’images affichées dans le viewer.

- **Valeur par défaut :** `512`

**Remarque :**
- n’affecte pas le training, seulement l’affichage.

---

## `--viewer.quit-on-train-completion {True,False}`
Quitte automatiquement le viewer à la fin du training.

- **Valeur par défaut :** `False`

---

## `--viewer.image-format {jpeg,png}`
Format d’image utilisé par le viewer.

- **Valeur par défaut :** `jpeg`

---

## `--viewer.jpeg-quality INT`
Qualité JPEG utilisée par le viewer.

- **Valeur par défaut :** `75`

---

## `--viewer.make-share-url {True,False}`
Active la génération d’une URL de partage pour le viewer.

- **Valeur par défaut :** `False`

---

## `--viewer.camera-frustum-scale FLOAT`
Échelle d’affichage des frustums caméra dans le viewer.

- **Valeur par défaut :** `0.1`

---

## `--viewer.default-composite-depth {True,False}`
Active le compositing depth par défaut dans le viewer.

- **Valeur par défaut :** `True`

---

# 5. Dataset, chargement et cache (Datamanager)

## `--pipeline.datamanager.data PATH | None`
Chemin vers les données d’entrée.

- **Valeur par défaut :** `None`

---

## `--pipeline.datamanager.masks-on-gpu {True,False}`
Traite les masques sur GPU.

- **Valeur par défaut :** `False`

**Remarque :**
- accélère certains cas
- augmente la consommation mémoire GPU

---

## `--pipeline.datamanager.images-on-gpu {True,False}`
Traite les images sur GPU.

- **Valeur par défaut :** `False`

**Attention :**
- si `True`, augmente la consommation mémoire GPU
- à éviter sur GPU déjà limite en VRAM

---

## `--pipeline.datamanager.cache-images {cpu,gpu}`
Définit où les images sont cachées en mémoire.

- `cpu` : cache sur CPU
- `gpu` : cache sur GPU
- **Valeur par défaut :** `gpu`

**Attention :**
- `gpu` est plus rapide
- `gpu` augmente fortement la pression VRAM
- sur un job instable en mémoire, `cpu` est souvent plus sûr

---

## `--pipeline.datamanager.cache-images-type {uint8,float32}`
Type utilisé pour le cache image.

- **Valeur par défaut :** `uint8`

**Interprétation :**
- `uint8` consomme moins de mémoire
- `float32` consomme plus

---

## `--pipeline.datamanager.max-thread-workers INT | None`
Nombre maximal de threads pour le cache image.

- **Valeur par défaut :** `None`

**Interprétation :**
- `None` = utilise tous les threads disponibles

---

## `--pipeline.datamanager.camera-res-scale-factor FLOAT`
Facteur d’échelle appliqué aux images et aux informations caméra associées.

- **Valeur par défaut :** `1.0`

**Exemples :**
- `1.0` = pleine résolution
- `0.75` = 75%
- `0.5` = moitié résolution

**Attention :**
- paramètre majeur pour la mémoire GPU

---

# 6. Évaluation

## `--steps-per-eval-batch INT`
Nombre d’itérations entre deux évaluations sur batchs aléatoires de rayons.

- **Valeur par défaut :** `0`

**Remarque :**
- peu coûteux comparé à l’évaluation image complète.

---

## `--steps-per-eval-image INT`
Nombre d’itérations entre deux évaluations sur une seule image.

- **Valeur par défaut :** `100`

**Remarque :**
- coût mémoire modéré.

---

## `--steps-per-eval-all-images INT`
Nombre d’itérations entre deux évaluations sur **toutes les images d’évaluation**.

- **Valeur par défaut :** `1000`

**Attention :**
- peut provoquer un **gros pic de mémoire GPU**
- particulièrement risqué avec LPIPS, cache GPU, haute résolution et scène densifiée

---

## `--pipeline.datamanager.eval-num-images-to-sample-from INT`
Nombre d’images candidates pour l’évaluation.

- **Valeur par défaut :** `-1`

**Interprétation :**
- `-1` signifie : utiliser toutes les images disponibles

---

## `--pipeline.datamanager.eval-num-times-to-repeat-images INT`
Nombre d’itérations avant de re-sélectionner de nouvelles images d’évaluation lorsque l’on n’évalue pas sur toutes.

- **Valeur par défaut :** `-1`

**Interprétation :**
- `-1` signifie : ne jamais changer d’images

---

## `--pipeline.datamanager.eval-image-indices {None}|{[INT [INT ...]]}`
Indices des images utilisées pour l’évaluation.

- **Valeur par défaut :** `0`

**Remarque :**
- permet de limiter l’évaluation à une ou plusieurs images précises

---

## `--pipeline.model.eval-num-rays-per-chunk INT`
Nombre de rayons traités par chunk lors de l’évaluation.

- **Valeur par défaut :** `4096`

**Conseil :**
- réduire cette valeur peut aider à limiter la mémoire pendant l’éval

---

# 7. Scène, rendu et géométrie globale

## `--pipeline.model.enable-collider {True,False}`
Active la création d’un collider de scène pour filtrer les rayons.

- **Valeur par défaut :** `True`

---

## `--pipeline.model.collider-params`
Paramètres utilisés pour initialiser le collider de scène.

- **Type :** `None | [STR FLOAT ...]`
- **Valeur par défaut :** `near_plane 2.0 far_plane 6.0`

---

## `--pipeline.model.background-color {random,black,white}`
Définit la couleur de fond.

- **random :** couleur aléatoire
- **black / white :** fond fixe
- **Valeur par défaut :** `random`

---

## `--pipeline.model.rasterize-mode {classic,antialiased}`
Mode de rasterisation.

- **classic :** standard, rapide
- **antialiased :** meilleure qualité mais incompatible avec certains viewers
- **Valeur par défaut :** `classic`

---

## `--pipeline.model.output-depth-during-training {True,False}`
Sortie des depth maps pendant le training.

- **Valeur par défaut :** `False`

---

# 8. Planning d’entraînement et raffinement

## `--pipeline.model.warmup-length INT`
Nombre d’itérations où la refinement est désactivée.

- **Valeur par défaut :** `500`

---

## `--pipeline.model.refine-every INT`
Fréquence de densification / culling des gaussiens.

- **Valeur par défaut :** `100`

---

## `--pipeline.model.resolution-schedule INT`
Contrôle la montée progressive de la résolution.

- Double la résolution toutes les N étapes.
- **Valeur par défaut :** `3000`

---

## `--pipeline.model.num-downscales INT`
Au début de l’entraînement, la résolution est divisée par `2^d`, où `d` est cette valeur.

- **Valeur par défaut :** `2`

---

## `--pipeline.model.stop-split-at INT`
Stoppe l’opération de split des gaussiens après un certain nombre d’itérations.

- **Valeur par défaut :** `15000`

---

## `--pipeline.model.stop-screen-size-at INT`
Arrête le culling/splitting basé sur la taille écran après cette étape.

- **Valeur par défaut :** `4000`

---

# 9. Densification, duplication et pruning des gaussiens

## `--pipeline.model.cull-alpha-thresh FLOAT`
Seuil d’opacité pour supprimer les gaussiens.

- **Valeur par défaut :** `0.1`

**Remarque :**
- une valeur plus faible conserve davantage de gaussiens
- une valeur plus élevée réduit la mémoire mais peut dégrader la qualité

---

## `--pipeline.model.cull-scale-thresh FLOAT`
Supprime les gaussiens trop grands.

- **Valeur par défaut :** `0.5`

---

## `--pipeline.model.cull-screen-size FLOAT`
Supprime les gaussiens trop grands à l’écran.

- **Valeur par défaut :** `0.15`

---

## `--pipeline.model.split-screen-size FLOAT`
Split des gaussiens trop grands à l’écran.

- **Valeur par défaut :** `0.05`

---

## `--pipeline.model.densify-grad-thresh FLOAT`
Seuil de gradient pour densification.

- **Valeur par défaut :** `0.0008`

**Remarque :**
- plus bas = densification plus agressive
- plus haut = moins de nouveaux gaussiens

---

## `--pipeline.model.use-absgrad {True,False}`
Utilise `absgrad` pour la densification.

- **Valeur par défaut :** `True`

---

## `--pipeline.model.densify-size-thresh FLOAT`
Seuil de taille pour split/duplication.

- **Valeur par défaut :** `0.01`

**Interprétation :**
- en-dessous : duplication
- au-dessus : split

---

## `--pipeline.model.n-split-samples INT`
Nombre de sous-échantillons lors d’un split.

- **Valeur par défaut :** `2`

---

## `--pipeline.model.reset-alpha-every INT`
Réinitialise l’alpha périodiquement.

- **Valeur par défaut :** `30`

---

# 10. Représentation visuelle et qualité

## `--pipeline.model.sh-degree INT`
Degré maximal des harmoniques sphériques.

- **Valeur par défaut :** `3`

---

## `--pipeline.model.sh-degree-interval INT`
Augmente progressivement le degré SH.

- **Valeur par défaut :** `1000`

---

## `--pipeline.model.ssim-lambda FLOAT`
Poids de la loss SSIM.

- **Valeur par défaut :** `0.2`

---

## `--pipeline.model.color-corrected-metrics {True,False}`
Applique une correction couleur avant le calcul des métriques.

- **Valeur par défaut :** `False`

---

# 11. Initialisation des gaussiens

## `--pipeline.model.random-init {True,False}`
Initialisation aléatoire des gaussiens.

- **Valeur par défaut :** `False`

---

## `--pipeline.model.num-random INT`
Nombre de gaussiens aléatoires.

- **Valeur par défaut :** `50000`

---

## `--pipeline.model.random-scale FLOAT`
Taille du cube d’initialisation.

- **Valeur par défaut :** `10.0`

---

# 12. Régularisation et contrôle de forme

## `--pipeline.model.use-scale-regularization {True,False}`
Active la régularisation des scales (PhysGaussian).

- **Valeur par défaut :** `False`

---

## `--pipeline.model.max-gauss-ratio FLOAT`
Seuil de ratio max/min des gaussiens.

- **Valeur par défaut :** `10.0`

---

# 13. Bilateral grid et correction ISP

## `--pipeline.model.use-bilateral-grid {True,False}`
Active la correction ISP via bilateral grid.

- **Valeur par défaut :** `False`

---

## `--pipeline.model.grid-shape INT INT INT`
Dimensions du bilateral grid `(X, Y, W)`.

- **Valeur par défaut :** `16 16 8`

---

# 14. Caméra et optimisation de pose

## `--pipeline.model.camera-optimizer.mode {off,SO3xR3,SE3}`
Optimisation des poses caméra.

- **off :** désactivé
- **SO3xR3 :** recommandé
- **SE3 :** complet
- **Valeur par défaut :** `off`

---

## `--pipeline.model.camera-optimizer.trans-l2-penalty FLOAT`
Régularisation translation.

- **Valeur par défaut :** `0.01`

---

## `--pipeline.model.camera-optimizer.rot-l2-penalty FLOAT`
Régularisation rotation.

- **Valeur par défaut :** `0.001`

---

# 15. Coefficients de loss

## `--pipeline.model.loss-coefficients.rgb-loss-coarse FLOAT`
Poids de la loss RGB coarse.

- **Valeur par défaut :** `1.0`

---

## `--pipeline.model.loss-coefficients.rgb-loss-fine FLOAT`
Poids de la loss RGB fine.

- **Valeur par défaut :** `1.0`

---

# 16. Optimizers

## Means Optimizer

### `--optimizers.means.optimizer.lr FLOAT`
- **Valeur par défaut :** `0.00016`

### `--optimizers.means.optimizer.eps FLOAT`
- **Valeur par défaut :** `1e-15`

### `--optimizers.means.optimizer.max-norm FLOAT | None`
- **Valeur par défaut :** `None`

### `--optimizers.means.optimizer.weight-decay FLOAT`
- **Valeur par défaut :** `0`

### `--optimizers.means.scheduler.lr-pre-warmup FLOAT`
- **Valeur par défaut :** `1e-08`

### `--optimizers.means.scheduler.lr-final FLOAT | None`
- **Valeur par défaut :** `1.6e-06`

### `--optimizers.means.scheduler.warmup-steps INT`
- **Valeur par défaut :** `0`

### `--optimizers.means.scheduler.max-steps INT`
- **Valeur par défaut :** `30000`

### `--optimizers.means.scheduler.ramp {linear,cosine}`
- **Valeur par défaut :** `cosine`

---

## Features DC Optimizer

### `--optimizers.features-dc.optimizer.lr FLOAT`
- **Valeur par défaut :** `0.0025`

### `--optimizers.features-dc.optimizer.eps FLOAT`
- **Valeur par défaut :** `1e-15`

### `--optimizers.features-dc.optimizer.max-norm FLOAT | None`
- **Valeur par défaut :** `None`

### `--optimizers.features-dc.optimizer.weight-decay FLOAT`
- **Valeur par défaut :** `0`

### `--optimizers.features-dc.scheduler {None}`
- **Valeur par défaut :** `None`

---

## Features Rest Optimizer

### `--optimizers.features-rest.optimizer.lr FLOAT`
- **Valeur par défaut :** `0.000125`

### `--optimizers.features-rest.optimizer.eps FLOAT`
- **Valeur par défaut :** `1e-15`

### `--optimizers.features-rest.optimizer.max-norm FLOAT | None`
- **Valeur par défaut :** `None`

### `--optimizers.features-rest.optimizer.weight-decay FLOAT`
- **Valeur par défaut :** `0`

### `--optimizers.features-rest.scheduler {None}`
- **Valeur par défaut :** `None`

---

## Opacities Optimizer

### `--optimizers.opacities.optimizer.lr FLOAT`
- **Valeur par défaut :** `0.05`

### `--optimizers.opacities.optimizer.eps FLOAT`
- **Valeur par défaut :** `1e-15`

### `--optimizers.opacities.optimizer.max-norm FLOAT | None`
- **Valeur par défaut :** `None`

### `--optimizers.opacities.optimizer.weight-decay FLOAT`
- **Valeur par défaut :** `0`

### `--optimizers.opacities.scheduler {None}`
- **Valeur par défaut :** `None`

---

## Scales Optimizer

### `--optimizers.scales.optimizer.lr FLOAT`
- **Valeur par défaut :** `0.005`

### `--optimizers.scales.optimizer.eps FLOAT`
- **Valeur par défaut :** `1e-15`

### `--optimizers.scales.optimizer.max-norm FLOAT | None`
- **Valeur par défaut :** `None`

### `--optimizers.scales.optimizer.weight-decay FLOAT`
- **Valeur par défaut :** `0`

### `--optimizers.scales.scheduler {None}`
- **Valeur par défaut :** `None`

---

## Quats Optimizer

### `--optimizers.quats.optimizer.lr FLOAT`
- **Valeur par défaut :** `0.001`

### `--optimizers.quats.optimizer.eps FLOAT`
- **Valeur par défaut :** `1e-15`

### `--optimizers.quats.optimizer.max-norm FLOAT | None`
- **Valeur par défaut :** `None`

### `--optimizers.quats.optimizer.weight-decay FLOAT`
- **Valeur par défaut :** `0`

### `--optimizers.quats.scheduler {None}`
- **Valeur par défaut :** `None`

---

## Camera Optimizer

### `--optimizers.camera-opt.optimizer.lr FLOAT`
- **Valeur par défaut :** `0.0001`

### `--optimizers.camera-opt.optimizer.eps FLOAT`
- **Valeur par défaut :** `1e-15`

### `--optimizers.camera-opt.optimizer.max-norm FLOAT | None`
- **Valeur par défaut :** `None`

### `--optimizers.camera-opt.optimizer.weight-decay FLOAT`
- **Valeur par défaut :** `0`

### `--optimizers.camera-opt.scheduler.lr-pre-warmup FLOAT`
- **Valeur par défaut :** `0`

### `--optimizers.camera-opt.scheduler.lr-final FLOAT | None`
- **Valeur par défaut :** `5e-07`

### `--optimizers.camera-opt.scheduler.warmup-steps INT`
- **Valeur par défaut :** `1000`

### `--optimizers.camera-opt.scheduler.max-steps INT`
- **Valeur par défaut :** `30000`

### `--optimizers.camera-opt.scheduler.ramp {linear,cosine}`
- **Valeur par défaut :** `cosine`

---

## Bilateral Grid Optimizer

### `--optimizers.bilateral-grid.optimizer.lr FLOAT`
- **Valeur par défaut :** `0.002`

### `--optimizers.bilateral-grid.optimizer.eps FLOAT`
- **Valeur par défaut :** `1e-15`

### `--optimizers.bilateral-grid.optimizer.max-norm FLOAT | None`
- **Valeur par défaut :** `None`

### `--optimizers.bilateral-grid.optimizer.weight-decay FLOAT`
- **Valeur par défaut :** `0`

### `--optimizers.bilateral-grid.scheduler.lr-pre-warmup FLOAT`
- **Valeur par défaut :** `0`

### `--optimizers.bilateral-grid.scheduler.lr-final FLOAT | None`
- **Valeur par défaut :** `0.0001`

### `--optimizers.bilateral-grid.scheduler.warmup-steps INT`
- **Valeur par défaut :** `1000`

### `--optimizers.bilateral-grid.scheduler.max-steps INT`
- **Valeur par défaut :** `30000`

### `--optimizers.bilateral-grid.scheduler.ramp {linear,cosine}`
- **Valeur par défaut :** `cosine`

---

# 17. Paramètres les plus importants pour éviter un OOM

## Réduire la fréquence d’évaluation complète
```bash
--steps-per-eval-all-images 0
```

## Réduire la résolution de travail
```bash
--pipeline.datamanager.camera-res-scale-factor 0.5
```

## Éviter le cache image GPU
```bash
--pipeline.datamanager.cache-images cpu
```

## Garder les images hors GPU
```bash
--pipeline.datamanager.images-on-gpu False
```

## Réduire le coût de rendu en évaluation
```bash
--pipeline.model.eval-num-rays-per-chunk 2048
```

## Rendre la densification moins agressive
```bash
--pipeline.model.densify-grad-thresh 0.0008
--pipeline.model.max-gauss-ratio 3.0
--pipeline.model.cull-alpha-thresh 0.15
```

---

# 18. Profil “safe” conseillé pour GPU limité

```bash
--steps-per-eval-all-images 0
--pipeline.datamanager.cache-images cpu
--pipeline.datamanager.images-on-gpu False
--pipeline.datamanager.camera-res-scale-factor 0.5
--pipeline.model.eval-num-rays-per-chunk 2048
--pipeline.model.densify-grad-thresh 0.0008
--pipeline.model.max-gauss-ratio 3.0
--pipeline.model.cull-alpha-thresh 0.15
```
