# Nerfstudio `ns-train splatfacto` — Documentation synthétique des paramètres

Ce document présente les principaux paramètres de `ns-train splatfacto` par **thématiques fonctionnelles**.  
Pour chaque paramètre, on indique son **rôle**, sa **valeur par défaut**, et **uniquement si c’est pertinent** l’effet d’une augmentation de sa valeur.

---

# 1. Configuration générale de l’expérience

## `--output-dir PATH`
**Rôle :** répertoire principal où sont stockés les checkpoints, logs et résultats d’entraînement.

**Valeur par défaut :** `outputs`

---

## `--method-name STR`
**Rôle :** nom de la méthode utilisée par Nerfstudio.  
Pour ce pipeline, il s’agit de `splatfacto`.

**Valeur par défaut :** `splatfacto`

---

## `--experiment-name STR | None`
**Rôle :** nom lisible de l’expérience.  
S’il est absent, il est généralement dérivé automatiquement du dataset.

**Valeur par défaut :** `None`

---

## `--project-name STR | None`
**Rôle :** nom du projet auquel rattacher l’expérience.  
Utile pour organiser plusieurs runs dans les outils de logging.

**Valeur par défaut :** `nerfstudio-project`

---

## `--timestamp STR`
**Rôle :** horodatage utilisé pour distinguer les expériences et éviter les collisions de dossiers.

**Valeur par défaut :** `{timestamp}`

---

## `--data PATH | None`
**Rôle :** alias de `--pipeline.datamanager.data`.  
Indique le chemin vers les données d’entrée.

**Valeur par défaut :** `None`

---

## `--prompt STR | None`
**Rôle :** alias de `--pipeline.model.prompt`.  
Paramètre textuel utilisé seulement dans certains pipelines compatibles.

**Valeur par défaut :** `None`

---

## `--relative-model-dir PATH`
**Rôle :** sous-répertoire relatif dans lequel les checkpoints du modèle sont enregistrés.

**Valeur par défaut :** `nerfstudio_models`

---

## `--save-only-latest-checkpoint {True,False}`
**Rôle :** si activé, seul le dernier checkpoint est conservé.  
Réduit l’usage disque mais limite les possibilités de retour arrière.

**Valeur par défaut :** `True`

---

## `--load-scheduler {True,False}`
**Rôle :** recharge l’état du scheduler lors de la restauration d’un checkpoint.  
Permet une reprise plus fidèle de la dynamique d’apprentissage.

**Valeur par défaut :** `True`

---

## `--load-dir PATH | None`
**Rôle :** répertoire contenant un modèle ou un run précédent à recharger.

**Valeur par défaut :** `None`

---

## `--load-step INT | None`
**Rôle :** permet de charger une étape précise depuis `load-dir`.

**Valeur par défaut :** `None`

**Si on augmente cette valeur :** on recharge un checkpoint plus avancé dans l’entraînement.

---

## `--load-config PATH | None`
**Rôle :** chemin vers un fichier YAML de configuration à charger pour rejouer une expérience.

**Valeur par défaut :** `None`

---

## `--load-checkpoint PATH | None`
**Rôle :** chemin vers un checkpoint précis à restaurer.

**Valeur par défaut :** `None`

---

# 2. Exécution et machine

## `--machine.seed INT`
**Rôle :** fixe la seed aléatoire pour améliorer la reproductibilité des runs.

**Valeur par défaut :** `42`

**Si on augmente cette valeur :** cela change surtout la trajectoire aléatoire, sans effet monotone garanti sur la qualité.

---

## `--machine.num-devices INT`
**Rôle :** nombre total de devices utilisés pour l’entraînement, notamment en multi-GPU.

**Valeur par défaut :** `1`

**Si on augmente cette valeur :** on peut paralléliser davantage l’entraînement, au prix d’une configuration plus complexe.

---

## `--machine.num-machines INT`
**Rôle :** nombre de machines utilisées dans un entraînement distribué multi-nœuds.

**Valeur par défaut :** `1`

**Si on augmente cette valeur :** on augmente la capacité distribuée, mais aussi la complexité de synchronisation.

---

## `--machine.machine-rank INT`
**Rôle :** rang de la machine courante dans un entraînement distribué.

**Valeur par défaut :** `0`

**Si on augmente cette valeur :** cela change seulement l’identité du nœud dans le setup distribué.

---

## `--machine.dist-url STR`
**Rôle :** adresse utilisée pour connecter les processus distribués entre eux.

**Valeur par défaut :** `auto`

---

## `--machine.device-type {cpu,cuda,mps}`
**Rôle :** type de device utilisé pour l’entraînement.

**Valeur par défaut :** `cuda`

---

## `--max-num-iterations INT`
**Rôle :** nombre maximal d’itérations d’entraînement.

**Valeur par défaut :** `30000`

**Si on augmente cette valeur :** l’entraînement dure plus longtemps, ce qui peut améliorer le raffinement final mais augmente le temps de calcul.

---

## `--mixed-precision {True,False}`
**Rôle :** active la précision mixte pour réduire l’usage mémoire et accélérer certains calculs sur GPU compatibles.

**Valeur par défaut :** `False`

---

## `--use-grad-scaler {True,False}`
**Rôle :** active un gradient scaler pour stabiliser certains calculs en précision réduite.

**Valeur par défaut :** `False`

---

## `--gradient-accumulation-steps [STR INT ...]`
**Rôle :** accumule les gradients sur plusieurs pas avant une mise à jour.  
Permet de simuler un batch plus grand sans tout faire tenir en mémoire d’un coup.

**Valeur par défaut :** vide

**Si on augmente cette valeur :** on réduit la pression mémoire instantanée mais on espace les mises à jour effectives.

---

## `--start-paused {True,False}`
**Rôle :** démarre l’entraînement en pause, utile pour inspection avant lancement.

**Valeur par défaut :** `False`

---

# 3. Sauvegarde, logs et profiling

## `--steps-per-save INT`
**Rôle :** nombre d’itérations entre deux sauvegardes de checkpoint.

**Valeur par défaut :** `2000`

**Si on augmente cette valeur :** on sauvegarde moins souvent, ce qui réduit l’I/O disque mais augmente la perte potentielle en cas d’arrêt.

---

## `--logging.relative-log-dir PATH`
**Rôle :** répertoire relatif dans lequel les logs sont stockés.

**Valeur par défaut :** `.`

---

## `--logging.steps-per-log INT`
**Rôle :** fréquence de logging des statistiques d’entraînement.

**Valeur par défaut :** `10`

**Si on augmente cette valeur :** les logs sont plus espacés, donc plus légers mais moins détaillés.

---

## `--logging.max-buffer-size INT`
**Rôle :** taille maximale du buffer utilisé pour lisser certaines statistiques.

**Valeur par défaut :** `20`

**Si on augmente cette valeur :** les métriques affichées sont plus lissées mais réagissent moins vite aux variations.

---

## `--logging.profiler {none,basic,pytorch}`
**Rôle :** niveau de profiling activé pour analyser les performances.

**Valeur par défaut :** `basic`

---

## `--log-gradients {True,False}`
**Rôle :** active le logging des gradients pour diagnostiquer la stabilité de l’entraînement.

**Valeur par défaut :** `False`

---

## `--logging.local-writer.enable {True,False}`
**Rôle :** active l’affichage local des statistiques d’entraînement.

**Valeur par défaut :** `True`

---

## `--logging.local-writer.stats-to-track [...]`
**Rôle :** liste des métriques suivies et affichées localement pendant l’entraînement.

**Valeur par défaut :**
- `ITER_TRAIN_TIME`
- `TRAIN_RAYS_PER_SEC`
- `CURR_TEST_PSNR`
- `VIS_RAYS_PER_SEC`
- `TEST_RAYS_PER_SEC`
- `ETA`

---

## `--logging.local-writer.max-log-size INT`
**Rôle :** nombre maximal de lignes affichées localement avant wrapping.

**Valeur par défaut :** `10`

**Si on augmente cette valeur :** plus d’informations sont visibles d’un coup, mais l’affichage devient plus dense.

---

# 4. Visualisation / Viewer

## `--vis {viewer,wandb,tensorboard,comet,viewer+wandb,viewer+tensorboard,viewer+comet,viewer_legacy}`
**Rôle :** choisit le backend de visualisation et de logging.

**Valeur par défaut :** `viewer`

---

## `--viewer.relative-log-filename STR`
**Rôle :** nom du fichier de log utilisé par le viewer.

**Valeur par défaut :** `viewer_log_filename.txt`

---

## `--viewer.websocket-port INT | None`
**Rôle :** port websocket utilisé par le viewer.  
Si absent, un port libre peut être choisi automatiquement.

**Valeur par défaut :** `None`

**Si on augmente cette valeur :** on change simplement le port réseau utilisé.

---

## `--viewer.websocket-port-default INT`
**Rôle :** port par défaut utilisé si aucun port explicite n’est fourni.

**Valeur par défaut :** `7007`

**Si on augmente cette valeur :** on déplace le port par défaut vers une autre valeur.

---

## `--viewer.websocket-host STR`
**Rôle :** adresse réseau sur laquelle le viewer écoute.

**Valeur par défaut :** `0.0.0.0`

---

## `--viewer.num-rays-per-chunk INT`
**Rôle :** nombre de rayons rendus par chunk dans le viewer.  
Ce paramètre influence directement le compromis fluidité / mémoire.

**Valeur par défaut :** `32768`

**Si on augmente cette valeur :** le rendu viewer peut être plus rapide, mais consomme plus de mémoire GPU.

---

## `--viewer.max-num-display-images INT`
**Rôle :** nombre maximal d’images affichées dans le viewer.

**Valeur par défaut :** `512`

**Si on augmente cette valeur :** plus d’images peuvent être affichées, au prix d’un viewer potentiellement plus lourd.

---

## `--viewer.quit-on-train-completion {True,False}`
**Rôle :** ferme automatiquement le viewer à la fin de l’entraînement.

**Valeur par défaut :** `False`

---

## `--viewer.image-format {jpeg,png}`
**Rôle :** format utilisé pour les images transmises au viewer.

**Valeur par défaut :** `jpeg`

---

## `--viewer.jpeg-quality INT`
**Rôle :** qualité de compression JPEG utilisée dans le viewer.

**Valeur par défaut :** `75`

**Si on augmente cette valeur :** l’image affichée est moins compressée donc plus fidèle, mais le flux est plus lourd.

---

## `--viewer.make-share-url {True,False}`
**Rôle :** active la génération d’une URL de partage pour le viewer.

**Valeur par défaut :** `False`

---

## `--viewer.camera-frustum-scale FLOAT`
**Rôle :** contrôle la taille d’affichage des frustums caméra dans le viewer.

**Valeur par défaut :** `0.1`

**Si on augmente cette valeur :** les frustums deviennent plus visibles, mais peuvent encombrer l’affichage.

---

## `--viewer.default-composite-depth {True,False}`
**Rôle :** active par défaut le compositing depth dans le viewer.

**Valeur par défaut :** `True`

---

# 5. Dataset, chargement et cache (Datamanager)

## `--pipeline.datamanager.data PATH | None`
**Rôle :** chemin vers les données d’entrée utilisées pour l’entraînement.

**Valeur par défaut :** `None`

---

## `--pipeline.datamanager.masks-on-gpu {True,False}`
**Rôle :** place les masques sur GPU pour accélérer certains traitements.

**Valeur par défaut :** `False`

---

## `--pipeline.datamanager.images-on-gpu {True,False}`
**Rôle :** place les images directement sur GPU pour accélérer l’accès pendant le training.

**Valeur par défaut :** `False`

---

## `--pipeline.datamanager.cache-images {cpu,gpu}`
**Rôle :** choisit où les images sont mises en cache : sur CPU ou GPU.  
C’est un paramètre important pour le compromis vitesse / VRAM.

**Valeur par défaut :** `gpu`

---

## `--pipeline.datamanager.cache-images-type {uint8,float32}`
**Rôle :** type mémoire utilisé pour stocker les images en cache.

**Valeur par défaut :** `uint8`

---

## `--pipeline.datamanager.max-thread-workers INT | None`
**Rôle :** nombre maximal de threads utilisés pour le chargement et le cache des images.

**Valeur par défaut :** `None`

**Si on augmente cette valeur :** on peut accélérer le chargement, mais avec plus de pression CPU et plus de concurrence système.

---

## `--pipeline.datamanager.camera-res-scale-factor FLOAT`
**Rôle :** facteur d’échelle appliqué à la résolution des images et des caméras associées.  
C’est un levier majeur pour réduire mémoire et coût de calcul.

**Valeur par défaut :** `1.0`

**Si on augmente cette valeur :** on travaille à plus haute résolution, ce qui peut améliorer le détail mais augmente fortement le coût mémoire et calcul.

---

# 6. Évaluation

## `--steps-per-eval-batch INT`
**Rôle :** nombre d’itérations entre deux évaluations sur batchs aléatoires de rayons.

**Valeur par défaut :** `0`

**Si on augmente cette valeur :** l’évaluation batch est moins fréquente, ce qui réduit le coût mais donne moins de visibilité sur la progression.

---

## `--steps-per-eval-image INT`
**Rôle :** nombre d’itérations entre deux évaluations sur une image complète.

**Valeur par défaut :** `100`

**Si on augmente cette valeur :** les évaluations image sont plus espacées, ce qui réduit la charge mais ralentit le feedback.

---

## `--steps-per-eval-all-images INT`
**Rôle :** nombre d’itérations entre deux évaluations complètes sur toutes les images d’évaluation.

**Valeur par défaut :** `1000`

**Si on augmente cette valeur :** les évaluations complètes deviennent plus rares, ce qui réduit les pics mémoire et le temps d’évaluation.

---

## `--pipeline.datamanager.eval-num-images-to-sample-from INT`
**Rôle :** nombre d’images candidates utilisées pour l’évaluation quand on ne prend pas tout le set.

**Valeur par défaut :** `-1`

**Si on augmente cette valeur :** l’évaluation couvre plus d’images potentielles et devient plus représentative, mais potentiellement plus coûteuse.

---

## `--pipeline.datamanager.eval-num-times-to-repeat-images INT`
**Rôle :** nombre d’itérations pendant lesquelles on réutilise les mêmes images d’évaluation avant d’en changer.

**Valeur par défaut :** `-1`

**Si on augmente cette valeur :** on garde plus longtemps les mêmes images, ce qui rend le suivi plus stable mais moins varié.

---

## `--pipeline.datamanager.eval-image-indices {None}|{[INT [INT ...]]}`
**Rôle :** indices des images explicitement choisies pour l’évaluation.

**Valeur par défaut :** `0`

---

## `--pipeline.model.eval-num-rays-per-chunk INT`
**Rôle :** nombre de rayons traités par chunk pendant l’évaluation.  
C’est un paramètre important pour contrôler la mémoire pendant le rendu d’éval.

**Valeur par défaut :** `4096`

**Si on augmente cette valeur :** l’évaluation peut être plus rapide, mais consomme davantage de mémoire GPU.

---

# 7. Scène, rendu et géométrie globale

## `--pipeline.model.enable-collider {True,False}`
**Rôle :** active un collider de scène pour filtrer les rayons selon un volume utile.

**Valeur par défaut :** `True`

---

## `--pipeline.model.collider-params`
**Rôle :** paramètres du collider, généralement plans proche et lointain pour borner la scène.

**Valeur par défaut :** `near_plane 2.0 far_plane 6.0`

---

## `--pipeline.model.background-color {random,black,white}`
**Rôle :** couleur de fond utilisée par le rendu.

**Valeur par défaut :** `random`

---

## `--pipeline.model.rasterize-mode {classic,antialiased}`
**Rôle :** mode de rasterisation utilisé pour le rendu des gaussiennes.

**Valeur par défaut :** `classic`

---

## `--pipeline.model.output-depth-during-training {True,False}`
**Rôle :** active la production de depth maps pendant l’entraînement.

**Valeur par défaut :** `False`

---

# 8. Planning d’entraînement et raffinement

## `--pipeline.model.warmup-length INT`
**Rôle :** nombre d’itérations initiales pendant lesquelles le raffinement est désactivé.

**Valeur par défaut :** `500`

**Si on augmente cette valeur :** on retarde les opérations de raffinement, ce qui stabilise parfois le début mais ralentit l’adaptation fine.

---

## `--pipeline.model.refine-every INT`
**Rôle :** fréquence des étapes de densification / culling des gaussiennes.

**Valeur par défaut :** `100`

**Si on augmente cette valeur :** le raffinement est moins fréquent, ce qui réduit le coût mais ralentit l’évolution de la structure.

---

## `--pipeline.model.resolution-schedule INT`
**Rôle :** cadence de montée progressive en résolution pendant l’entraînement.

**Valeur par défaut :** `3000`

**Si on augmente cette valeur :** la montée en résolution est plus lente, ce qui réduit la charge au début mais retarde l’accès au plein détail.

---

## `--pipeline.model.num-downscales INT`
**Rôle :** nombre de réductions de résolution appliquées au début du training.

**Valeur par défaut :** `2`

**Si on augmente cette valeur :** la résolution initiale est plus basse, ce qui réduit mémoire et coût mais retarde l’apprentissage des détails fins.

---

## `--pipeline.model.stop-split-at INT`
**Rôle :** étape à partir de laquelle on arrête de splitter les gaussiennes.

**Valeur par défaut :** `15000`

**Si on augmente cette valeur :** le split reste actif plus longtemps, ce qui peut enrichir la scène mais augmente le risque de croissance mémoire.

---

## `--pipeline.model.stop-screen-size-at INT`
**Rôle :** étape à partir de laquelle on arrête les opérations liées à la taille écran.

**Valeur par défaut :** `4000`

**Si on augmente cette valeur :** les heuristiques basées sur la taille écran restent actives plus longtemps.

---

# 9. Densification, duplication et pruning des gaussiens

## `--pipeline.model.cull-alpha-thresh FLOAT`
**Rôle :** seuil d’opacité sous lequel les gaussiens sont supprimés.

**Valeur par défaut :** `0.1`

**Si on augmente cette valeur :** le pruning devient plus agressif, ce qui réduit souvent mémoire et nombre de gaussiens, mais peut supprimer du détail utile.

---

## `--pipeline.model.cull-scale-thresh FLOAT`
**Rôle :** seuil au-delà duquel les gaussiens trop grands sont supprimés.

**Valeur par défaut :** `0.5`

**Si on augmente cette valeur :** on tolère des gaussiens plus grands avant suppression, ce qui peut conserver plus de couverture mais aussi plus d’artefacts grossiers.

---

## `--pipeline.model.cull-screen-size FLOAT`
**Rôle :** seuil de taille écran au-delà duquel certains gaussiens sont supprimés.

**Valeur par défaut :** `0.15`

**Si on augmente cette valeur :** on laisse survivre de plus gros gaussiens à l’écran avant culling.

---

## `--pipeline.model.split-screen-size FLOAT`
**Rôle :** seuil de taille écran à partir duquel un gaussien est split.

**Valeur par défaut :** `0.05`

**Si on augmente cette valeur :** le split se déclenche moins facilement, donc la densification basée écran devient moins agressive.

---

## `--pipeline.model.densify-grad-thresh FLOAT`
**Rôle :** seuil de gradient à partir duquel un gaussien est candidat à la densification.

**Valeur par défaut :** `0.0008`

**Si on augmente cette valeur :** la densification devient moins agressive, donc on crée moins de nouveaux gaussiens.

---

## `--pipeline.model.use-absgrad {True,False}`
**Rôle :** utilise la valeur absolue du gradient pour piloter la densification.

**Valeur par défaut :** `True`

---

## `--pipeline.model.densify-size-thresh FLOAT`
**Rôle :** seuil de taille utilisé pour décider entre duplication et split.

**Valeur par défaut :** `0.01`

**Si on augmente cette valeur :** plus de gaussiens sont considérés comme “petits”, donc la duplication peut être favorisée par rapport au split.

---

## `--pipeline.model.n-split-samples INT`
**Rôle :** nombre de sous-échantillons créés lors d’une opération de split.

**Valeur par défaut :** `2`

**Si on augmente cette valeur :** chaque split génère plus d’enfants, ce qui peut enrichir plus vite la scène mais augmente mémoire et coût.

---

## `--pipeline.model.reset-alpha-every INT`
**Rôle :** fréquence de réinitialisation de l’alpha pendant l’entraînement.

**Valeur par défaut :** `30`

**Si on augmente cette valeur :** les resets d’alpha sont plus espacés, donc l’opacité évolue plus librement entre deux remises à zéro.

---

# 10. Représentation visuelle et qualité

## `--pipeline.model.sh-degree INT`
**Rôle :** degré maximal des harmoniques sphériques utilisées pour modéliser l’apparence directionnelle.

**Valeur par défaut :** `3`

**Si on augmente cette valeur :** le modèle peut représenter des variations angulaires plus riches, mais avec plus de paramètres et un coût plus élevé.

---

## `--pipeline.model.sh-degree-interval INT`
**Rôle :** intervalle d’itérations entre deux augmentations progressives du degré SH.

**Valeur par défaut :** `1000`

**Si on augmente cette valeur :** la montée en complexité d’apparence est plus lente, ce qui peut stabiliser le début mais retarder les détails directionnels.

---

## `--pipeline.model.ssim-lambda FLOAT`
**Rôle :** poids de la loss SSIM dans l’objectif total.  
Elle favorise davantage la cohérence structurale que la simple erreur pixel à pixel.

**Valeur par défaut :** `0.2`

**Si on augmente cette valeur :** l’optimisation met plus l’accent sur la structure perceptuelle, parfois au détriment d’une fidélité RGB stricte.

---

## `--pipeline.model.color-corrected-metrics {True,False}`
**Rôle :** applique une correction couleur avant le calcul des métriques d’évaluation.

**Valeur par défaut :** `False`

---

# 11. Initialisation des gaussiens

## `--pipeline.model.random-init {True,False}`
**Rôle :** active une initialisation aléatoire des gaussiens au lieu d’une initialisation guidée par les données.

**Valeur par défaut :** `False`

---

## `--pipeline.model.num-random INT`
**Rôle :** nombre de gaussiens créés lors de l’initialisation aléatoire.

**Valeur par défaut :** `50000`

**Si on augmente cette valeur :** on démarre avec plus de gaussiens, ce qui peut mieux couvrir la scène mais augmente le coût initial.

---

## `--pipeline.model.random-scale FLOAT`
**Rôle :** taille du volume dans lequel les gaussiens aléatoires sont initialisés.

**Valeur par défaut :** `10.0`

**Si on augmente cette valeur :** les gaussiens initiaux sont dispersés dans un volume plus grand, ce qui couvre plus large mais peut ralentir la convergence.

---

# 12. Régularisation et contrôle de forme

## `--pipeline.model.use-scale-regularization {True,False}`
**Rôle :** active une régularisation des scales pour limiter les formes dégénérées.

**Valeur par défaut :** `False`

---

## `--pipeline.model.max-gauss-ratio FLOAT`
**Rôle :** ratio maximal autorisé entre les dimensions d’un gaussien.  
Il sert à limiter les formes trop étirées.

**Valeur par défaut :** `10.0`

**Si on augmente cette valeur :** on autorise des gaussiens plus anisotropes, ce qui peut mieux épouser certaines structures mais aussi créer des artefacts.

---

# 13. Bilateral grid et correction ISP

## `--pipeline.model.use-bilateral-grid {True,False}`
**Rôle :** active une correction de type ISP via un bilateral grid.

**Valeur par défaut :** `False`

---

## `--pipeline.model.grid-shape INT INT INT`
**Rôle :** dimensions du bilateral grid utilisé pour la correction.

**Valeur par défaut :** `16 16 8`

**Si on augmente cette valeur :** la correction peut devenir plus fine et plus expressive, mais coûte plus cher en mémoire et calcul.

---

# 14. Caméra et optimisation de pose

## `--pipeline.model.camera-optimizer.mode {off,SO3xR3,SE3}`
**Rôle :** choisit le mode d’optimisation des poses caméra.

**Valeur par défaut :** `off`

---

## `--pipeline.model.camera-optimizer.trans-l2-penalty FLOAT`
**Rôle :** pénalité L2 appliquée aux corrections de translation des caméras.

**Valeur par défaut :** `0.01`

**Si on augmente cette valeur :** on contraint davantage les translations à rester proches des poses initiales.

---

## `--pipeline.model.camera-optimizer.rot-l2-penalty FLOAT`
**Rôle :** pénalité L2 appliquée aux corrections de rotation des caméras.

**Valeur par défaut :** `0.001`

**Si on augmente cette valeur :** on contraint davantage les rotations à rester proches des poses initiales.

---

# 15. Coefficients de loss

## `--pipeline.model.loss-coefficients.rgb-loss-coarse FLOAT`
**Rôle :** poids de la loss RGB coarse dans l’objectif total.  
Elle guide la fidélité couleur du rendu coarse.

**Valeur par défaut :** `1.0`

**Si on augmente cette valeur :** le modèle donne plus d’importance à l’alignement couleur du rendu coarse.

---

## `--pipeline.model.loss-coefficients.rgb-loss-fine FLOAT`
**Rôle :** poids de la loss RGB fine dans l’objectif total.  
Elle influence directement la qualité couleur du rendu final fin.

**Valeur par défaut :** `1.0`

**Si on augmente cette valeur :** le modèle met davantage l’accent sur la fidélité du rendu fine.

---

# 16. Optimizers

## Means Optimizer

### `--optimizers.means.optimizer.lr FLOAT`
**Rôle :** learning rate appliqué aux positions 3D des gaussiennes (`means`).  
Il contrôle la vitesse à laquelle la géométrie se déplace pendant l’apprentissage.

**Valeur par défaut :** `0.00016`

**Si on augmente cette valeur :** les positions s’ajustent plus vite, avec un risque accru d’instabilité.

---

### `--optimizers.means.optimizer.eps FLOAT`
**Rôle :** epsilon numérique de l’optimiseur.  
Il sert surtout à stabiliser les calculs internes.

**Valeur par défaut :** `1e-15`

**Si on augmente cette valeur :** l’optimiseur devient un peu plus conservateur numériquement, avec une dynamique parfois légèrement modifiée.

---

### `--optimizers.means.optimizer.max-norm FLOAT | None`
**Rôle :** seuil de clipping de norme sur les gradients des `means`.

**Valeur par défaut :** `None`

**Si on augmente cette valeur :** on laisse passer des gradients plus grands avant clipping, donc les mises à jour peuvent devenir plus agressives.

---

### `--optimizers.means.optimizer.weight-decay FLOAT`
**Rôle :** régularisation L2 appliquée aux positions des gaussiennes.

**Valeur par défaut :** `0`

**Si on augmente cette valeur :** on contraint davantage les positions, ce qui peut freiner leur adaptation.

---

### `--optimizers.means.scheduler.lr-pre-warmup FLOAT`
**Rôle :** learning rate utilisé au tout début avant la montée vers le LR principal.

**Valeur par défaut :** `1e-08`

**Si on augmente cette valeur :** le démarrage devient moins prudent et les premiers déplacements sont plus marqués.

---

### `--optimizers.means.scheduler.lr-final FLOAT | None`
**Rôle :** learning rate cible en fin d’entraînement pour les `means`.

**Valeur par défaut :** `1.6e-06`

**Si on augmente cette valeur :** les positions restent plus mobiles en fin de training, avec moins de raffinement fin.

---

### `--optimizers.means.scheduler.warmup-steps INT`
**Rôle :** nombre d’étapes pendant lesquelles le LR monte progressivement.

**Valeur par défaut :** `0`

**Si on augmente cette valeur :** le démarrage est plus progressif et souvent plus stable, mais plus lent.

---

### `--optimizers.means.scheduler.max-steps INT`
**Rôle :** horizon total sur lequel la décroissance du learning rate est calculée.

**Valeur par défaut :** `30000`

**Si on augmente cette valeur :** la décroissance du LR est plus étalée dans le temps.

---

### `--optimizers.means.scheduler.ramp {linear,cosine}`
**Rôle :** forme de la décroissance du learning rate.

**Valeur par défaut :** `cosine`

---

## Features DC Optimizer

### `--optimizers.features-dc.optimizer.lr FLOAT`
**Rôle :** learning rate des composantes couleur DC, qui capturent la couleur de base des gaussiennes.

**Valeur par défaut :** `0.0025`

**Si on augmente cette valeur :** la couleur de base s’adapte plus vite, avec un risque accru d’instabilité visuelle.

---

### `--optimizers.features-dc.optimizer.eps FLOAT`
**Rôle :** epsilon numérique de l’optimiseur des features DC.

**Valeur par défaut :** `1e-15`

**Si on augmente cette valeur :** l’optimiseur devient un peu plus stable numériquement, avec une légère modification possible de la dynamique.

---

### `--optimizers.features-dc.optimizer.max-norm FLOAT | None`
**Rôle :** seuil de clipping des gradients pour les features DC.

**Valeur par défaut :** `None`

**Si on augmente cette valeur :** on autorise des gradients plus grands avant limitation.

---

### `--optimizers.features-dc.optimizer.weight-decay FLOAT`
**Rôle :** régularisation L2 sur les composantes couleur DC.

**Valeur par défaut :** `0`

**Si on augmente cette valeur :** on contraint davantage les couleurs de base, ce qui peut lisser l’apprentissage.

---

### `--optimizers.features-dc.scheduler {None}`
**Rôle :** scheduler du learning rate pour les features DC.  
Ici, aucun scheduler n’est utilisé par défaut.

**Valeur par défaut :** `None`

---

## Features Rest Optimizer

### `--optimizers.features-rest.optimizer.lr FLOAT`
**Rôle :** learning rate des composantes SH non-DC, responsables des variations directionnelles de couleur.

**Valeur par défaut :** `0.000125`

**Si on augmente cette valeur :** les effets directionnels apprennent plus vite, avec un risque plus élevé d’artefacts.

---

### `--optimizers.features-rest.optimizer.eps FLOAT`
**Rôle :** epsilon numérique de l’optimiseur des features SH restantes.

**Valeur par défaut :** `1e-15`

**Si on augmente cette valeur :** la stabilité numérique augmente légèrement, au prix d’une dynamique un peu modifiée.

---

### `--optimizers.features-rest.optimizer.max-norm FLOAT | None`
**Rôle :** seuil de clipping des gradients pour les features-rest.

**Valeur par défaut :** `None`

**Si on augmente cette valeur :** on permet des mises à jour plus fortes avant clipping.

---

### `--optimizers.features-rest.optimizer.weight-decay FLOAT`
**Rôle :** régularisation L2 sur les composantes SH restantes.

**Valeur par défaut :** `0`

**Si on augmente cette valeur :** on pénalise davantage les valeurs élevées, ce qui peut freiner l’expressivité directionnelle.

---

### `--optimizers.features-rest.scheduler {None}`
**Rôle :** scheduler du learning rate pour les features-rest.  
Aucun scheduler n’est appliqué par défaut.

**Valeur par défaut :** `None`

---

## Opacities Optimizer

### `--optimizers.opacities.optimizer.lr FLOAT`
**Rôle :** learning rate appliqué aux opacités des gaussiennes.

**Valeur par défaut :** `0.05`

**Si on augmente cette valeur :** les opacités changent plus vite, ce qui peut accélérer le pruning ou la structuration, mais avec plus d’instabilité.

---

### `--optimizers.opacities.optimizer.eps FLOAT`
**Rôle :** epsilon numérique de l’optimiseur des opacités.

**Valeur par défaut :** `1e-15`

**Si on augmente cette valeur :** on rend l’optimiseur légèrement plus robuste numériquement.

---

### `--optimizers.opacities.optimizer.max-norm FLOAT | None`
**Rôle :** seuil de clipping des gradients sur les opacités.

**Valeur par défaut :** `None`

**Si on augmente cette valeur :** on laisse passer des variations d’opacité plus fortes avant clipping.

---

### `--optimizers.opacities.optimizer.weight-decay FLOAT`
**Rôle :** régularisation L2 sur les opacités.

**Valeur par défaut :** `0`

**Si on augmente cette valeur :** on freine davantage l’évolution libre des opacités.

---

### `--optimizers.opacities.scheduler {None}`
**Rôle :** scheduler du learning rate pour les opacités.  
Aucun scheduler n’est utilisé par défaut.

**Valeur par défaut :** `None`

---

## Scales Optimizer

### `--optimizers.scales.optimizer.lr FLOAT`
**Rôle :** learning rate appliqué aux tailles des gaussiennes.

**Valeur par défaut :** `0.005`

**Si on augmente cette valeur :** les tailles évoluent plus vite, ce qui peut accélérer l’adaptation mais aussi créer des gaussiens instables.

---

### `--optimizers.scales.optimizer.eps FLOAT`
**Rôle :** epsilon numérique de l’optimiseur des scales.

**Valeur par défaut :** `1e-15`

**Si on augmente cette valeur :** la stabilité numérique augmente légèrement.

---

### `--optimizers.scales.optimizer.max-norm FLOAT | None`
**Rôle :** seuil de clipping des gradients sur les scales.

**Valeur par défaut :** `None`

**Si on augmente cette valeur :** on permet des changements de taille plus importants avant limitation.

---

### `--optimizers.scales.optimizer.weight-decay FLOAT`
**Rôle :** régularisation L2 appliquée aux scales.

**Valeur par défaut :** `0`

**Si on augmente cette valeur :** on contraint davantage l’amplitude des scales.

---

### `--optimizers.scales.scheduler {None}`
**Rôle :** scheduler du learning rate pour les scales.  
Aucun scheduler n’est utilisé par défaut.

**Valeur par défaut :** `None`

---

## Quats Optimizer

### `--optimizers.quats.optimizer.lr FLOAT`
**Rôle :** learning rate appliqué aux rotations des gaussiennes, représentées par quaternions.

**Valeur par défaut :** `0.001`

**Si on augmente cette valeur :** les orientations changent plus vite, ce qui peut aider l’adaptation mais aussi rendre la forme plus instable.

---

### `--optimizers.quats.optimizer.eps FLOAT`
**Rôle :** epsilon numérique de l’optimiseur des quaternions.

**Valeur par défaut :** `1e-15`

**Si on augmente cette valeur :** on stabilise légèrement les calculs internes.

---

### `--optimizers.quats.optimizer.max-norm FLOAT | None`
**Rôle :** seuil de clipping des gradients sur les quaternions.

**Valeur par défaut :** `None`

**Si on augmente cette valeur :** on autorise des corrections de rotation plus fortes avant clipping.

---

### `--optimizers.quats.optimizer.weight-decay FLOAT`
**Rôle :** régularisation L2 appliquée aux quaternions.

**Valeur par défaut :** `0`

**Si on augmente cette valeur :** on contraint davantage les rotations apprises.

---

### `--optimizers.quats.scheduler {None}`
**Rôle :** scheduler du learning rate pour les quaternions.  
Aucun scheduler n’est appliqué par défaut.

**Valeur par défaut :** `None`

---

## Camera Optimizer

### `--optimizers.camera-opt.optimizer.lr FLOAT`
**Rôle :** learning rate de l’optimisation des poses caméra.

**Valeur par défaut :** `0.0001`

**Si on augmente cette valeur :** les poses caméra se corrigent plus vite, avec un risque plus grand de dérive.

---

### `--optimizers.camera-opt.optimizer.eps FLOAT`
**Rôle :** epsilon numérique de l’optimiseur caméra.

**Valeur par défaut :** `1e-15`

**Si on augmente cette valeur :** l’optimiseur devient un peu plus robuste numériquement.

---

### `--optimizers.camera-opt.optimizer.max-norm FLOAT | None`
**Rôle :** seuil de clipping des gradients pour les poses caméra.

**Valeur par défaut :** `None`

**Si on augmente cette valeur :** on autorise des corrections de pose plus fortes avant limitation.

---

### `--optimizers.camera-opt.optimizer.weight-decay FLOAT`
**Rôle :** régularisation L2 appliquée aux paramètres du camera optimizer.

**Valeur par défaut :** `0`

**Si on augmente cette valeur :** on freine davantage l’amplitude des corrections de pose.

---

### `--optimizers.camera-opt.scheduler.lr-pre-warmup FLOAT`
**Rôle :** learning rate initial très faible avant warmup pour l’optimisation caméra.

**Valeur par défaut :** `0`

**Si on augmente cette valeur :** les corrections caméra commencent plus tôt et plus fort.

---

### `--optimizers.camera-opt.scheduler.lr-final FLOAT | None`
**Rôle :** learning rate final visé pour l’optimisation caméra.

**Valeur par défaut :** `5e-07`

**Si on augmente cette valeur :** les caméras restent plus mobiles en fin d’entraînement.

---

### `--optimizers.camera-opt.scheduler.warmup-steps INT`
**Rôle :** nombre d’étapes de warmup avant d’atteindre le LR principal pour les caméras.

**Valeur par défaut :** `1000`

**Si on augmente cette valeur :** l’optimisation caméra démarre plus progressivement.

---

### `--optimizers.camera-opt.scheduler.max-steps INT`
**Rôle :** horizon total du scheduler caméra.

**Valeur par défaut :** `30000`

**Si on augmente cette valeur :** la décroissance du LR caméra est plus étalée.

---

### `--optimizers.camera-opt.scheduler.ramp {linear,cosine}`
**Rôle :** forme de décroissance du learning rate caméra.

**Valeur par défaut :** `cosine`

---

## Bilateral Grid Optimizer

### `--optimizers.bilateral-grid.optimizer.lr FLOAT`
**Rôle :** learning rate appliqué aux paramètres du bilateral grid.

**Valeur par défaut :** `0.002`

**Si on augmente cette valeur :** la correction ISP s’adapte plus vite, avec un risque accru d’instabilité ou de sur-correction.

---

### `--optimizers.bilateral-grid.optimizer.eps FLOAT`
**Rôle :** epsilon numérique de l’optimiseur du bilateral grid.

**Valeur par défaut :** `1e-15`

**Si on augmente cette valeur :** on améliore légèrement la robustesse numérique.

---

### `--optimizers.bilateral-grid.optimizer.max-norm FLOAT | None`
**Rôle :** seuil de clipping des gradients sur le bilateral grid.

**Valeur par défaut :** `None`

**Si on augmente cette valeur :** on autorise des mises à jour plus fortes avant limitation.

---

### `--optimizers.bilateral-grid.optimizer.weight-decay FLOAT`
**Rôle :** régularisation L2 appliquée aux paramètres du bilateral grid.

**Valeur par défaut :** `0`

**Si on augmente cette valeur :** on contraint davantage les corrections apprises.

---

### `--optimizers.bilateral-grid.scheduler.lr-pre-warmup FLOAT`
**Rôle :** learning rate initial avant warmup pour le bilateral grid.

**Valeur par défaut :** `0`

**Si on augmente cette valeur :** les corrections ISP commencent plus tôt avec plus d’amplitude.

---

### `--optimizers.bilateral-grid.scheduler.lr-final FLOAT | None`
**Rôle :** learning rate final visé pour le bilateral grid.

**Valeur par défaut :** `0.0001`

**Si on augmente cette valeur :** la correction reste plus active en fin d’entraînement.

---

### `--optimizers.bilateral-grid.scheduler.warmup-steps INT`
**Rôle :** nombre d’étapes de warmup du bilateral grid.

**Valeur par défaut :** `1000`

**Si on augmente cette valeur :** l’apprentissage du bilateral grid devient plus progressif au démarrage.

---

### `--optimizers.bilateral-grid.scheduler.max-steps INT`
**Rôle :** horizon total du scheduler du bilateral grid.

**Valeur par défaut :** `30000`

**Si on augmente cette valeur :** la décroissance du LR est plus étalée dans le temps.

---

### `--optimizers.bilateral-grid.scheduler.ramp {linear,cosine}`
**Rôle :** forme de décroissance du learning rate du bilateral grid.

**Valeur par défaut :** `cosine`

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
