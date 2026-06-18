# video_to_ply

# video_to_ply

`video_to_ply` est un pipeline permettant de générer un fichier 3D `.ply` de **splats gaussiens** à partir d’une ou plusieurs **vidéos** ou d’un **dossier d’images**

Le projet s’appuie sur :

- [Nerfstudio](https://docs.nerf.studio/)
- [COLMAP](https://colmap.github.io/)
- [Hierarchical Localization (HLOC)](https://github.com/cvg/Hierarchical-Localization)

et sur un post-traitement final pour produire un `.ply` exploitable dans des outils 3D.

Le résultat produit n’est pas un simple nuage de points : c’est un **`.ply` enrichi** contenant les paramètres des gaussiennes appris par le modèle :

- position
- échelle
- rotation
- opacité
- attributs de couleur
- autres attributs nécessaires au rendu

Ce fichier peut ensuite être ouvert dans un viewer compatible Gaussian Splats, comme [SuperSplat](https://superspl.at/editor).

---

## Présentation générale

Le pipeline automatise les étapes suivantes :

- préparation des images à partir d’une vidéo ou d’un dossier d’entrée
- reconstruction photogrammétrique des poses caméra
- entraînement d’un modèle `splatfacto` avec Nerfstudio
- export d’un `.ply` de splats gaussiens
- nettoyage final du `.ply` à partir de la reconstruction sparse COLMAP

Chaîne globale :

```text
VIDEO / IMAGES
    ↓
préparation des images
    ↓
préprocess photogrammétrique
(COLMAP / HLOC via Nerfstudio)
    ↓
transforms.json + sparse COLMAP
    ↓
entraînement Gaussian Splat
    ↓
export `.ply`
    ↓
nettoyage et export du `.ply` final
```

Le livrable final recherché est :

```text
<root>/exports/<basename>.ply
```

---

## Prérequis

Le pipeline est conçu pour tourner sur un environnement **Linux** avec :

- `bash`
- **GPU NVIDIA**
- **CUDA**
- **conda / mamba**

Le cas d’usage principal repose sur un entraînement **Gaussian Splat sur GPU** avec `splatfacto`.

L’environnement prêt à l’emploi utilisé pour ce projet est fourni ici :

```text
environment/ign.slurm/conda_env.yml
```

Il contient notamment la stack nécessaire pour :

- **PyTorch**
- **Nerfstudio**
- **COLMAP / pycolmap**
- **ffmpeg**
- **HLOC**

Exemple :

```bash
mamba env create -f environment/ign.slurm/conda_env.yml
mamba activate gsplat
```

Le projet a été pensé et testé principalement pour :

- un contexte **serveur / Slurm**
- un notebook **Google Colab**

---

## Entrées

Le script principal supporte deux modes.

### Mode vidéo

Entrée :
- une ou plusieurs vidéos via `--video`

Le pipeline :
- copie les vidéos dans l’espace de travail
- extrait les frames
- traite ensuite les images produites

Options utiles :
- `--fps`
- `--num-frames`
- variables d’environnement optionnelles :
  - `VIDEO_START`
  - `VIDEO_END`

### Mode images

Entrée :
- un dossier d’images via `--images`

Le pipeline :
- prépare les images
- lance directement le préprocess photogrammétrique

---

## Sorties

Sous le dossier `--root`, le pipeline génère plusieurs fichiers intermédiaires, mais le livrable final recherché est :

```text
<root>/exports/<basename>.ply
```

Ce fichier est un **PLY enrichi pour représenter des Gaussian Splats**.

Les sorties intermédiaires importantes sont notamment :

- `ori/transforms.json`
- `ori/colmap/...`
- les checkpoints d’entraînement Nerfstudio
- un `.ply` exporté avant nettoyage
- un `.ply` final nettoyé dans `exports/`

---

## Étapes du pipeline

### 1. Préparation des entrées

#### Si l’entrée est une vidéo

Les frames sont extraites via `ffmpeg` grâce à :

- `scripts/extract_frames.sh`

Les images sont stockées dans :

```text
<root>/ori/images/
```

#### Si l’entrée est un dossier d’images

Les images sont préparées / copiées via :

- `scripts/prepare_images.sh`

Objectif :
- obtenir un dossier d’images homogène et éventuellement sous-échantillonné pour la suite du pipeline

---

### 2. Préprocess photogrammétrique

Le pipeline lance un préprocess via Nerfstudio pour produire :

- les poses caméra
- `transforms.json`
- la reconstruction sparse COLMAP

Script utilisé :

- `scripts/preprocess_nerfstudio.sh`

#### Profils disponibles

- `colmap`  
  pipeline photogrammétrique classique basé directement sur **COLMAP**.  
  C’est l’option la plus simple et la plus standard pour reconstruire les poses caméra et la géométrie sparse.

- `hloc`  
  pipeline basé sur **Hierarchical Localization (HLOC)**.  
  Il utilise des descripteurs et appariements plus modernes que COLMAP seul, ce qui peut être plus robuste sur des scènes difficiles, répétitives ou avec de grands changements de point de vue.

- `hloc-lightblue`  
  variante de `hloc` avec un réglage spécifique au projet.  
  C’est un profil plus spécialisé, pensé pour certains jeux de données ou contraintes internes, tout en restant dans la logique HLOC.

Sortie typique :

```text
<root>/ori/
├── images/
├── transforms.json
└── colmap/
```

Cette étape reconstruit la géométrie de prise de vue et prépare les données pour l’entraînement.

---

### 3. Estimation des plans near / far

Avant l’entraînement, le pipeline estime automatiquement des plans proches / lointains à partir du sparse COLMAP.

Script utilisé :

- `scripts/estimate_planes.py`

Les valeurs estimées sont injectées dans l’entraînement via le collider.

Cette étape améliore la cohérence du rendu et du training sur certaines scènes.

---

### 4. Entraînement du modèle

Le pipeline entraîne ensuite un modèle `splatfacto` de la librairie **Nerfstudio** sur les données de `ori/`.

Script utilisé :

- `scripts/train.sh`

Sorties :
- configuration de run
- checkpoints
- logs d’entraînement

---

### 5. Export en `.ply`

Après entraînement, le pipeline exporte le résultat vers un fichier `.ply` grâce au script :

- `scripts/export_splat_to_ply.sh`

Il s’agit d’un `.ply` enrichi décrivant des **Gaussian Splats**, et non d’un simple nuage de points.

---

### 6. Nettoyage final du `.ply`

Le `.ply` exporté est post-traité à partir de la reconstruction sparse COLMAP.

Script utilisé :

- `scripts/cleaning/clean-ply.py`

Entrées :
- le `.ply` exporté
- `points3D.bin`
- `dataparser_transforms.json`

Sortie :
- un `.ply` final dans :

```text
<root>/exports/<basename>.ply
```

#### Principe

Le script :

1. charge les points 3D COLMAP
2. applique le même transform que Nerfstudio pour les remettre dans le bon repère
3. filtre les points COLMAP aberrants
4. estime une distance de voisinage caractéristique
5. conserve uniquement les gaussiennes :
   - proches des points COLMAP
   - ou incluses dans la boîte englobante de la scène reconstruite

L’objectif est de supprimer les gaussiennes isolées, parasites ou trop éloignées de la géométrie réellement reconstruite par COLMAP, afin d’obtenir un `.ply` final plus propre.

---

## Paramètres principaux

### Entrées

- `--video <file ...>` : une ou plusieurs vidéos
- `--images <dir>` : dossier d’images
- `--root <dir>` : dossier de sortie principal
- `--name <name>` : nom de base des sorties

### Extraction vidéo

- `--fps <int ...>`
- `--num-frames <int ...>`

### Profils

- `--preprocess-profile <name>` : `colmap | hloc | hloc-lightblue`
- `--gsplat-profile <name>` : `fast | balanced | quality | quality_plus | best`

### Options pipeline

- `--automask`
- `--skip-conda`
- `--no-proxy`
- `--skip-frame-extraction`
- `--skip-preprocess`
- `--skip-training`
- `--skip-export`

---

## Reprise et exécution partielle

Le pipeline est conçu pour être relancé sans tout recalculer.

Il peut :
- sauter l’extraction si les images existent déjà
- sauter le préprocess si `transforms.json` est déjà présent
- sauter l’entraînement ou l’export via les flags `--skip-*`

Cela facilite :

- les reprises après erreur
- le debug
- les exécutions par étapes


---

## Architecture du projet

Structure principale :

```text
video_to_ply/
├── config/
├── doc/
├── environment/
├── install/
├── io/
├── profiles/
│   ├── gsplat/
│   └── preprocess/
├── scripts/
│   ├── cleaning/
│   ├── masking/
│   ├── rendering/
│   ├── analysis/
│   ├── extract_frames.sh
│   ├── prepare_images.sh
│   ├── preprocess_nerfstudio.sh
│   ├── train.sh
│   ├── export_nerf_to_ply.sh
│   ├── export_splat_to_ply.sh
│   └── estimate_planes.py
├── run.sh
└── README.md
```

### Rôle des principaux dossiers

- `config/` : configuration globale
- `profiles/` : presets de préprocess et d’entraînement
- `scripts/` : briques métier du pipeline
- `environment/` : environnements d’exécution (Slurm, Colab, Kaggle, macOS)
- `doc/` : documentation et notes
- `io/` : outils de transfert / import-export

---

## Exemples

### Vidéo

```bash
./run.sh \
  --video input.mov \
  --numframes 150 \
  --root runs/test \
  --preprocess-profile colmap \
  --gsplat-profile quality
```

### Images

```bash
./run.sh \
  --images ./imgs \
  --root runs/test \
  --preprocess-profile hloc \
  --gsplat-profile quality
```

### Reprendre sans refaire le préprocess

```bash
./run.sh \
  --images ./imgs \
  --root runs/test \
  --skip-preprocess \
  --gsplat-profile quality
```
