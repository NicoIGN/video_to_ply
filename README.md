# video_to_ply

Pipeline pour générer un nuage de points 3D `.ply` à partir d’une vidéo, en utilisant Nerfstudio (NeRF) et COLMAP.

---

## 🧠 Vue d’ensemble

Le pipeline transforme une vidéo en reconstruction 3D en plusieurs étapes :

VIDEO → IMAGES → POSES CAMÉRA → NeRF → POINT CLOUD (.ply)

---

## 🔁 Étape 1 — Extraction des frames

La vidéo est convertie en images avec `ffmpeg`.

- Paramètre principal : `--fps`
- Les images sont générées directement dans :

<root>/ori/images/

Exemple :
- frame_0001.png  
- frame_0002.png  
- ...

👉 Objectif : obtenir un ensemble d’images exploitables par COLMAP et NeRF.

---

## 🧭 Étape 2 — Reconstruction des caméras (COLMAP)

`ns-process-data` lance COLMAP pour :

- détecter des points caractéristiques dans les images  
- les faire correspondre entre les frames  
- reconstruire les positions et orientations des caméras  

Sortie :

<root>/ori/  
├── images/  
├── transforms.json  
├── sparse/  

👉 Cette étape reconstruit la géométrie de la prise de vue.

---

## 🧠 Étape 3 — Entraînement NeRF

Entraînement d’un modèle NeRF avec Nerfstudio :

- `nerfacto` → CPU  
- `splatfacto` → GPU  

Entrée :
<root>/ori/

Sortie :
<root>/outputs/ori/<model>/<timestamp>/  
├── config.yml  
├── nerfstudio_models/  

👉 Le modèle apprend à représenter la scène en 3D à partir des images.

---

## 📦 Étape 4 — Export en point cloud (.ply)

Conversion du modèle en nuage de points :

- commande : `ns-export pointcloud`
- calcul des normales via Open3D

Sortie :
<root>/exports/*.ply

👉 Résultat final exploitable dans des outils 3D.

---

## ⚙️ Paramètres principaux

- `--video` : chemin vers la vidéo  
- `--root` : dossier de sortie  
- `--fps` : nombre d’images extraites par seconde  
- `--device` : cpu ou gpu  
- `--max-iter` : nombre d’itérations d’entraînement  

---

## ♻️ Comportement du pipeline

- Reprise automatique : saute les étapes déjà complétées  
- Structure de dossier unifiée sous `<root>`  
- Compatible CPU (lent) et GPU (rapide)  

---

## ✅ Résultat

Un fichier `.ply` contenant un nuage de points 3D, compatible avec :

- Blender  
- MeshLab  
- CloudCompare  
- Three.js  

---

## 🚀 Résumé

Ce pipeline automatise :

Transformer une vidéo en une scène 3D exploitable.
