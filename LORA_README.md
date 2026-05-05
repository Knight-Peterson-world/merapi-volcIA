# Fine-tuning LoRA — Stable Diffusion 1.5 sur le dataset Merapi

## Vue d'ensemble

Ce module entraîne un adaptateur **LoRA** (Low-Rank Adaptation) sur le U-Net de
**Stable Diffusion 1.5** à partir des images de surveillance du Mont Merapi
(réseau VELI/TéléVolc, caméras Canon EOS 1100D).

### Caractéristiques du dataset

| Propriété     | Valeur                      |
|---------------|-----------------------------|
| Images totales | 2 671 (flag `usable`)      |
| Format         | PNG 256×256, niveaux de gris |
| Jour (6h-18h)  | 802 images                 |
| Nuit (<6h, ≥18h)| 1 683 images              |
| Caméras        | Suki_Canon, Kalor_Canon    |
| Années         | 2014–2018                  |

### Architecture

```
merapi_anomaly/
├── train_lora_merapi.py        # Script d'entraînement LoRA principal
├── evaluate_lora_merapi.py     # Évaluation FID / SSIM
├── prepare_dataset_lora.py     # Préparation dataset (split, captions)
├── data/
│   ├── index/index.csv         # Index des images
│   ├── processed/              # Images prétraitées (PNG 256×256)
│   └── lora_dataset/           # Dataset préparé (après prepare_dataset_lora.py)
│       ├── train/day/
│       ├── train/night/
│       ├── val/day/
│       └── val/night/
└── outputs/lora_merapi/
    ├── lora_merapi_final/      # Poids LoRA finaux
    ├── checkpoints/            # Checkpoints périodiques
    ├── samples/                # Images générées (avant/après FT)
    ├── textual_inversion_merapi/  # Token <merapi> (si activé)
    ├── training_loss.csv       # Historique des pertes
    └── evaluation/             # Résultats d'évaluation
```

---

## Installation

```bash
# Depuis l'environnement Anaconda (Python 3.10, Mac M1)
/opt/anaconda3/bin/pip install peft accelerate diffusers transformers torch
/opt/anaconda3/bin/pip install scikit-image scipy torchvision  # pour l'évaluation
```

## Utilisation

### 1. Préparer le dataset (optionnel)

```bash
cd merapi_anomaly/
python prepare_dataset_lora.py
python prepare_dataset_lora.py --val-ratio 0.15 --resolution 512
```

Cela crée `data/lora_dataset/` avec les splits train/val séparés jour/nuit et
les fichiers `metadata.jsonl` au format HuggingFace.

### 2. Entraîner le LoRA

```bash
# Configuration par défaut (recommandée pour Mac M1)
python train_lora_merapi.py

# Configuration personnalisée
python train_lora_merapi.py \
    --epochs 100 \
    --lr 1e-5 \
    --batch-size 2 \
    --grad-accum 4 \
    --lora-rank 4 \
    --resolution 256 \
    --max-images 500

# Avec Textual Inversion (token <merapi>)
python train_lora_merapi.py --textual-inversion --token "<merapi>"
```

### 3. Évaluer

```bash
# SSIM uniquement (rapide)
python evaluate_lora_merapi.py --num-gen 50

# SSIM + FID (nécessite torchvision + scipy)
python evaluate_lora_merapi.py --fid --num-gen 100

# Avec un LoRA spécifique
python evaluate_lora_merapi.py --lora-path outputs/lora_merapi/checkpoints/lora_epoch_050
```

---

## Hyperparamètres recommandés (Mac M1, 8-16 GB)

| Paramètre           | Valeur recommandée | Notes                               |
|---------------------|--------------------|-------------------------------------|
| `--lr`              | `1e-5`             | Réduire à `5e-6` si instable        |
| `--batch-size`      | `2`                | Max 2 en 256×256 sur 8 GB           |
| `--grad-accum`      | `4`                | Effective batch = 8                  |
| `--epochs`          | `100`              | Early stopping à 20 epochs          |
| `--lora-rank`       | `4`                | 4-8, augmenter si sous-fitting      |
| `--resolution`      | `256`              | 512 possible mais plus lent         |
| `--max-images`      | `500`              | Augmenter si la mémoire le permet   |
| `--ema-decay`       | `0.9999`           | Stabilise l'entraînement            |

### Précision

- **MPS (Apple Silicon)** : `fp32` uniquement — MPS ne supporte pas fp16 nativement
- **CUDA** : Ajouter `--fp16` pour accélérer l'entraînement

---

## Conditionnement jour/nuit

Les images diurnes et nocturnes ont des caractéristiques très différentes :
- **Jour** : terrain gris, coulées de lave solidifiée, panaches de fumée
- **Nuit** : incandescence, lave en fusion visible, fond sombre

Le script utilise des **prompts conditionnés** différents pour chaque catégorie,
évitant le mélange non supervisé recommandé par la littérature sur le domaine-gap.

---

## Textual Inversion

Le token `<merapi>` capture le « concept visuel » du Merapi tel que vu par les
caméras de surveillance VELI. Il est initialisé depuis l'embedding de "volcano"
et entraîné conjointement avec le LoRA.

```bash
python train_lora_merapi.py --textual-inversion --token "<merapi>"
```

À l'inférence, utiliser le token dans le prompt :
```
"<merapi> volcanic eruption, nighttime, glowing lava"
```

---

## Métriques d'évaluation

- **FID** (Fréchet Inception Distance) : mesure la distance entre les distributions
  d'images réelles et générées. Plus bas = meilleur.
- **SSIM** (Structural Similarity Index) : similarité structurelle entre paires
  d'images. Plus haut = meilleur (max 1.0).

Le script `evaluate_lora_merapi.py` compare automatiquement :
1. Baseline SD 1.5 (sans LoRA) vs images réelles
2. SD 1.5 + LoRA vs images réelles
3. Grille visuelle avant/après fine-tuning
