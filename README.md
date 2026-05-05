# Merapi Anomaly Detection

**Détection automatique d'anomalies dans des séries d'images de surveillance volcanique**  
Application au volcan Merapi (Indonésie) — Projet de recherche appliquée IA + volcanologie

---

## Structure du projet

```
merapi_anomaly/
├── data/
│   ├── raw/                  # Images brutes téléchargées (jamais modifiées)
│   │   └── {year}/{month:02d}/
│   ├── processed/            # Images prétraitées
│   │   └── {year}/{month:02d}/
│   └── index/
│       └── index.csv         # Index centralisé des métadonnées
├── src/
│   ├── utils.py              # Utilitaires partagés
│   ├── scraper.py            # Scraping et téléchargement progressif
│   ├── indexer.py            # Gestion de l'index CSV
│   ├── preprocessing.py      # Prétraitement images (Phase 3)
│   ├── quality_filter.py     # Filtrage qualité (Phase 3)
│   ├── baselines.py          # Méthodes simples de détection (Phase 4)
│   ├── autoencoder.py        # Autoencodeur convolutif (Phase 5)
│   ├── vae.py                # VAE + score anomalie (Phase 5)
│   ├── anomaly_scorer.py     # Interface unifiée de scoring (Phase 5)
│   └── generative.py         # Extensions génératives (Phase 6)
├── notebooks/
│   ├── 01_scraping.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_baselines.ipynb
│   ├── 05_autoencoder.ipynb
│   ├── 06_vae.ipynb
│   └── 07_generative.ipynb
├── app/
│   ├── streamlit_app.py      # Interface volcanologues (Phase 7)
│   └── components/
├── config/
│   └── settings.yaml         # Configuration centrale
├── outputs/
│   ├── figures/
│   ├── scores/
│   └── reports/
├── models/                   # Poids des modèles sauvegardés
├── logs/
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Créer et activer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Installer les dépendances
pip install -r requirements.txt
```

---

## Démarrage rapide — Phase 1

### 1. Scraper un mois et générer l'index

```python
from src.utils import load_config, setup_logger
from src.scraper import MerapiScraper
from src.indexer import MerapiIndexer

config = load_config()
setup_logger(config)

scraper = MerapiScraper(config)
indexer = MerapiIndexer(config)

# Scraper novembre 2014 (environ 1 000 images)
records = scraper.scrape_month(year=2014, month=11)

# Indexer sans télécharger
indexer.upsert(records)

# Télécharger les 50 premières images uniquement
scraper.download_images(records, max_images=50)

# Résumé
indexer.print_summary()
```

### 2. Test en ligne de commande

```bash
# Scraper + télécharger 10 images pour tester
python -m src.scraper --year 2014 --month 11 --max-download 10

# Afficher le résumé de l'index
python -m src.indexer
```

---

## Pipeline complet (10 ans)

```bash
# Pipeline complet : scraping + download + preprocessing + quality + baselines
python run_full_pipeline.py

# Année unique
python run_full_pipeline.py --year 2014

# Scraping + indexation seulement (pas de téléchargement)
python run_full_pipeline.py --no-download

# Limiter le download par mois
python run_full_pipeline.py --max-per-month 100

# Si les images sont déjà téléchargées (preprocessing + baselines seulement)
python run_full_pipeline.py --skip-scraping
```

---

## Application Streamlit

```bash
streamlit run app/streamlit_app.py
```

Fonctionnalités :
- Parcours temporel des images (année / mois / slider)
- Score d'anomalie par image + top priorités
- Heatmap jour × heure
- Vue comparative brute / prétraitée
- Statistiques globales du dataset

---

## Source des données

Site de télésurveillance volcanologique :  
`https://wwwobs.univ-bpclermont.fr/SO/televolc/stereovolc/data/domerapi/`

Accès public — respecter un délai d'au moins 1,5 s entre les requêtes (configuré dans `settings.yaml`).

---

## Feuille de route

| Phase | Statut | Description |
|-------|--------|-------------|
| 1 — Scraping + indexation | ✅ | `scraper.py`, `indexer.py` |
| 2 — EDA | ✅ | `notebooks/02_eda.ipynb` |
| 3 — Prétraitement | ✅ | `preprocessing.py`, `quality_filter.py` |
| 4 — Baselines | ✅ | `baselines.py` — MAD, SSIM, luminosité nocturne |
| 5 — Modèles IA | 🔜 | `autoencoder.py`, `vae.py` |
| 6 — Génératif | 🔜 | `generative.py` (optionnel) |
| 7 — Prototype Streamlit | ✅ | `app/streamlit_app.py` |
