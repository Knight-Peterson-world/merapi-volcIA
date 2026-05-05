# VolcIA — Documentation complète de l'application

> **Stage de Master 2 — Laboratoire Magmas et Volcans (LMV), Clermont-Ferrand**  
> Application Streamlit d'analyse, de détection d'anomalies et de génération d'images pour la surveillance volcanique du Merapi (Indonésie).

---

## Table des matières

1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [Architecture technique](#2-architecture-technique)
3. [Pipeline de données (en coulisses)](#3-pipeline-de-données-en-coulisses)
4. [Page 1 — 🏠 Accueil](#4-page-1--accueil)
5. [Page 2 — 🔍 Exploration](#5-page-2--exploration)
6. [Page 3 — 🚨 Anomalies](#6-page-3--anomalies)
7. [Page 4 — 🖼️ Galerie temporelle](#7-page-4--galerie-temporelle)
8. [Page 5 — 🤖 Génération IA](#8-page-5--génération-ia)
9. [Page 6 — 🌋 Simulation d'écoulements](#9-page-6--simulation-découlements)
10. [Page 7 — 📈 Statistiques](#10-page-7--statistiques)
11. [Page 8 — 📖 À propos](#11-page-8--à-propos)
12. [Page 9 — 🔬 DINOv2 + PatchCore](#12-page-9--dinov2--patchcore)
13. [Page 10 — ⚡ Early Warning](#13-page-10--early-warning)
14. [Page 11 — 🧪 Analyse avancée](#14-page-11--analyse-avancée)
15. [Page 12 — 🌋 Analyse volcanique avancée](#15-page-12--analyse-volcanique-avancée)
16. [Récapitulatif des métriques](#16-récapitulatif-des-métriques)
17. [Conclusion et feuille de route](#17-conclusion-et-feuille-de-route)

---

## 1. Vue d'ensemble du projet

### Objectif scientifique

VolcIA est un outil autonome destiné aux volcanologues. Il permet de :

- **Détecter des anomalies visuelles** sur les images de la caméra Kalor (Merapi, Indonésie) sans supervision humaine.
- **Reconstruire** une image vers un état « normal » via un modèle de diffusion, puis mesurer la différence pour localiser les zones anormales.
- **Classifier** automatiquement les événements (pyroclastique, coulée de lave, nuage, activité normale).
- **Anticiper** des événements volcaniques via un signal précurseur (Early Warning).
- **Générer** des images synthétiques réalistes de volcans par text-to-image, utiles pour la communication et l'augmentation de données.
- **Simuler** des écoulements de lave en 2D à des fins de démonstration pédagogique.

### Données sources

| Attribut | Valeur |
|----------|--------|
| Source | Réseau VELI/TéléVolc (OPGC/LMV) — wwwobs.univ-bpclermont.fr |
| Caméras | Kalor et Suki (Canon EOS 1100D) |
| Localisation | Merapi, Indonésie |
| Période | 2014 – 2024 (10 ans) |
| Volume indexé | ~76 820 images |
| Résolution brute | 4272 × 2848 px (JPG) |
| Résolution traitée | 512 × 512 px (PNG, normalisation min-max) |
| Événement clé | Septembre 2020 — effondrement du cratère |

### Technologies principales

Python 3.10 · Streamlit · PyTorch · DINOv2 · Stable Diffusion 1.5 · LoRA (PEFT) · scikit-learn · NumPy · Pandas · Matplotlib · Seaborn

### Pourquoi ces technologies ?

Chaque technologie a été retenue pour une raison précise en rapport avec les contraintes du projet.

**Python 3.10** est le standard de facto de la communauté ML/IA et dispose d'un écosystème sans équivalent pour le traitement d'images et l'apprentissage profond. Il permet de passer sans friction de l'exploration dans un notebook à un script de production.

**Streamlit** a été choisi comme framework d'interface parce qu'il permet de construire un tableau de bord interactif en Python pur, sans avoir à écrire une seule ligne de JavaScript ou de HTML. Il s'adresse directement aux scientifiques qui veulent exposer leurs analyses sans devenir développeurs web. La mise en cache native (`@st.cache_data`) est un avantage concret pour ne pas re-charger 76 820 lignes de CSV à chaque interaction.

**PyTorch** est préféré à TensorFlow pour sa flexibilité en mode recherche (exécution dynamique, débogage facile) et parce que DINOv2 et Stable Diffusion sont tous deux distribués sous PyTorch. Sur Apple Silicon (M1/M2), PyTorch dispose du backend **MPS** qui exploite le GPU intégré.

**DINOv2** (Meta AI, 2023) est un Vision Transformer entraîné par auto-supervision sur 142 millions d'images. Le choix d'un modèle auto-supervisé est fondamental ici : nous n'avons **aucune étiquette** sur nos images (pas d'annotation manuelle « ceci est une éruption »). DINOv2 apprend des représentations visuelles riches sans supervision, ce qui le rend directement applicable à la détection d'anomalies.

**Stable Diffusion 1.5** (SD 1.5) a été retenu sur SD 2.1 ou SDXL pour des raisons pratiques : il est plus léger (tourne sur 8 Go de VRAM), mieux supporté par la communauté, et compatible avec les poids LoRA pré-existants pour le domaine volcanique. SD 1.5 opère dans un espace latent de 64×64 correspondant à des images 512×512, ce qui est exactement notre résolution de preprocessing.

**LoRA (Low-Rank Adaptation, PEFT)** résout un problème de ressources : fine-tuner l'intégralité de SD 1.5 (~860 M de paramètres) nécessiterait des dizaines de GPU-heures. LoRA n'entraîne que quelques millions de paramètres supplémentaires (adaptateurs de faible rang) insérés dans les couches d'attention, ce qui rend le fine-tuning faisable sur un MacBook ou un Colab gratuit.

**scikit-learn** pour le classifieur RandomForest : les méthodes ensemblistes sur features tabulaires sont interprétables (importances des variables), rapides à entraîner, et robustes sur des datasets de quelques milliers d'images annotées — exactement ce que nous pouvons réalistement annoter manuellement.

---

## 2. Architecture technique

```
merapi_anomaly/
├── app/
│   └── streamlit_app.py         ← Application principale (ce fichier)
├── src/
│   ├── utils.py                 ← Fonctions utilitaires (load_config, parse_filename…)
│   ├── preprocessing.py         ← MerapiPreprocessor (redimensionnement, normalisation)
│   ├── generator.py             ← ImageGenerator (Stable Diffusion, DALL·E 3)
│   ├── physics_prompts.py       ← Construction de prompts physiques riches
│   ├── scraper.py               ← Scraping des images depuis wwwobs
│   ├── indexer.py               ← Construction et gestion de l'index CSV
│   ├── quality_filter.py        ← Classification qualité (usable/dark/cloudy/corrupted)
│   ├── baselines.py             ← Scores d'anomalie baseline (pixel-level)
│   ├── features/
│   │   ├── physical_features.py ← Extraction de features physiques (brightness, entropy…)
│   │   └── attention_maps.py    ← Cartes d'attention DINOv2
│   ├── models/
│   │   ├── patchcore_detector.py← Détecteur DINOv2 + PatchCore
│   │   ├── volcano_classifier.py← Classifieur RandomForest + heuristique
│   │   └── diffusion_reconstructor.py ← Reconstruction img2img SD 1.5
│   ├── evaluation/
│   │   └── early_warning.py     ← Calcul de précurseurs + permutation test
│   └── analysis/
│       └── activity_heatmap.py  ← Heatmap spatiale et timeline d'activité
├── data/
│   ├── index/index.csv          ← Index principal (~76 820 lignes)
│   ├── raw/{année}/{mois}/      ← Images brutes JPG téléchargées
│   ├── processed/{année}/{mois}/← Images PNG prétraitées 512×512
│   └── events/merapi_events_2014_2018.csv ← Événements BPPTKG documentés
├── outputs/
│   ├── scores/
│   │   ├── patchcore_scores.csv ← Scores DINOv2+PatchCore (~76 821 lignes)
│   │   ├── baselines_{y}_{m}.csv← Scores baseline mensuels
│   │   └── evaluation_report.csv← AUC-PR baseline vs PatchCore
│   ├── models/
│   │   ├── volcano_clf.pkl      ← Modèle RandomForest entraîné
│   │   └── physical_features.csv← Features physiques de toutes les images
│   ├── lora_merapi_physics/     ← Poids LoRA fine-tunés
│   └── generated/               ← Images synthétiques générées
└── config/settings.yaml         ← Configuration centrale
```

### Chargement des données dans l'app (`load_index`)

Au démarrage, l'app exécute `load_index()` qui :

1. Lit `data/index/index.csv` (ou `index_demo.csv` en fallback).
2. Convertit les colonnes temporelles (`year`, `month`, `day`, `hour`, `minute`, `second`) en numérique.
3. Extrait la date depuis le nom de fichier si les champs sont manquants (`parse_filename_datetime`).
4. Enrichit la colonne `patchcore_score` depuis `outputs/scores/patchcore_scores.csv` pour les images qui n'ont pas encore de score dans l'index.
5. Initialise les colonnes optionnelles manquantes (`patchcore_score`, `quality_flag`, `is_night`) à `NaN` pour éviter les `KeyError`.
6. Filtre pour ne garder que les images de la caméra **Kalor** (filtre sur le nom de fichier).
7. Met le résultat en cache pendant 5 minutes (`@st.cache_data(ttl=300)`).

> **Pourquoi cette conception ?** L'index CSV est le seul fichier qui évolue au cours du pipeline (les scores PatchCore sont calculés séparément, dans un fichier dédié). Fusionner les deux à l'affichage — et non dans le pipeline — permet de les faire évoluer indépendamment : on peut recalculer les scores PatchCore sans toucher à l'index, et vice-versa. Le filtre Kalor est appliqué ici une seule fois plutôt que dans chaque page pour éviter la duplication de code. Le cache de 5 minutes est un compromis : assez court pour voir les mises à jour du pipeline, assez long pour ne pas re-lire 3 Mo de CSV à chaque clic.

---

## 3. Pipeline de données (en coulisses)

Le pipeline est orchestré depuis `run_full_pipeline.py` ou `run_v1_pipeline.py`. L'application Streamlit est le **front-end de visualisation** de ce pipeline.

> **Pourquoi un pipeline séparé de l'application ?** Il aurait été possible de tout faire dans l'app (scraping, prétraitement, scoring au clic). Ce choix aurait été désastreux : certaines étapes durent des heures (téléchargement de 7 000 images, calcul de 76 000 scores PatchCore). En séparant le pipeline hors-ligne de l'application de visualisation, on garantit que l'app reste réactive et que les calculs longs peuvent tourner en arrière-plan, sur un serveur ou un GPU distant, sans interrompre l'exploration.

```
Phase 1 ── Scraping + Indexation
            src/scraper.py → src/indexer.py
            └→ data/index/index.csv (~76 820 lignes)

Phase 2 ── Téléchargement des images
            src/scraper.py --step download
            └→ data/raw/{année}/{mois:02d}/*.jpg

Phase 3 ── Prétraitement + Classification qualité
            src/preprocessing.py (MerapiPreprocessor)
            src/quality_filter.py
            └→ data/processed/{année}/{mois:02d}/*.png
            └→ colonnes quality_flag, is_night, mean_brightness, variance dans index.csv

Phase 4 ── Scores baseline (pixel-level)
            src/baselines.py
            └→ outputs/scores/baselines_{année}_{mois:02d}.csv

Phase 5 ── DINOv2 + PatchCore (features sémantiques)
            src/models/patchcore_detector.py
            └→ outputs/scores/patchcore_scores.csv

Phase 6 ── Features physiques
            src/features/physical_features.py
            └→ outputs/models/physical_features.csv

Phase 7 ── Classification volcanique
            src/models/volcano_classifier.py
            └→ outputs/models/volcano_clf.pkl

Phase 8 ── Fine-tuning LoRA (optionnel)
            train_lora_physics.py
            └→ outputs/lora_merapi_physics/lora_merapi_physics_final/

Phase 9 ── Application Streamlit
            app/streamlit_app.py
```

### Logique des transitions entre phases

**Phase 1 → 2 (Indexation avant téléchargement)** : On indexe d'abord toutes les URLs disponibles sur le serveur OPGC *sans* télécharger les images. Cela permet d'avoir une vision complète du dataset (76 820 images sur 10 ans) avant de décider quelles années ou quels mois télécharger réellement. Le téléchargement sélectif évite de saturer le disque local (les 76 820 images brutes représenteraient plusieurs centaines de Go).

**Phase 2 → 3 (Téléchargement avant prétraitement)** : Le prétraitement ne peut s'appliquer qu'aux images effectivement présentes sur disque. La phase 3 lit les JPG bruts téléchargés en phase 2 pour les convertir en PNG 512×512 normalisés. La normalisation min-max par image est nécessaire pour homogénéiser les conditions d'éclairage très variables (lever de soleil, nuit, brume).

**Phase 3 → 4 (Qualité avant scoring)** : Il serait contre-productif de calculer un score d'anomalie sur une image noire (nuit) ou entièrement blanche (nuage épais) : ces images produiraient des faux positifs massifs. La classification qualité en phase 3 permet à la phase 4 (et aux phases suivantes) de travailler uniquement sur des images `usable`.

**Phase 4 → 5 (Baseline avant PatchCore)** : Les scores baseline (phase 4) sont des métriques pixel-level simples et rapides à calculer (MAD, SSIM, différence de luminosité). Ils servent de **ligne de base** pour évaluer le gain apporté par PatchCore (phase 5). Sans baseline, on ne peut pas quantifier si DINOv2 apporte réellement quelque chose (voir AUC-PR dans la page 9).

**Phase 5 → 6 (PatchCore avant features physiques)** : Les features physiques (entropie, ratio pixels brillants, texture) sont extraites dans un but différent des scores PatchCore : elles servent à *expliquer* et *classifier* les anomalies, pas seulement à les détecter. Elles dépendent des images prétraitées (phase 3) mais pas des scores (phases 4–5), donc elles pourraient être calculées en parallèle. L'ordre retenu est séquentiel par souci de simplicité.

**Phase 6 → 7 (Features avant classification)** : Le classifieur RandomForest (phase 7) prend en entrée les features physiques (phase 6) ET les scores PatchCore (phase 5). Il faut donc que les deux soient disponibles avant d'entraîner le classifieur.

**Phase 7 → 8 (Classification avant LoRA, optionnel)** : Le fine-tuning LoRA est indépendant des phases précédentes du point de vue des données (il utilise les images `usable` de phase 3). Toutefois, disposer de la classification (phase 7) permet de construire des datasets de fine-tuning ciblés : on peut générer des images synthétiques uniquement pour les classes sous-représentées (pyroclastique, lave nocturne).

**Schéma du flux de données dans l'app :**

```
index.csv ──────────────────────────────────────────────────────────────┐
patchcore_scores.csv ──── load_index() ──── df (DataFrame principal) ───┤
                                                                         ├──► Toutes les pages
data/raw/ + data/processed/ ───── find_image_path() ─── images PIL ────┘
```

---

## 4. Page 1 — 🏠 Accueil

### Rôle

Page d'entrée du tableau de bord. Elle donne une **vue synthétique immédiate** de l'état du dataset et du projet scientifique.

> **Pourquoi une page d'accueil avec des métriques de pipeline ?** Dans un projet de traitement massif de données, la première question que se pose un utilisateur n'est pas « où est l'anomalie » mais « les données sont-elles prêtes ? ». Cette page répond immédiatement : 79 570 images indexées, combien sont téléchargées ? Combien sont scôrées ? Cela évite de lancer une analyse sur un dataset incomplet. C'est aussi un outil de communication : montrer ces métriques à un directeur de stage ou lors d'une soutenance donne immédiatement la dimension du travail accompli.

### Étapes et flux

1. Chargement du DataFrame principal via `load_index()`.
2. Affichage de **5 métriques clés** dans une ligne de colonnes.
3. Tracé d'un histogramme « images par année ».
4. Tracé d'un camembert « répartition des flags de qualité ».
5. Présentation du périmètre scientifique en 3 colonnes (données / anomalies cibles / approches IA).

### Métriques affichées

| Métrique | Définition | Utilité |
|----------|-----------|---------|
| **Total indexées** | Nombre total de lignes dans `index.csv` | Donne le volume global du dataset |
| **Téléchargées** | Nombre d'images avec `downloaded == True` | Mesure la progression du scraping |
| **Années** | Plage `min(year)–max(year)` | Confirme la couverture temporelle 2014–2024 |
| **Scorées** | Nombre d'images avec `anomaly_score` non nul | Indique si la Phase 4 (baselines) a été exécutée |
| **Classifiées** | Nombre d'images avec `quality_flag` non nul | Indique si la Phase 3 (qualité) a été exécutée |

### Contribution au projet

Cette page sert de **tableau de bord de suivi de pipeline**. Elle permet de vérifier d'un coup d'œil quelles phases sont complètes et combien d'images sont exploitables scientifiquement.

> **Transition vers la page suivante** : une fois confirmé que des images sont disponibles et des scores calculés, l'étape naturelle est d'explorer le dataset méthodiquement — ce que fait la page Exploration.

---

## 5. Page 2 — 🔍 Exploration

### Rôle

Navigation chronologique interactive dans le dataset. Permet d'**explorer et d'exporter** les images mois par mois.

> **Pourquoi une page d'exploration distincte ?** Un dataset de 76 820 images couvrant 10 ans ne peut pas s'analyser en bloc. Il faut pouvoir naviguer par période, filtrer par qualité, et identifier des mois«anomaux» avant de lancer des calculs coûteux. Cette page remplit ce rôle de navigation éclairée. La heatmap heure×jour est particulièrement utile pour détecter les « trous » dans le dataset (pannes de caméra, téléchargements manquants), qui pourraient biaiser silencieusement les analyses temporelles.

### Étapes et flux

1. **Filtrage sidebar** : sélection de l'année, du mois, des flags qualité et du statut « téléchargée ».
2. Affichage d'une **table de données** triée par jour/heure (colonnes : `filename`, `day`, `hour`, `quality_flag`, `anomaly_score`, `downloaded`, `file_size_bytes`).
3. Construction d'un **pivot heure × jour** pour visualiser la couverture temporelle sous forme de heatmap.
4. Bouton **export CSV** de la sélection courante.

### Métriques affichées

| Métrique | Définition | Utilité |
|----------|-----------|---------|
| **Images** | Nombre d'images dans la sélection courante | Donne la densité du mois sélectionné |
| **Téléchargées** | Images avec `downloaded == True` dans la sélection | Indique si les images du mois sont disponibles |
| **Avec score** | Images avec `anomaly_score` non nul | Indique si les baselines ont été calculés sur ce mois |

### Heatmap heure × jour

- **Axe X** : jour du mois (1–31)
- **Axe Y** : heure UTC (0–23)
- **Couleur** : nombre d'images dans cette case

Elle révèle les **créneaux de prise de vue** (ex. : absence d'images la nuit si la caméra est programmée uniquement en journée) et les **jours manquants** (outages, conditions météo).

### Contribution au projet

Permet de vérifier la qualité de la couverture temporelle avant d'entraîner un modèle. Un dataset avec des lacunes importantes peut biaiser les résultats.

> **Transition vers la page suivante** : après avoir identifié une période intéressante (ex. : septembre 2020, effondrement du cratère), l'utilisateur passe naturellement à la page Anomalies pour voir quelles images de cette période ont reçu un score élevé.

---

## 6. Page 3 — 🚨 Anomalies

### Rôle

Page centrale de **détection et de visualisation des anomalies**. Elle présente les images les plus suspectes, leur distribution de scores et leur localisation temporelle.

> **Pourquoi trois niveaux de scores (baseline, anomaly_score, patchcore_score) ?** Les trois scores ne sont pas disponibles en même temps selon l'état du pipeline. Les baselines peuvent être calculés dès la phase 4 (quelques minutes), alors que les scores PatchCore nécessitent de faire tourner DINOv2 sur toutes les images (phase 5, plus longue). La priorité donnée aux baselines permet d'avoir une détection « bonne mais imparfaite » rapidement, avant d'avoir les scores sémantiques plus précis. C'est une conception pragmatique : l'outil est utilisable dès la phase 4, pas seulement après la phase 5.

### Étapes et flux

1. Filtrage sidebar (année / mois / qualité).
2. **Sélection automatique du score disponible** selon la priorité suivante :
   - `baselines_{year}_{month}.csv` (scores pixel-level pré-calculés) → `combined_score`
   - `anomaly_score` dans l'index → score brut
   - `patchcore_scores.csv` → `patchcore_score` (DINOv2)
3. Si aucun score n'est disponible → message d'information avec les commandes à lancer.
4. Calcul du **seuil dynamique** μ + 2σ sur la distribution des scores.
5. Affichage via 3 onglets :
   - **Distribution** : histogramme + évolution temporelle
   - **Top anomalies** : tableau + aperçu visuel des N images les plus anormales
   - **Heatmap** : score moyen par case heure × jour

### Métriques affichées

| Métrique | Définition | Calcul | Utilité |
|----------|-----------|--------|---------|
| **Images scorées** | Nombre d'images avec un score non nul | `df[score_col].notna().sum()` | Couverture de la détection |
| **Score moyen (μ)** | Moyenne du score sur la période | `mean(scores)` | Niveau d'activité "normal" de référence |
| **Score max** | Valeur maximale du score | `max(scores)` | Pic d'anomalie le plus fort du mois |
| **Anomalies (>2σ)** | Nombre d'images au-dessus du seuil μ+2σ | `count(score > μ + 2σ)` | Nombre d'événements suspects détectés |

### Seuil statistique μ + 2σ

Le seuil μ + 2σ est un **seuil adaptatif** basé sur la distribution des scores du mois considéré :

$$\text{seuil} = \mu + 2\sigma$$

- **μ** = score moyen du mois
- **σ** = écart-type du mois

Une image est considérée **anormale** si son score dépasse ce seuil. Sous hypothèse gaussienne, environ 2,3% des images sont naturellement au-dessus → les images au-dessus représentent des événements rares.
> **Pourquoi μ+2σ et non un seuil fixe absolu ?** La caractéristique fondamentale de la caméra Kalor est que la « normale » change selon les conditions météorologiques, les saisons, et l'état général du volcan. Un seuil fixe de type « score > 40 = anomalie » fonctionnerait en saison sèche mais générerait des faux positifs massifs en saison des pluies (où les images sont toutes plus variables). Le seuil adaptatif μ+2σ se recalibre automatiquement chaque mois, ce qui le rend robuste aux variations saisonnières. L'inconvénient est qu'il ne permet pas de comparer les mois entre eux — c'est l'objet de la page 9 (PatchCore) qui donne un score absolu comparable.
### Scores disponibles

| Score | Source | Nature |
|-------|--------|--------|
| `combined_score` | `baselines_{y}_{m}.csv` | Combinaison de métriques pixel (nuit, MAD, SSIM) |
| `anomaly_score` | `index.csv` | Score générique brut |
| `patchcore_score` | `patchcore_scores.csv` | Distance au coreset DINOv2 (sémantique) |

### Contribution au projet

C'est la page de **validation principale** de la chaîne de détection. Elle permet de confirmer visuellement que les images les plus scorées correspondent bien à des événements volcaniques réels.
> **Transition vers la page suivante** : la détection par score est aveugle à ce que l'image montre réellement. Pour confirmer qu'une image anomale correspond bien à un événement volcanique (et pas à un artefact de caméra), il faut la visualiser. C'est le rôle de la galerie temporelle.
---

## 7. Page 4 — 🖼️ Galerie temporelle

### Rôle

Navigateur image par image avec affichage côte à côte de la version **brute (raw JPG)** et de la version **prétraitée (PNG 512×512)**. Permet d'inspecter visuellement les images individuelles et leurs métadonnées.

> **Pourquoi afficher les deux versions (brute et traitée) ?** Le prétraitement (redimensionnement à 512×512 + normalisation min-max) peut parfois introduire des artefacts ou masquer des informations visuelles importantes (ex. : une couleur particulière visible dans le JPG brut mais écrasée après normalisation). Comparer les deux permet au volcanologue de détecter immédiatement si le preprocessing dénature l'information. C'est aussi un outil de débogage : si l'image traitée a l'air étrange, on sait où chercher le problème.

### Étapes et flux

1. Filtrage sidebar (année / mois / qualité / téléchargées).
2. Tri chronologique du DataFrame filtré (jour → heure → minute).
3. Slider de navigation 0…N-1.
4. Résolution du chemin réel de l'image via `find_image_path()` (cherche dans `data/raw/`, `data/processed/` et `local_path`).
5. Affichage en 2 colonnes :
   - Gauche : image brute (ou message d'erreur détaillé si absente)
   - Droite : version PNG prétraitée
6. Expandeur de métadonnées JSON (filename, URL, qualité, heure, etc.).
7. Bande de miniatures des 10 premières images de la sélection.

### Métadonnées affichées

| Champ | Description |
|-------|-------------|
| `filename` | Nom du fichier source |
| `url` | URL de téléchargement d'origine |
| `day`, `hour`, `minute`, `second` | Horodatage UTC de la prise de vue |
| `quality_flag` | Classification qualité (usable / dark / cloudy / corrupted) |
| `is_night` | Indicateur nuit (`True` si heure entre 18h et 6h ou luminosité < seuil) |
| `anomaly_score` | Score d'anomalie si calculé |
| `file_size_bytes` | Taille du fichier source en octets |

### Contribution au projet

Permet une **inspection qualitative manuelle** du dataset pour valider les annotations automatiques et identifier des cas limites (ex. : nuage ressemblant à un panache).

> **Transition vers la page suivante** : l'inspection visuelle rassure sur la qualité des données. L'étape suivante naturelle dans un projet de recherche est de se demander « peut-on générer des images synthétiques qui ressemblent à ces vraies images ? » — ce que propose la page Génération IA.

---

## 8. Page 5 — 🤖 Génération IA

### Rôle

Module de **génération d'images volcaniques synthétiques** via deux approches : text-to-image libre et génération conditionée par des paramètres physiques. Prototype de l'axe IA générative du projet.
> **Pourquoi la génération d'images dans un projet de surveillance volcanique ?** La détection d'anomalies souffre d'un déséquilibre de classes sévère : sur 76 820 images, les éruptions majeures ne représentent qu'une fraction infime. Entraîner un classifieur supervisé sur un tel déséquilibre produit un modèle qui prédit toujours « normal » et obtient quand même 98% de précision. La génération synthétique permet d'augmenter artificiellement les classes rares (pyroclastique, lave nocturne). De plus, Stable Diffusion fine-tuné sur images Kalor peut générer des scénarios hypothétiques (ex. : éruption de nuit par temps de brume) qui n'ont pas été photographiés mais pourraient survenir.
### Onglets

#### Onglet 1 — Texte → Image

**Flux :**

1. Détection du backend disponible (`diffusers` local ou `openai` DALL·E 3).
2. Détection du device (MPS / CUDA / CPU) pour adapter les recommandations.
3. Saisie d'un prompt texte libre ou sélection dans 7 exemples prédéfinis (calqués sur les objectifs scientifiques).
4. Configuration du modèle (SD 1.5, SD 2.1, SDXL, DALL·E 3 selon disponibilité).
5. Paramétrage : résolution (384×384 à 1024×1024), mode rapide/qualité (15 ou 30 steps), guidance scale, seed.
6. Génération via `gen.generate()` → sauvegarde automatique dans `outputs/generated/`.

**Paramètres clés :**

| Paramètre | Plage | Rôle |
|-----------|-------|------|
| **Guidance scale** | 3 – 15 | Fidélité au prompt (7–8 = équilibré, >10 = très dirigé) |
| **Steps** | 10 – 50 | Qualité de la débruitage (plus = meilleur mais plus lent) |
| **Strength** | — | Utilisé en img2img (voir page 11) |
| **Seed** | 0 – 2³¹ | Reproductibilité (même seed = même image) |

#### Onglet 2 — Paramètres physiques

**Flux :**

1. Sélection des paramètres physiques en 3 colonnes :
   - **Observation** : caméra (Kalor/Suki/Kali), moment de la journée, luminosité
   - **Activité volcanique** : intensité lave, viscosité, température apparente
   - **Environnement** : météo, type d'éruption, panache
2. Construction automatique d'un **prompt en langage naturel riche** via `build_rich_prompt()` (src/physics_prompts.py).
3. Génération avec guidance scale élevé (10.0) pour coller aux paramètres physiques.
4. Si LoRA volcanique détecté → chargé automatiquement.

**Paramètres physiques disponibles :**

| Paramètre | Valeurs possibles |
|-----------|-----------------|
| Caméra | Suki, Kalor, Kali |
| Moment | early_morning, midday, afternoon, dusk, night |
| Luminosité | daylight, bright_with_incandescence, incandescent_glow, dim_glow, dark |
| Intensité lave | none, low, moderate, high, very_high |
| Viscosité | low, medium, high |
| Température | low, moderate, high, extreme |
| Météo | clear, overcast, hazy, clear_night |
| Type éruption | none, effusive, explosive, phreatic |
| Panache | none, low, medium, high |

#### Onglet 3 — Comparaison VAE vs Diffusion

Explication qualitative des deux approches et recommandation de **l'approche hybride** :
- **VAE** → représentation paramétrique (paramètres physiques → espace latent)
- **Modèle de diffusion** → génération haute qualité (espace latent → image réaliste)

> **Pourquoi l'approche hybride VAE+Diffusion est-elle recommandée ?** Un VAE seul génère des images floues (biais vers la moyenne car il minimise la log-vraisemblance). Un modèle de diffusion seul n'a pas de structure explicite dans son espace latent (on ne peut pas régler paramètres physiques à la main). L'approche hybride est le meilleur des deux mondes : le VAE encode des paramètres physiques interprétables dans un vecteur latent, et le modèle de diffusion décode ce vecteur en image photoristique. Cette architecture est proche de ce que font les modèles « conditioned on » dans la littérature (ex. : ControlNet pour SD).

### Contribution au projet

La génération d'images synthétiques sert à :
1. **Augmenter le dataset** d'entraînement pour les classes rares (pyroclastique, coulée de nuit).
2. **Communiquer** des scénarios volcaniques sans avoir besoin d'images réelles.
3. **Tester** la chaîne d'anomalie sur des images synthétiques contrôlées.

> **Transition vers la page suivante** : la génération nous a familiarisés avec les paramètres physiques de la lave (viscosité, température, pente). Pour ancrer cette intuition physique dans un modèle quantitatif, la simulation d'écoulements de la page suivante adopte les mêmes paramètres et les traduit en trajectoires.

---

## 9. Page 6 — 🌋 Simulation d'écoulements

### Rôle

Simulation **2D simplifiée** d'un écoulement lavique gravitaire. Outil pédagogique pour visualiser l'effet des paramètres physiques sur la dynamique d'une coulée.

> **Pourquoi une simulation 2D simplifiée et non un modèle 3D complet (type FLOWGO ou MAGFLOW) ?** Les modèles 3D de dynamique des fluides (CFD) pour la lave nécessitent des données topoégraphiques précises (MNT LiDAR), des heures de calcul, et une expertise en physique des fluides hors du périmètre de ce stage. La simulation 2D simplifiée basée sur la formule de lubrification est un compromis : elle est physiquement fondée (même famille d'équations que FLOWGO), instantanée, interactive, et suffisante pour illustrer les concepts. Elle vise à développer l'intuition physique du volcanologue plutôt qu'à prédire la traçabilité réelle d'une coulée.

### Étapes et flux

1. Saisie des paramètres physiques (pente, viscosité, volume, durée, nombre de particules).
2. Calcul de la **vitesse de front** via la formule de lubrication :

$$v_{front} = \frac{\rho \cdot g \cdot \sin(\alpha) \cdot h^2}{3\mu}$$

Où :
- $\rho$ = densité du magma (~2500 kg/m³)
- $g$ = 9.81 m/s²
- $\alpha$ = angle de pente
- $h$ = épaisseur estimée de la coulée
- $\mu$ = viscosité dynamique (Pa·s)

3. Génération de N particules simulant l'écoulement (déviation latérale aléatoire pondérée par le volume et la viscosité).
4. Attribution d'une **température** décroissante le long de la coulée (1050°C → 600°C).
5. Affichage en 2 sous-graphes :
   - **Vue de dessus** : scatter coloré par température (palette `hot`)
   - **Profil en coupe** : section transversale avec terrain et coulée

### Paramètres de simulation

| Paramètre | Plage | Effet sur la simulation |
|-----------|-------|------------------------|
| **Pente (°)** | 5 – 50° | Plus la pente est forte, plus la coulée est rapide et longue |
| **Viscosité (Pa·s)** | 100 – 100 000 | Haute viscosité → coulée lente, plus étroite |
| **Volume (×1000 m³)** | 1 – 500 | Influence la largeur et la distance parcourue |
| **Durée (h)** | 1 – 48 | Contrôle la distance maximale (v × t) |
| **Particules** | 50 – 500 | Résolution de la visualisation |

### Contribution au projet

Cette simulation complète le projet en donnant un **contexte physique** aux images de surveillance. Elle illustre pourquoi certaines zones de l'image sont plus actives (corrélation avec la pente du flanc observé par la caméra Kalor).

> **Transition vers la page suivante** : les simulations et visualisations des pages précédentes portent toutes sur des sous-ensembles du dataset. La page Statistiques donne une vision globale et quantitative de l'ensemble des données, indispensable pour justifier les choix méthodologiques dans un rapport.

---

## 10. Page 7 — 📈 Statistiques

### Rôle

**Analyse exploratoire (EDA)** complète du dataset. Visualisation des distributions temporelles, de la qualité, des tailles de fichiers et de la progression du pipeline.

> **Pourquoi faire de l'EDA aussi tardivement dans le pipeline (phase 7 sur 9) ?** En réalité, l'EDA devrait être faite en premier, mais dans l'application, cette page apparaît après les pages d'exploration car elle est plus technique et moins accessible à un utilisateur non-technicien. Dans la logique scientifique du pipeline en coulisses, l'EDA est bien la première chose à faire : vérifier que la distribution temporelle est suffisamment uniforme, que la proportion de nuits n'est pas trop élevée (ce qui biaiserait les scores d'anomalie), et que les années post-2020 ne contiennent pas de rupture de protocole (changement de caméra, résolution différente).

### Onglets

#### Onglet Temporal

- Histogramme mensuel de 2014 à 2024 (granularité : mois)
- Distribution horaire UTC (0–23h)

Ces graphiques permettent d'identifier :
- Les **périodes de maintenance** (créneaux sans images)
- Les **plages horaires de prise de vue** (caméra active en journée uniquement ?)
- La **densité d'images** selon l'année

#### Onglet Qualité

| Visualisation | Description |
|--------------|-------------|
| Histogramme global | Compte par `quality_flag` (usable / dark / cloudy / corrupted) |
| Barres empilées par année | Évolution de la qualité au fil du temps |

**Flags de qualité :**

| Flag | Critère de détection |
|------|---------------------|
| `usable` | Image téléchargée, non sombre, non nuageuse, non corrompue |
| `dark` | Luminosité moyenne < seuil (`night_brightness_threshold` = 30 dans settings.yaml) |
| `cloudy` | Variance < seuil (`cloud_variance_threshold` = 50.0) → image homogène = nuage |
| `corrupted` | Erreur PIL à l'ouverture, fichier vide, ou taille anormalement petite |

#### Onglet Fichiers

- Histogramme des tailles en Ko (distribution typiquement bimodale : jour vs nuit)
- Métriques : taille totale (Mo), taille moyenne (Ko), taille max (Ko)

#### Onglet Progression

Barre de progression pour chaque phase du pipeline :

| Phase | Source | Indicateur |
|-------|--------|-----------|
| Indexation | `len(df)` | Total des images référencées |
| Téléchargement | `downloaded == True` | Images disponibles sur disque |
| Classification qualité | `quality_flag` non nul | Images prétraitées et évaluées |
| Scoring anomalie | `anomaly_score` non nul | Images avec score baseline |

### Contribution au projet

Essentielle pour **justifier les choix méthodologiques** dans le rapport : combien d'images sont exploitables ? Y a-t-il un biais temporel ? Quelle est la proportion de nuits (perturbant les modèles de détection) ?

> **Transition vers la page suivante** : une fois que les statistiques globales ont été consultées et que l'on est confiant dans la qualité du dataset, la page « À propos » offre une documentation de référence consolidant tous les paramètres du projet pour une consultation rapide.

---

## 11. Page 8 — 📖 À propos

### Rôle

Documentation intégrée du projet, accessible sans quitter l'application. Organisée en 4 onglets.

> **Pourquoi intégrer la documentation dans l'app elle-même ?** Une documentation externe (fichier Markdown, wiki, PDF) est souvent désynchronisée du code. En la tenant dans l'application, elle reste accessible à tout moment lors d'une démonstration ou d'une session de travail, sans avoir à chercher un onglet de navigateur séparé. C'est aussi un moyen de guider un nouveau collaborateur qui découvre l'outil : il peut lire les explications et observer immédiatement le résultat dans les autres pages.

| Onglet | Contenu |
|--------|---------|
| **Projet** | Objectifs, contexte scientifique LMV, données, technologies |
| **Pipeline** | Schéma des 9 phases avec commandes bash |
| **Sections de l'app** | Tableau récapitulatif des 12 pages avec données requises |
| **Guide fine-tuning** | Paramètres LoRA recommandés, pièges à éviter, critères d'évaluation |

### Paramètres LoRA recommandés (résumé)

| Paramètre | Valeur | Raison |
|-----------|--------|--------|
| Learning rate | 1e-5 à 5e-6 | Évite le catastrophic forgetting (SD 1.5 pré-entraîné) |
| Epochs | 50 – 200 | Avec early stopping sur la loss de validation |
| Batch size | 2 – 4 | Contrainte VRAM (MPS 8–16 Go, CUDA 8+ Go) |
| Résolution | 512×512 | Cohérent avec le preprocessing |
| Mixed precision | fp32 (MPS) / fp16 (CUDA) | MPS ne supporte pas fp16 stable |

> **Catastrophic forgetting** : c'est le phénomène par lequel un modèle pré-entraîné oublie ses connaissances générales en apprenant le domaine volcanique. Avec un learning rate trop élevé (ex. : 1e-4), les poids de SD 1.5 sont trop perturbés et le modèle perd la capacité de générer autre chose que des volcans. LoRA contourne partiellement ce problème en ajoutant des poids légers sans modifier les poids pré-entraînés, mais un learning rate raisonnable reste nécessaire.

> **Transition vers la page suivante** : après les pages de visualisation et d'outils, les pages qui suivent forment le cœur analytique du projet. La page DINOv2+PatchCore est la pièce maîttresse de la détection d'anomalies sémantique.

---

## 12. Page 9 — 🔬 DINOv2 + PatchCore

### Rôle

Page principale de **détection d'anomalies sémantique sans supervision**. DINOv2 extrait des features visuelles riches, PatchCore mesure la distance par rapport à un coreset d'images normales.

> **Pourquoi DINOv2 plutôt qu'un ResNet ou un VGG classique ?** Les réseaux convolutifs supervisés comme ResNet ont besoin d'étiquettes de classification pour entraînement. Or, nos images n'ont pas d'étiquettes. DINOv2 est entraîné par auto-supervision : il apprend seul à représenter les images de façon semantiquement cohérente, sans label. De plus, étant un Vision Transformer, il traite l'image sous forme de patches 14×14 px et modélise les dépendances spatiales longue portée, ce qui est essentiel pour détecter des événements qui occupent une région spécifique de l'image (ex. : coulée de lave sur le flanc droit).

> **Pourquoi PatchCore plutôt qu'un auto-encodeur d'anomalie ?** Un auto-encodeur (AE) standard entraîné à reconstruire des images normales détecte les anomalies via l'erreur de reconstruction. Il a deux limites connues : (1) les AE peuvent parfois « apprendre à reconstruire » même des images anormales si le bottleneck est trop grand ; (2) il faut entraîner le modèle, ce qui nécessite plusieurs jours si les images sont nombreuses. PatchCore est un **méthode à mémoire** : il ne fait que mémoriser un sous-ensemble compact de patches normaux (coreset), puis calcule des distances au moment du scoring. Pas d'entraînement, déploiement immédiat, et performances state-of-the-art sur les benchmarks d'anomalie industriels (MVTec). C'est idéal pour un contexte de recherche où les ressources sont limitées.

### Principe de DINOv2 + PatchCore

```
Image (224×224) ──► DINOv2-small (ViT) ──► Features patches (197 × 384)
                                                        │
                                         Coreset (images normales)
                                                        │
                                         Distance min patch-à-patch
                                                        │
                                         Agrégation max → patchcore_score
```

- **DINOv2-small** : Vision Transformer entraîné en auto-supervision (DINO v2, Meta AI, 2023). Produit des features sémantiques stables (pas sensibles aux variations d'éclairage).
- **PatchCore** : mémorise un sous-ensemble représentatif (coreset) des patches de la distribution normale. Pour une nouvelle image, le score est la **distance maximale** entre ses patches et le coreset.

### Étapes et flux

1. Chargement de `patchcore_scores.csv` (ou calcul à la volée si absent).
2. **4 métriques clés** + timeline mensuelle avec marqueur éruption mai 2018.
3. Affichage des N images les plus anormales avec prévisualisation.
4. **Carte d'attention DINOv2** sur une image sélectionnée (calcul à la demande).
5. **Rapport d'évaluation** (AUC-PR) si `evaluation_report.csv` est disponible.

### Métriques affichées

| Métrique | Définition | Calcul | Utilité |
|----------|-----------|--------|---------|
| **Images scorées** | Nombre d'images avec `patchcore_score` non nul | Comptage | Couverture de la détection sémantique |
| **Score médian** | Médiane de tous les scores | `median(patchcore_scores)` | Niveau d'activité typique (robuste aux outliers) |
| **Score max** | Maximum absolu | `max(patchcore_scores)` | Pic d'anomalie le plus fort sur toute la période |
| **Images > P90** | Nombre au-dessus du 90e percentile | `count(score > quantile(0.90))` | Taille de la queue supérieure (événements rares) |
| **AUC-PR Baseline** | Aire sous la courbe Précision-Rappel (baseline pixel) | Calculé sur `evaluation_report.csv` | Performance de la méthode de référence |
| **AUC-PR PatchCore** | Aire sous la courbe Précision-Rappel (PatchCore) | Calculé sur `evaluation_report.csv` | Performance de la méthode sémantique |
| **Amélioration Δ AUC-PR** | `AUC-PR PatchCore − AUC-PR Baseline` | Différence | Gain apporté par DINOv2 vs méthode pixel |

### AUC-PR (Aire sous la courbe Précision-Rappel)

L'AUC-PR est préférable à l'AUC-ROC sur des datasets déséquilibrés (comme le nôtre, où les anomalies représentent ~2% des images) :

- **Précision** = TP / (TP + FP) → parmi les anomalies détectées, combien sont vraies ?
- **Rappel** = TP / (TP + FN) → parmi toutes les vraies anomalies, combien sont détectées ?
- **AUC-PR = 1.0** = détection parfaite
- **AUC-PR = proportion d'anomalies** = modèle aléatoire

> **Pourquoi l'AUC-PR est plus pertinente que l'AUC-ROC ici ?** L'AUC-ROC mesure la capacité du modèle à séparer les positifs des négatifs, mais elle est trop optimiste sur des datasets déséquilibrés : un modèle qui prédit toujours « normal » peut avoir une AUC-ROC > 0.95. L'AUC-PR, en focalisant sur les positifs (anomalies), révèle vraiment si le modèle sait détecter les événements rares. Ici, le modèle aléatoire aurait une AUC-PR ≈ 0.02 (2% d'anomalies), donc tout score significativement au-dessus démontre une valeur ajoutée.

### Carte d'attention DINOv2

Visualisation **où le modèle focalise son attention** sur l'image :
- Rouge intense = zone la plus discriminante (patch très éloigné du coreset normal)
- Bleu = zone normale
- Superposée à l'image originale en niveaux de gris (alpha 55%)

### Contribution au projet

DINOv2 + PatchCore est le **cœur de la détection d'anomalies**. Les scores PatchCore sont utilisés en aval par les pages Early Warning, Analyse avancée et Analyse volcanique avancée.

> **Transition vers la page suivante** : PatchCore nous donne un score par image, mais ce score est un nombre sans contexte temporel. La question scientifique suivante est : ces scores augmentent-ils *avant* les éruptions connues ? C'est ce que teste la page Early Warning.

---

## 13. Page 10 — ⚡ Early Warning

### Rôle

Validation scientifique du **signal précurseur** : est-ce que les scores d'anomalie augmentent significativement **avant** les événements volcaniques documentés par le BPPTKG ?

> **Pourquoi c'est la question centrale du projet ?** La surveillance volcanique en temps réel ne sert à rien si elle détecte l'éruption *pendant* l'éruption. La valeur ajoutée d'un système automatique est de détecter des signaux anormaux *avant* l'événement, pour permettre une alerte précoce. Cette page teste directement cette hypothèse : sur les événements connus (catalog BPPTKG 2014–2018), est-ce que les scores PatchCore étaient effectivement plus élevés dans les jours qui ont précédé l'événement ? Si oui, et si le test de permutation confirme que ce n'est pas dû au hasard, alors l'approche a une valeur prédictive réelle.

### Principe

```
Événement volcanique (ex. éruption 12/05/2018)
         │
         ├── J-1 : score moyen des images de la veille
         ├── J-3 : score moyen des 3 jours avant
         └── J-7 : score moyen des 7 jours avant

Comparer ces scores au "background" (activité normale des semaines précédentes)

Ratio = score_précurseur / score_background
Si Ratio > 1.5 → signal précurseur détecté
```

### Étapes et flux

1. Chargement des événements depuis `data/events/merapi_events_2014_2018.csv`.
2. Si `early_warning_precursors.csv` existe → chargement direct des résultats pré-calculés.
3. Sinon → calcul via `EarlyWarningAnalyzer.compute_precursor_scores()`.
4. Tableau des scores précurseurs par événement (coloré selon le ratio).
5. Graphique ratio par horizon de temps (J-1, J-3, J-7) pour chaque événement.
6. **Permutation test** (1000 itérations) pour valider la significativité statistique.
7. Affichage de la timeline complète si la figure PNG est disponible.

### Métriques affichées

| Métrique | Définition | Calcul | Utilité |
|----------|-----------|--------|---------|
| **mean_score** | Score moyen dans la fenêtre précurseur (J-N avant l'événement) | `mean(scores[event_date - N days : event_date])` | Niveau d'activité détecté avant l'événement |
| **background_score** | Score moyen de la période de référence (activité normale) | `mean(scores[référence])` | Niveau de base pour comparaison |
| **ratio** | Rapport précurseur / background | `mean_score / background_score` | >1.5 = signal fort, >1.2 = signal modéré |
| **lead_days** | Horizon de prédiction testé | 1, 3 ou 7 jours avant | Mesure la précocité du signal |
| **n_images** | Nombre d'images dans la fenêtre | Comptage | Fiabilité statistique du score moyen |
| **p-value** | Probabilité d'obtenir un ratio aussi élevé par hasard | Test de permutation (1000 it.) | Significativité statistique du signal |

### Test de permutation

Le test de permutation est une méthode non-paramétrique :
1. On calcule le ratio observé pour tous les événements réels.
2. On permute aléatoirement les dates (1000 fois) et on recalcule le ratio.
3. La **p-value** = proportion de permutations avec un ratio ≥ ratio observé.
4. Si `p < 0.05` → le signal précurseur est statistiquement significatif.

> **Pourquoi un test de permutation plutôt qu'un t-test classique ?** Le t-test suppose que la distribution des scores suit une loi normale. Ce n'est pas garanti pour nos scores PatchCore, qui peuvent avoir une queue épaisse (quelques images très anormales). Le test de permutation ne fait aucune hypothèse sur la distribution : il construit empiriquement la distribution nulle en mélangeant les données. C'est plus robuste et plus honnête scientifiquement. Avec 1000 permutations, on obtient une p-value avec une précision de ±0.003.

### Coloration du tableau précurseurs

| Couleur | Condition | Signification |
|---------|-----------|--------------|
| 🔴 Rouge | ratio > 1.5 | Signal précurseur fort |
| 🟠 Orange | ratio > 1.2 | Signal précurseur modéré |
| Blanc | ratio ≤ 1.2 | Pas de signal détecté |

### Contribution au projet

C'est la **validation scientifique principale** du projet : si les scores PatchCore sont statistiquement plus élevés avant les événements, cela démontre la valeur de l'approche pour la **surveillance en temps réel**.

> **Transition vers la page suivante** : l'Early Warning donne un signal global (« l'activité globale augmente avant l'éruption ») mais ne dit pas *où* dans l'image se trouve l'anomalie. La page Analyse avancée répond à cette question en reconstruisant l'image et en comparant pixel à pixel.

---

## 14. Page 11 — 🧪 Analyse avancée

### Rôle

Reconstruction d'images par diffusion img2img pour **localiser spatialement les anomalies** dans une image individuelle.

> **Pourquoi la reconstruction par diffusion pour localiser les anomalies ?** PatchCore donne un score global pour l'image entière. Mais pour un volcanologue, savoir « cette image est anormale à score 52 » est moins utile que « le flanc sud-est de l'image est anormalement brillant ». La reconstruction img2img utilise Stable Diffusion pour « corriger » l'image vers un état normal : le modèle génère ce que l'image *aurait dû* ressembler s'il n'y avait pas d'événement. La différence |original − reconstruit| localise précisément les zones anormales. C'est un paradigme connu sous le nom de « reconstruction-based anomaly detection », plus puissant que les approches pixel-level naïves.

### Onglets

#### Onglet 1 — Reconstruction img2img

**Principe :**

```
Image réelle (brute)
        │
  Encodage VAE → espace latent z
        │
  Ajout de bruit (σ × strength, 0.1–0.7)
        │
  Débruitage SD 1.5 guidé par le prompt "volcan normal"
        │
  Décodage VAE → image reconstruite
        │
  |original − reconstruit| → carte d'anomalie pixel-à-pixel
```

**Flux :**
1. Sélection de l'image source via 3 modes : Top anomalies PatchCore / Sélection manuelle / Upload.
2. Réglage du paramètre `strength` (0.10–0.70).
3. Option LoRA volcanique (améliore le réalisme de la reconstruction).
4. Lancement de la reconstruction via `DiffusionReconstructor.reconstruct()`.
5. Affichage côte-à-côte : original / reconstruit / carte d'anomalie colorisée.
6. Sauvegarde optionnelle dans `outputs/generated/`.

**Backends de reconstruction :**

| Backend | Condition | Résultat |
|---------|-----------|---------|
| SD 1.5 (diffusers) | `diffusers` installé | Reconstruction sémantique complète (~10–40s) |
| Fallback (gaussien) | `diffusers` absent | Flou gaussien simple (~instantané, indicatif) |
> **Pourquoi un fallback gaussien ?** L'objectif est que l'application reste utilisable même sur un poste sans GPU ou sans la bibliothèque `diffusers` installée. Le flou gaussien ne donne pas de localisation fiable, mais il permet de comprendre l'interface et de voir la carte de différence (toujours informative pour des anomalies très localisées). C'est un choix de dégradation gracieuse (graceful degradation), principe général de bonne conception logicielle.
#### Métriques de reconstruction

| Métrique | Définition | Calcul | Utilité |
|----------|-----------|--------|---------|
| **Score reconstruction** | Score d'anomalie de la carte de différence | `mean(|original - reconstruit|)` normalisé | Quantifie l'anomalie globale de l'image |
| **Diff. max (%)** | Différence pixel maximale | `max(diff_map) × 100` | Localise le pixel le plus anormal |
| **Diff. P95 (%)** | 95e percentile de la différence | `percentile(diff_map, 95) × 100` | Seuil robuste (ignore les outliers extrêmes) |

#### Paramètre `strength`

| Valeur | Effet |
|--------|-------|
| 0.10 – 0.20 | Image quasi-identique à l'original → anomalie = bruit |
| 0.30 – 0.40 | Équilibre : le modèle "corrige" les anomalies réelles vers le normal |
| 0.50 – 0.70 | Forte modification → comparaison à une image très différente |

**Valeur recommandée : 0.35** pour la détection d'anomalies volcaniques.
> **Pourquoi 0.35 précisément ?** C'est un paramètre critique qui résulte d'un compromis empirique. Avec `strength < 0.2`, le modèle ne change presque rien à l'image (la différence est du bruit). Avec `strength > 0.5`, le modèle change tellement l'image que la différence ne reflète plus l'anomalie mais les caprices génératifs du modèle. À 0.35, on observe expérimentalement que le modèle « répare » les régions anormales (les ramenant vers l'apparence normale d'un volcan au repos) mais laisse les régions normales intactes. Ce seuil peut varier selon le type d'image et devrait idéalement être validé sur un ensemble annoté.
#### Onglet 2 — Comparaison scores

- Distribution des scores PatchCore (histogramme avec P90 et P95)
- Score moyen ± σ par année (barres d'erreur)

#### Onglet 3 — Simulation rapide

Version allégée de la simulation d'écoulements (sans onglet dédié).

### Contribution au projet

La reconstruction diffusion permet de **localiser spatialement les anomalies** (quelle zone de l'image est anormale ?) là où PatchCore donne seulement un score global. Combinés, les deux approches fournissent une détection plus complète.

> **Transition vers la page suivante** : les analyses des pages 9–11 répondent à « y a-t-il une anomalie et où est-elle ? ». La dernière page étend cela à une question plus globale : « que se passe-t-il sur le volcan, comment le classer, et depuis quand cette activité persiste-t-elle ? »

---

## 15. Page 12 — 🌋 Analyse volcanique avancée

### Rôle

Classification automatique des images, heatmap d'activité spatiale et timeline d'anomalies. Synthèse de toutes les analyses en une vue intégrée.

> **Pourquoi cette page est-elle la dernière ?** Elle mobilise les résultats de toutes les phases précédentes (scores PatchCore de la phase 5, features physiques de la phase 6, classifieur de la phase 7) et les présente sous forme synthétique. C'est la page qui répond à la question de fond du volcanologue : « sur 10 ans d'images, comment évoluait l'activité du Merapi, où étaient les zones les plus actives, et quelles types d'événements ont été observés ? »

### Onglets

#### Onglet 1 — 🏷️ Classification

**Flux :**
1. Fusion du DataFrame principal avec les features physiques (`physical_features.csv`).
2. Chargement du classifieur (`volcano_clf.pkl` si disponible, sinon heuristique).
3. Classification manuelle déclenchée par bouton (évite de traiter 76k images à chaque rechargement).
4. Affichage : métriques par classe, camembert, histogramme par année, table des pyroclastiques.
5. Importances des features (RandomForest) si modèle disponible.

**Classifieur heuristique (règles calibrées) :**

| Condition | Classe détectée |
|-----------|----------------|
| `patchcore_score > 46.5` (top 2%) | **pyroclastique** |
| `is_night == True` et `bright_pixel_ratio > 2%` | **lave** (incandescence nocturne) |
| `entropy < 3.5` et `pixdiff < 0.03` | **nuage** (scène homogène) |
| Sinon | **normal** |

> **Pourquoi un classifieur heuristique en parallèle du RandomForest ?** Le RandomForest nécessite des données annotées pour l'entraînement, qui ne sont pas encore disponibles en quantité suffisante. L'heuristique permet d'avoir une classification « bonne et immédiate » basée sur des règles expertes (à savoir, les seuils ont été calibrés à partir de l'inspection manuelle de plusieurs centaines d'images). Une fois que suffisamment d'images seront annotées, le RandomForest prendra le relais et la comparaison heuristique/modèle (onglet 4) permettra de mesurer le gain. Le seuil `patchcore_score > 46.5` correspond au **98e percentile** de la distribution des scores, soit la définition directe des 2% d'images les plus anormales.

**Classes de sortie :**

| Classe | Description visuelle |
|--------|---------------------|
| `pyroclastique` | Nuée ardente, dépôt de cendres, écoulement rapide |
| `lave` | Coulée de lave incandescente (visible la nuit) |
| `nuage` | Couverture nuageuse cachant le cratère |
| `normal` | Activité de fond normale |

**Métriques affichées :**

| Métrique | Définition |
|----------|-----------|
| **Pyroclastiques** | Nombre d'images classées pyroclastique |
| **Coulées lave** | Nombre d'images classées lave |
| **Nuages** | Nombre d'images classées nuage |
| **Normaux** | Nombre d'images classées normal |

#### Onglet 2 — 🗺️ Zones actives (heatmap spatiale)

**Principe :**
- Grille 16×16 simulant les 256 patches d'une image traitée par DINOv2-small (images 224×224, patch_size=14).
- Chaque cellule représente une zone spatiale du volcan (16 lignes N→S × 16 colonnes E→W).
- La valeur = score d'activité agrégé (mean ou max) de tous les patches correspondants.

> **Comment lire cette heatmap ?** Une zone rouge persistante dans la grille 16×16 signifie que cette région spatiale du volcan est systématiquement plus anormale que le reste de l'image, sur l'ensemble de la période analysée. Pour la caméra Kalor du Merapi, on s'attend à voir une activité élevée dans la zone correspondant au cratère et au flanc sud (zone d'effusion préférentielle). Si l'activité apparait dans une zone inattendue (ex. : flanc nord), cela peut indiquer une nouvelle voie d'écoulement ou un artefact (réflexion solaire, arbre). La grille 16×16 correspond à la résolution native des patches DINOv2 sur une image 224×224 (224 / 14 = 16 patches par côté).

**Métriques :**

| Métrique | Définition | Utilité |
|----------|-----------|---------|
| Score d'activité normalisé | Score PatchCore de la cellule normalisé [0,1] | Localise les zones les plus actives |
| **Clusters actifs** | Zones contiguës avec score > seuil et taille ≥ min_size | Identifie les régions du volcan anormalement actives |

**Colonnes du tableau clusters :**
- `row`, `col` : position dans la grille 16×16
- `score` : score d'activité moyen du cluster

#### Onglet 3 — 📅 Timeline d'activité

**Flux :**
1. Calcul du score moyen par période (jour / semaine / mois) via `timeline_activity()`.
2. Tracé de la courbe + zone ombrée + seuil μ+2σ.
3. Marquage des pics dépassant le seuil.
4. Export CSV de la timeline.

**Métriques :**

| Métrique | Définition |
|----------|-----------|
| **Périodes analysées** | Nombre de points temporels dans la timeline |
| **Score max** | Pic maximum de la série temporelle |
| **Pics >2σ** | Nombre de périodes dépassant le seuil μ+2σ |

#### Onglet 4 — 🔎 Comparaison heuristique vs modèle

Compare la classification heuristique et le modèle RandomForest sur la période filtrée :

| Métrique | Définition |
|----------|-----------|
| **Accord heuristique/modèle** | % d'images classées identiquement par les deux méthodes |
| **Désaccords** | Nombre d'images avec classification différente |

### Contribution au projet

Cette page fournit la **synthèse analytique complète** en un seul endroit : où le volcan est-il le plus actif (heatmap) ? Quand (timeline) ? Quel type d'événement (classification) ? Les deux méthodes sont-elles cohérentes (comparaison) ?

---

## 16. Récapitulatif des métriques

> **Comment utiliser ce tableau ?** Ce récapitulatif est conçu comme une référence rapide pour retrouver l'origine et la signification d'une métrique vue dans l'application. Chaque métrique est rattachée à une ou plusieurs pages (col. *Page(s)*) : si un chiffre vous surprend dans l'interface, reprenez la ligne correspondante ici pour comprendre comment il a été calculé.

### Tableau complet de toutes les métriques

| Métrique | Page(s) | Définition | Calcul |
|----------|---------|-----------|--------|
| `patchcore_score` | 3, 9, 10, 11, 12 | Distance sémantique au coreset normal (DINOv2) | `max patch-to-nearest-coreset distance` |
| `anomaly_score` | 1, 2, 3, 7, 12 | Score d'anomalie générique (baseline ou autre) | Variable selon la source |
| `combined_score` | 3 | Score combiné des baselines pixel-level | Combinaison pondérée night/MAD/SSIM |
| `night_score` | 3 | Score d'activité nocturne | Luminosité × ratio nuit |
| `mad_prev` | 3 | Écart absolu médian par rapport à l'image précédente | `MAD(image_t − image_t-1)` |
| `ssim_prev` | 3 | Similarité structurelle avec l'image précédente | SSIM (Structural Similarity Index) ∈ [0,1] |
| `quality_flag` | 1, 2, 5, 7, 10 | Classification qualité | Règles seuils sur luminosité/variance |
| `mean_brightness` | 7 | Luminosité moyenne de l'image | `mean(pixels)` ∈ [0,255] |
| `std_brightness` | 7 | Écart-type de la luminosité | `std(pixels)` |
| `variance` | 7 | Variance des pixels | `var(pixels)` |
| `is_night` | 4, 7 | Indicateur nuit | `hour ∈ [18,6]` ou `brightness < seuil` |
| `file_size_bytes` | 2, 7 | Taille du fichier source | Octet (header HTTP ou stat) |
| `ratio` (EW) | 10 | Ratio précurseur / background | `mean_precursor / mean_background` |
| `p-value` (EW) | 10 | Significativité statistique du précurseur | Test de permutation (1000 it.) |
| `AUC-PR` | 9 | Aire sous la courbe Précision-Rappel | Intégrale de la courbe P-R |
| Score reconstruction | 11 | Anomalie de la carte de différence img2img | `mean(|orig − recon|)` normalisé |
| `diff_max` | 11 | Différence pixel maximale | `max(|orig − recon|)` |
| `diff_P95` | 11 | 95e percentile de la différence | `percentile(|orig − recon|, 95)` |
| Score d'activité spatiale | 12 | Score PatchCore agrégé par cellule 16×16 | Agrégation mean ou max par patch |
| Accord heuristique/modèle | 12 | % d'accord entre les deux classifieurs | `mean(class_heuristic == class_model)` |

---

## 17. Conclusion et feuille de route

### Ce qui fonctionne actuellement

| Composant | Statut |
|-----------|--------|
| Indexation et scraping | ✅ ~76 820 images indexées |
| Téléchargement | ✅ ~7 000 images disponibles |
| Prétraitement + qualité | ✅ 6 696 usables / 176 sombres / 2 nuageuses / 247 corrompues |
| Scores PatchCore | ✅ ~76 821 scores calculés |
| App Streamlit (démarrage) | ✅ Corrigé (KeyError patchcore_score résolu) |
| Pages Accueil, Exploration, Galerie, Stats, À propos | ✅ Fonctionnelles |
| Génération IA (SD 1.5) | ✅ Avec backend diffusers ou fallback |
| Simulation écoulements | ✅ Modèle simplifié opérationnel |

### Prochaines étapes recommandées

1. **Valider visuellement les top anomalies** (page 3 et 9) sur la période septembre 2020.
2. **Entraîner le VolcanoClassifier** (RandomForest) sur un échantillon annoté (~500 images) pour remplacer l'heuristique.
3. **Calculer les features physiques** (`python run_v1_pipeline.py --step features`) pour alimenter la heatmap spatiale.
4. **Lancer le calcul Early Warning** (`python run_v1_pipeline.py --step early_warning`) sur les événements BPPTKG.
5. **Fine-tuner LoRA** sur les images `usable` de la caméra Kalor pour améliorer la génération.

### Commandes de lancement

```bash
# Lancer l'application
cd /chemin/vers/merapi_anomaly
USE_TF=0 USE_TORCH=1 /opt/anaconda3/bin/streamlit run app/streamlit_app.py

# Calculer les features physiques (page 12 heatmap)
python -m src.pipeline.run_pipeline --step features

# Calculer les précurseurs Early Warning (page 10)
python -m src.pipeline.run_pipeline --step early_warning

# Recalculer les scores PatchCore (page 9)
python -m src.pipeline.run_pipeline --step patchcore
```

---

*Document généré automatiquement à partir du code source de `app/streamlit_app.py` — Version du 27 avril 2026.*
