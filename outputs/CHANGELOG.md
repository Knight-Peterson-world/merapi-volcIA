# CHANGELOG — Pipeline PatchCore Merapi

## 2026-04-30

### Bugs corrigés

#### [BUG CRITIQUE] F1-macro incorrect dans `src/evaluation/metrics.py`

**Fichier** : `src/evaluation/metrics.py`, fonction `compute_f1_triage()`  
**Symptôme** : F1-macro = 1.0 rapporté alors que F1-météo = 0.0 et F1-volcanique = 1.0.  
**Cause** : `sklearn.f1_score(..., average='macro')` sans `labels=[0,1]` ne calcule
la moyenne que sur les classes présentes dans `y_true`. Si aucune image "cloudy" ne
survit au merge (2 images "cloudy" au total dans l'index), seule la classe 1 est
présente → macro = 1.0 au lieu de 0.5.  
**Correction** : ajout de `labels=[0, 1]` dans l'appel à `f1_score`.  
**Valeur corrigée** : F1-macro = 0.5  

#### [AMÉLIORATION] AUC-ROC ajoutée dans le rapport d'évaluation

**Fichier** : `src/evaluation/metrics.py`  
**Ajout** : nouvelle fonction `compute_auc_roc()` + champ `auc_roc_patchcore` dans
`EvaluationReport` + intégration dans `evaluate_patchcore()`.  
**Valeur mesurée** : AUC-ROC = 0.7112 (bonne discrimination malgré AUC-PR modérée).  

#### [CORRECTIF] `evaluation_report.csv` mis à jour

**Fichier** : `outputs/scores/evaluation_report.csv`  
**Changements** :
- `f1_macro_triage` : 1.0 → 0.5 (bug corrigé)
- `auc_roc_patchcore` : absent → 0.7112 (ajouté)  

### Nouveaux scripts

#### `outputs/check_score_inversion.py`

Script de diagnostic pour vérifier si les scores PatchCore doivent être inversés.  
Compare la distribution des scores entre images "dark" (anomalie) et "usable" (normal).  
**Résultat** : pas d'inversion nécessaire — les anomalies ont déjà des scores plus hauts
(mean dark = 45.96 > mean usable = 42.13).  
**AUC-PR original** = 0.1313, **AUC-PR inversé** = 0.0478 → original meilleur.

#### `outputs/figures/plot_patchcore_timeline_interactive.py`

Timeline interactive Plotly HTML avec :
- Courbe des scores moyens/max mensuels
- Zone de remplissage entre mean et max
- Seuil P90 = 49.975
- Annotations pour 3 événements (Éruption 2018, Crise sismique 2020, Alerte météo 2022)
- Tooltips enrichis (date, score, nb images)
- Export HTML autonome (~30 Ko avec CDN Plotly)
- Panneau secondaire : histogramme mensuel du nombre d'images

**Sortie** : `outputs/figures/patchcore_timeline_interactive.html`

#### `outputs/scores/evaluation_notes.md`

Documentation détaillée des limitations du rapport d'évaluation :
- Explication de l'AUC-PR modérée
- Correction du bug F1-macro
- Interprétation de l'effect size négatif
- Résultat du diagnostic d'inversion
- Recommandations futures (annotation manuelle, calibration du seuil, etc.)

### Fichiers modifiés (récapitulatif)

| Fichier | Type de modification |
|---|---|
| `src/evaluation/metrics.py` | Bug F1-macro + AUC-ROC ajouté |
| `outputs/scores/evaluation_report.csv` | Métriques corrigées |
| `outputs/figures/plot_patchcore_timeline.py` | Fix imports matplotlib.dates |
| `outputs/check_score_inversion.py` | **Nouveau** |
| `outputs/figures/plot_patchcore_timeline_interactive.py` | **Nouveau** |
| `outputs/scores/evaluation_notes.md` | **Nouveau** |

---

---

## 2026-05-01 (session 6 — suite)

### Nouveaux scripts

#### `src/evaluation/test_f1_macro.py`

5 tests unitaires pour la fonction `compute_f1_triage()` :
- `test_f1_macro_seule_classe_positive` — attend 0.5 (pas 1.0) quand seule classe 1 présente
- `test_f1_macro_deux_classes_parfait` — attend 1.0 avec classification parfaite
- `test_f1_macro_coherence_interne` — 4 sous-cas (FP purs, FN purs, mixte, vide)
- `test_f1_macro_seule_classe_negative` — attend 0.5 quand seule classe 0 présente
- `test_f1_macro_aucune_image` — attend nan avec liste vide

**Résultat** : 5/5 tests passent.  
**Lancement** : `/opt/anaconda3/bin/python src/evaluation/test_f1_macro.py`

#### `outputs/patchcore_scores_corrected.csv`

Scores PatchCore avec indication de correction :
- 4 589 lignes, colonnes : `filename`, `patchcore_score`, `correction_applied`
- `correction_applied = False` pour toutes les lignes (pas d'inversion appliquée)
- Généré par `outputs/check_score_inversion.py` (mis à jour cette session)

### Scripts mis à jour

#### `outputs/check_score_inversion.py` — ajout sauvegarde CSV

Nouvelle fonction `save_corrected_scores(df, apply_inversion)` qui :
- **Toujours** sauvegarde `outputs/patchcore_scores_corrected.csv`
- Ajoute colonne `correction_applied` (True=inversé, False=original)
- Décision finale : pas d'inversion (AUC-ROC original = 0.7112 > AUC inversé)

#### `outputs/figures/plot_patchcore_timeline_interactive.py` — REFONTE COMPLÈTE

Remplacé par une version avec **filtrage réel** (pas seulement zoom visuel) :

| Fonctionnalité | Ancienne version | Nouvelle version |
|---|---|---|
| Filtrage par année | ❌ absent | ✅ Dropdown → `relayout xaxis.range` |
| Filtrage par mois | ❌ absent | ✅ Dropdown → `restyle visible` |
| Combinaison année+mois | ❌ absent | ✅ (ex : mai 2018 uniquement) |
| Bouton Reset | ❌ absent | ✅ `method="update"` (les deux filtres) |
| Points individuels | ❌ moyennes seulement | ✅ 5 673 points réels (12 traces mois) |
| Quality_flag | ❌ absent | ✅ couleur + symbole par type |
| Rangeslider | ✅ présent | ✅ présent |
| Export PNG | ❌ absent | ✅ haute résolution (1400×800 ×2) |

Architecture des 17 traces : indices 0–11 = données par mois, 12 = moyenne mensuelle,
13 = seuil P90, 14–16 = lignes verticales événements.

**Données** : 5 673 images réelles (2014–2025, 12 mois couverts).  
**Sortie** : `outputs/figures/patchcore_timeline_interactive.html`

#### `outputs/figures/patchcore_dashboard.py` — NOUVEAU

Application Dash 4.x interactive avec callbacks Python :
- Dropdown Année + Dropdown Mois → filtre `DF_FULL` en Python avant re-render
- Bouton Reset → callback réinitialisant les deux dropdowns
- KPI cards : N images, score moyen, score max, % > P90
- Graphique scatter (par quality_flag) + courbe moyenne mensuelle + seuil P90
- Histogramme empilé de distribution des scores en bas de page
- **Vrai filtrage** : contrairement à l'HTML statique, les données sont filtrées côté serveur

**Lancement** : `/opt/anaconda3/bin/python outputs/figures/patchcore_dashboard.py`  
**URL** : http://127.0.0.1:8050  
**Dépendances** : Dash 4.1.0 (installé), plotly, pandas, numpy

### Fichiers modifiés (session 6 complète)

| Fichier | Modification |
|---|---|
| `src/evaluation/metrics.py` | Bug F1-macro corrigé + AUC-ROC ajouté |
| `src/evaluation/test_f1_macro.py` | **Nouveau** — 5 tests unitaires |
| `outputs/check_score_inversion.py` | `save_corrected_scores()` → CSV garanti |
| `outputs/patchcore_scores_corrected.csv` | **Nouveau** — 4 589 lignes |
| `outputs/figures/plot_patchcore_timeline_interactive.py` | Refonte complète filtrage année/mois |
| `outputs/figures/patchcore_dashboard.py` | **Nouveau** — app Dash |
| `outputs/figures/patchcore_timeline_interactive.html` | Regénéré (5 673 points réels) |
| `outputs/scores/evaluation_report.csv` | f1_macro=0.5, auc_roc=0.7112 |

---

## Sessions précédentes (résumé)

### 2026-04 (sessions 1–4)

- Correction crash PIL (`safe_load_image`, `_pil_timeout` SIGALRM)
- Ajout anti-freeze `audit_image_dataset()` (ProcessPoolExecutor)
- Ajout colonne `on_disk` dans `load_index()`
- Couverture pipeline exposée dans Streamlit (Accueil, Statistiques, sidebar)
- Création `outputs/figures/plot_patchcore_timeline.py` (matplotlib statique)
- Corrections diverses dans `run_v1_pipeline.py` et `src/models/patchcore_detector.py`

## Session 7 — 2025-04-30

### Corrections métriques (`src/evaluation/metrics.py`)
- **Correction signe `effect_size`** : formule corrigée de `1 - 2U/n1n2` → `2U/n1n2 - 1` (rank-biserial correlation standard). Résultat : **+0.546** (dark > usable confirmé, signal réel positif).
- **Ajout `compute_auc_roc()`** : fonction manquante ajoutée (AUC-ROC = 0.7732, n+=139, n-=1903).
- **Correction baseline AUC-PR** : `anomaly_score` était une copie de `patchcore_score` → delta=0 impossible. Remplacé par **classifieur aléatoire = prévalence** (Davis & Goadrich 2006) : baseline = **0.0681**, delta = **+0.1074**, amélioration = **+157.7%** (PatchCore 2.6× mieux que random).
- **Régénération** `outputs/scores/evaluation_report.csv` avec métriques corrigées.

### Nettoyage UI (`app/streamlit_app.py`)
- **Suppression timeline matplotlib** dans `page_patchcore()` (~35 lignes) → remplacée par une note de redirection vers **📊 Timeline PatchCore**.
- **Suppression section "Timeline complète"** dans `page_early_warning()` (~18 lignes, `early_warning_timeline.png`) → redondante avec les pages dédiées.
- **Ajout interprétation automatique** dans le rapport d'évaluation : 5 règles UX (signal détecté, p-value, amélioration ×, F1 limité, AUC-PR faible).
