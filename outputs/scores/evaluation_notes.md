# Rapport d'évaluation PatchCore — Limitations et corrections

Date : 2026-04-30  
Version pipeline : V1

---

## Métriques corrigées (résumé)

| Métrique | Valeur initiale | Valeur corrigée | Note |
|---|---|---|---|
| AUC-PR PatchCore | 0.1754 | 0.1313 | Périmètre d'éval. différent (voir §1) |
| AUC-ROC PatchCore | — | **0.7112** | Ajouté (voir §2) |
| F1-macro triage | **1.0 (incorrect)** | **0.5** | Bug corrigé (voir §3) |
| F1-volcanique | 1.0 | 1.0 | Inchangé |
| F1-météo | 0.0 | 0.0 | Inchangé |
| Mann-Whitney p | 2.43e-27 | 2.43e-27 | Inchangé (positif, pas négatif) |
| Effect size | -0.5464 | -0.5464 | Inchangé (signe attendu, voir §4) |

---

## §1 — AUC-PR : périmètre d'évaluation

**Valeur initiale** : 0.1754  
Calculée sur les images de l'année 2018 (n_test=486, n_positive_proxy=139 images "dark").

**Valeur recalculée** : 0.1313  
Calculée sur un sous-ensemble plus propre : 128 images "dark" + 1653 images "usable"
diurnes (6h-17h), toutes années confondues.

**Baseline aléatoire** : 0.0719 (7.2% de positifs)  
Les deux valeurs sont au-dessus du hasard — le modèle est informatifd.

**Pourquoi l'AUC-PR reste modérée** :
- Le proxy de positifs ("dark") capture des nuits normales + des nuits avec activité volcanique,
  sans distinguer les deux. Le label est bruité.
- PatchCore est entraîné sur 2014–2017 (images "usable"). Il détecte les écarts de texture,
  pas spécifiquement l'activité volcanique.
- L'AUC-PR est sensible au déséquilibre de classes. Avec ~7% de positifs, une AUC-PR de
  0.13 représente un signal réel mais faible.

---

## §2 — AUC-ROC : ajout d'une métrique complémentaire

L'**AUC-ROC** (0.7112) est plus adaptée que l'AUC-PR sur des classes aussi déséquilibrées.
Elle mesure la capacité de classement global, indépendamment du seuil et de la fréquence
des classes.

**Interprétation** : AUC-ROC = 0.71 signifie que dans 71% des paires (anomalie, normale)
tirées au sort, l'anomalie reçoit un score plus élevé. C'est une bonne discrimination.

→ **Le modèle PatchCore est utilisable comme système de ranking** (top-K% à inspecter),
même si sa précision binaire reste limitée.

---

## §3 — Bug F1-macro : description et correction

**Bug** : `f1_score(y_true, y_pred, average='macro', zero_division=0)` retournait 1.0
au lieu de 0.5.

**Cause** : lorsque `y_true` ne contient qu'une seule classe (ici uniquement des images "dark",
aucune "cloudy" dans le merge), sklearn calcule la moyenne sur les classes *présentes* dans
`y_true`. Avec une seule classe (f1=1.0), la moyenne = 1.0, masquant le fait que la classe
météo n'était pas évaluée.

**Correction appliquée** dans `src/evaluation/metrics.py`, fonction `compute_f1_triage()` :

```python
# Avant (incorrect)
f1_score(y_true, y_pred, average='macro', zero_division=0)

# Après (corrigé)
f1_score(y_true, y_pred, average='macro', labels=[0, 1], zero_division=0)
```

Le paramètre `labels=[0, 1]` force l'inclusion des deux classes dans la moyenne,
même si l'une est absente de `y_true`.

**Valeur corrigée** : F1-macro = (1.0 + 0.0) / 2 = **0.5**

Ce résultat reflète la réalité : le classifieur SVM identifie parfaitement les images
volcaniques (dark) mais échoue sur les images météo (cloudy), qui sont quasi-absentes
du dataset d'entraînement.

---

## §4 — Effect size négatif : explication

**Valeur** : effect_size = -0.5464 (rank-biserial correlation)

**Ce n'est PAS une erreur**. Le signe dépend de la formule :

```python
effect_size = 1 - (2 * U) / (n1 * n2)
```

Avec `mannwhitneyu(dark_scores, usable_scores, alternative="greater")` :
- U grand (dark scores > usable scores) → effect_size négatif
- U petit (dark scores < usable scores) → effect_size positif

Le diagnostic (`check_score_inversion.py`) confirme :
- Score moyen DARK = 45.96 > Score moyen USABLE = 42.13
- Les anomalies ont bien des scores **plus hauts** → pas d'inversion nécessaire

La p-value (2.43e-27) confirme que la différence est hautement significative.

---

## §5 — Inversion des scores : résultat du diagnostic

Script : `outputs/check_score_inversion.py`

| Configuration | AUC-PR | AUC-ROC | Au-dessus du hasard ? |
|---|---|---|---|
| Score original | 0.1313 | 0.7112 | ✓ Oui |
| Score inversé (1 - norm) | 0.0478 | 0.2888 | ✗ Non |

**Conclusion : ne pas inverser.** Le score original est la meilleure orientation.

---

## Recommandations futures

1. **Améliorer le proxy de labels** : annoter manuellement ~200 images (éruption confirmée vs
   nuit calme) pour obtenir un ground truth fiable et recalculer l'AUC-PR.

2. **Calibrer le seuil** : le P90 (49.975) est arbitraire. Utiliser une courbe PR pour choisir
   le seuil selon la tolérance aux faux positifs.

3. **Compléter le dataset météo** : avec seulement 2 images "cloudy" dans l'index, le
   classifieur SVM ne peut pas apprendre la classe météo. Augmenter le dataset ciblé.

4. **Évaluation temporelle** : séparer le jeu de test par saison (sèche vs humide) pour
   identifier des biais saisonniers dans les scores.
