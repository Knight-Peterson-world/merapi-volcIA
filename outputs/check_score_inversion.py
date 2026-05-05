"""
check_score_inversion.py — Diagnostic de l'inversion des scores PatchCore.

Problème investigué :
  L'effet_size Mann-Whitney est négatif (-0.5464), ce qui signifie que les images
  « dark » (proxy d'anomalies volcanique) ont des scores PatchCore PLUS BAS que
  les images « usable » normales.

  Cause probable : PatchCore calcule la distance au coreset d'images normales.
  Une image sombre (nuit) a une texture uniforme similaire à certains fonds dans
  le coreset → faible distance → score bas.

  Correction envisagée : inverser les scores (ex. -score ou 1 - score normalisé)
  avant de calculer l'AUC-PR, pour que les anomalies aient des scores PLUS HAUTS.

Usage :
    /opt/anaconda3/bin/python outputs/check_score_inversion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ajouter la racine du projet au path pour accéder à src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SCORES_PATH = PROJECT_ROOT / "outputs/scores/patchcore_scores.csv"
INDEX_PATH  = PROJECT_ROOT / "data/index/index.csv"


# ─── Chargement des données ──────────────────────────────────────────────────

def load_labeled_scores() -> pd.DataFrame:
    """
    Fusionne patchcore_scores.csv avec l'index pour obtenir quality_flag,
    puis filtre les images ayant un label connu (dark ou usable).
    """
    scores = pd.read_csv(SCORES_PATH)
    index  = pd.read_csv(INDEX_PATH, low_memory=False)

    df = scores.merge(
        index[["filename", "quality_flag", "hour"]].drop_duplicates("filename"),
        on="filename",
        how="inner",
    )

    # Garder uniquement les images scorées
    df = df.dropna(subset=["patchcore_score"])

    print(f"[INFO] Scores chargés : {len(df)} images avec score non-NaN")
    print(f"[INFO] quality_flag distribution :")
    print(df["quality_flag"].value_counts().to_string())

    # Filtre : positifs = dark, négatifs = usable diurnes (06h-17h)
    mask_dark   = df["quality_flag"] == "dark"
    mask_usable = (
        (df["quality_flag"] == "usable") &
        df["hour"].between(6, 17, inclusive="both")
    )
    df_labeled = df[mask_dark | mask_usable].copy()
    df_labeled["y_true"] = mask_dark[mask_dark | mask_usable].astype(int)

    n_pos = int(df_labeled["y_true"].sum())
    n_neg = int((df_labeled["y_true"] == 0).sum())
    print(f"\n[INFO] Sous-ensemble évaluation : {n_pos} dark (positifs) + {n_neg} usable-diurnes (négatifs)")

    return df_labeled


# ─── Statistiques descriptives ───────────────────────────────────────────────

def describe_groups(df: pd.DataFrame) -> None:
    """Affiche les statistiques de scores par groupe."""
    print("\n" + "=" * 60)
    print("DISTRIBUTION DES SCORES PAR GROUPE")
    print("=" * 60)
    for label, name in [(1, "DARK (proxy anomalie)"), (0, "USABLE diurne (normal)")]:
        sub = df[df["y_true"] == label]["patchcore_score"]
        print(f"\n  {name}  (n={len(sub)})")
        print(f"    mean  = {sub.mean():.4f}")
        print(f"    median= {sub.median():.4f}")
        print(f"    std   = {sub.std():.4f}")
        print(f"    min   = {sub.min():.4f}")
        print(f"    max   = {sub.max():.4f}")
        print(f"    P10   = {sub.quantile(0.1):.4f}")
        print(f"    P90   = {sub.quantile(0.9):.4f}")


# ─── Test d'inversion ────────────────────────────────────────────────────────

def evaluate_with_optional_inversion(df: pd.DataFrame) -> dict:
    """
    Calcule AUC-PR et AUC-ROC avec et sans inversion des scores.

    Retourne un dict avec les résultats comparés.
    """
    y_true  = df["y_true"].values
    s_orig  = df["patchcore_score"].values

    # Score normalisé dans [0, 1]
    s_min, s_max = s_orig.min(), s_orig.max()
    s_norm = (s_orig - s_min) / (s_max - s_min + 1e-9)

    # Score inversé normalisé
    s_inv  = 1.0 - s_norm

    results = {}

    for variant, scores in [("original", s_orig), ("inversé (1-norm)", s_inv)]:
        auc_pr  = float(average_precision_score(y_true, scores))
        auc_roc = float(roc_auc_score(y_true, scores))
        random_baseline = float(y_true.mean())

        results[variant] = {
            "auc_pr" : auc_pr,
            "auc_roc": auc_roc,
            "above_random": auc_pr > random_baseline,
            "random_baseline": random_baseline,
        }

    return results


# ─── Recommandation ──────────────────────────────────────────────────────────

def print_recommendation(results: dict, mean_dark: float, mean_usable: float) -> bool:
    """
    Affiche la recommandation et retourne True si l'inversion est conseillée.
    """
    print("\n" + "=" * 60)
    print("RÉSULTATS COMPARÉS")
    print("=" * 60)
    rb = results["original"]["random_baseline"]
    print(f"\n  Baseline aléatoire (fréquence classe +) : {rb:.4f}")
    print()

    for variant, r in results.items():
        marker = "✓ AU-DESSUS du hasard" if r["above_random"] else "✗ EN-DESSOUS du hasard"
        print(f"  [{variant}]")
        print(f"    AUC-PR  = {r['auc_pr']:.4f}  {marker}")
        print(f"    AUC-ROC = {r['auc_roc']:.4f}")
        print()

    print("=" * 60)

    inverted_better = results["inversé (1-norm)"]["auc_pr"] > results["original"]["auc_pr"]
    inversion_above_random = results["inversé (1-norm)"]["above_random"]

    print("\nDIAGNOSTIC :")
    if mean_dark < mean_usable:
        print(f"  → Score moyen DARK ({mean_dark:.2f}) < USABLE ({mean_usable:.2f})")
        print("  → Les anomalies ont des scores plus BAS : inversion nécessaire")
    else:
        print(f"  → Score moyen DARK ({mean_dark:.2f}) > USABLE ({mean_usable:.2f})")
        print("  → Les anomalies ont des scores plus HAUTS : pas d'inversion nécessaire")

    if inverted_better and inversion_above_random:
        print("\n  RECOMMANDATION : APPLIQUER l'inversion (1 - score normalisé)")
        print("  → L'AUC-PR inversée est plus haute ET au-dessus du hasard.")
        return True
    elif inverted_better:
        print("\n  RECOMMANDATION PARTIELLE : l'inversion améliore l'AUC-PR")
        print("  mais reste en-dessous du hasard → proxy de labels inadapté.")
        return False
    else:
        print("\n  RECOMMANDATION : ne PAS inverser")
        print("  → Le score original est déjà la meilleure orientation.")
        return False


# ─── Sauvegarde des scores corrigés ────────────────────────────────────────────────

def save_corrected_scores(
    df_all_scores: pd.DataFrame,
    apply_inversion: bool,
) -> Path:
    """
    Sauvegarde toujours un fichier patchcore_scores_corrected.csv.

    - Si inversion nécessaire : scores normalisés inversés (1 - score_norm)
    - Sinon           : scores originaux (non-NaN uniquement, non modifiés)

    Le fichier inclut une colonne 'correction_applied' (True/False) pour
    tracer la décision de correction dans le pipeline aval.

    Returns:
        Path du fichier sauvegardé.
    """
    out_dir  = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "patchcore_scores_corrected.csv"

    df_out = df_all_scores[["filename", "patchcore_score"]].copy()

    if apply_inversion:
        s = df_out["patchcore_score"]
        s_min, s_max = s.min(), s.max()
        df_out["patchcore_score"] = 1.0 - (s - s_min) / (s_max - s_min + 1e-9)
        df_out["correction_applied"] = True
        action = "INVERSÉS (1 - score normalisé)"
    else:
        df_out["correction_applied"] = False
        action = "ORIGINAUX (aucune modification)"

    df_out.to_csv(out_path, index=False)
    print(f"\n[OK] Scores corrigés ({action})")
    print(f"     Sauvegardés → {out_path}")
    print(f"     {len(df_out)} lignes | correction_applied={apply_inversion}")
    return out_path


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("CHECK SCORE INVERSION — PatchCore Merapi")
    print("=" * 60)

    df_labeled = load_labeled_scores()

    if df_labeled["y_true"].sum() == 0:
        print("[ERREUR] Aucune image 'dark' trouvée dans l'index — impossible de comparer.")
        return

    describe_groups(df_labeled)

    mean_dark   = df_labeled[df_labeled["y_true"] == 1]["patchcore_score"].mean()
    mean_usable = df_labeled[df_labeled["y_true"] == 0]["patchcore_score"].mean()

    results = evaluate_with_optional_inversion(df_labeled)
    should_invert = print_recommendation(results, mean_dark, mean_usable)

    # Toujours sauvegarder patchcore_scores_corrected.csv
    all_scores = pd.read_csv(SCORES_PATH)
    non_nan = all_scores["patchcore_score"].notna()
    save_corrected_scores(all_scores[non_nan].copy(), apply_inversion=should_invert)

    if not should_invert:
        print("\n[INFO] Les scores originaux sont conservés dans patchcore_scores_corrected.csv.")
        print("       Si l'AUC-PR reste faible, le problème vient du proxy de labels")
        print("       (images 'dark' ne sont pas toutes des anomalies volcaniques).")


if __name__ == "__main__":
    main()
