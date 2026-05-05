#!/usr/bin/env python3
"""
train_volcano_pipeline.py — Entraînement du VolcanoClassifier Merapi.

Étapes exécutées :
  1. Recalcul des 10 features physiques (remplace le CSV à 5 colonnes)
  2. Auto-labellisation : patchcore + heuristiques + événements connus Merapi
  3. Entraînement VolcanoClassifier (RandomForest)
  4. Sauvegarde outputs/models/volcano_clf.pkl + rapport

Usage :
    USE_TF=0 USE_TORCH=1 python train_volcano_pipeline.py

    # Sauter le recalcul des features si déjà calculées avec 10 colonnes
    USE_TF=0 USE_TORCH=1 python train_volcano_pipeline.py --skip-features

    # Afficher les tops anomalies pour validation manuelle
    USE_TF=0 USE_TORCH=1 python train_volcano_pipeline.py --show-top 20
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_volcano_pipeline")

# ─── Chemins ──────────────────────────────────────────────────────────────
FEATURES_CSV   = PROJECT_ROOT / "outputs" / "models" / "physical_features.csv"
LABELS_CSV     = PROJECT_ROOT / "outputs" / "models" / "labels.csv"
MODEL_PATH     = PROJECT_ROOT / "outputs" / "models" / "volcano_clf.pkl"
REPORT_PATH    = PROJECT_ROOT / "outputs" / "models" / "volcano_clf_report.txt"
INDEX_CSV      = PROJECT_ROOT / "data"    / "index"  / "index.csv"

# ─── Événements connus Merapi (dates de fort pyroclastisme documenté) ─────
# Source : Smithsonian GVP + BPPTKG bulletins
KNOWN_PYRO_EVENTS: list[tuple[int, int]] = [
    # (year, month) des éruptions importantes avec PDC
    (2018, 5),  # éruption phréatique + PDC 11 mai 2018
    (2018, 6),  # activité post-éruption
    (2020, 11), # éruption VEI-2 (PDC summit area)
    (2021, 1),  # activité continue post-2020
    (2022, 3),  # éruptions fréquentes
    (2023, 4),  # activité strombolienne
]

# ─── Seuils d'auto-labellisation — calibrés sur vraies distributions ──────
# Plages observées sur 848 images Merapi 2014-2018 :
#   patchcore_score  : min=9.15  p50=34.9  p95=44.7  p99=47.3  max=51.3
#   texture_roughness: p50=0.00019  p90=0.00033  p99=0.00058  max=0.001
#   bright_pixel_ratio: p50=0.20  p90=0.60  p95=0.93
#   lbp_entropy      : p50=3.75   p99=4.34
#   pixel_diff_mean  : p90=0.108  p99=0.176
#   NB: cv2 absent → optical_flow_mag=0, edge_density=0, contour_convexity=1 toujours
THRESHOLDS = {
    # pyroclastique — top ~2% en anomalie patchcore (valeur absolue)
    "pyro_patchcore_high":  46.5,   # p98 patchcore → anomalie très forte
    "pyro_patchcore_mid":   44.0,   # p95 patchcore + texture chaotique
    "pyro_texture_p90":     0.00033, # texture_roughness > p90
    "pyro_tchange":         0.12,    # temporal_change_score > p95
    # lave — nocturne uniquement (bright_pixel_ratio élevé de nuit = incandescence)
    "lave_bright_night":    0.02,    # bright_pixel_ratio > 2% la nuit → lave
    "lave_bright_day":      0.92,    # > p95 le jour → surexposition thermique
    # nuage — patchcore bas + entropy basse (scène homogène)
    "nuage_patchcore_max":  30.0,    # < p25 patchcore → scène très ordinaire
    "nuage_entropy_max":    3.50,    # < p50 entropy → texture homogène
    "nuage_pixdiff_max":    0.03,    # pixel_diff_mean très bas → scène stable
}


# ══════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — (Re)calcul des 10 features physiques
# ══════════════════════════════════════════════════════════════════════════

def step_features(force: bool = False) -> pd.DataFrame:
    """Calcule ou charge les 10 features physiques.

    Si le CSV existant n'a que 5 colonnes ou si force=True, relance
    le calcul complet via PhysicalFeatureExtractor.compute_all().
    """
    logger.info("━" * 60)
    logger.info("ÉTAPE 1 — Features physiques (10 colonnes)")

    from src.features.physical_features import PhysicalFeatureExtractor, FEATURE_NAMES

    # Vérifier si le CSV existant est complet (10 features)
    if FEATURES_CSV.exists() and not force:
        df_existing = pd.read_csv(FEATURES_CSV)
        existing_feats = set(df_existing.columns) - {"filename"}
        missing = set(FEATURE_NAMES) - existing_feats
        if not missing:
            logger.info("Features déjà calculées avec 10 colonnes → chargement.")
            return df_existing
        logger.info(
            "CSV incomplet (%d/10 features, manquent : %s) → recalcul.",
            len(existing_feats), missing,
        )

    # Charger l'index
    df_index = pd.read_csv(INDEX_CSV, low_memory=False)
    dl_mask = df_index["downloaded"].astype(str).str.lower() == "true"
    df_dl = df_index[dl_mask].copy()
    logger.info("%d images téléchargées dans l'index.", len(df_dl))

    # Calcul complet
    extractor = PhysicalFeatureExtractor()
    logger.info("Calcul des 10 features sur %d images...", len(df_dl))
    df_features = extractor.compute_all(df_dl, image_root=PROJECT_ROOT)

    # Sauvegarder
    PhysicalFeatureExtractor.save(df_features, FEATURES_CSV)
    logger.info("Features sauvegardées → %s (%d lignes, %d features)", FEATURES_CSV, len(df_features), len(df_features.columns) - 1)
    return df_features


# ══════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Auto-labellisation
# ══════════════════════════════════════════════════════════════════════════

def _label_row(row: pd.Series, is_known_pyro: bool, is_night: bool) -> str:
    """
    Attribue un label basé sur les vraies plages de valeurs observées.

    Règles calibrées sur 848 images Merapi 2014-2018 :
      - cv2 absent → optical_flow_mag=0, edge_density=0, contour_convexity=1 (ignorés)
      - patchcore_score brut dans [9, 51] (non normalisé)
    """
    pc      = float(row.get("patchcore_score", 0.0) or 0.0)
    rough   = float(row.get("texture_roughness", 0.0) or 0.0)
    bright  = float(row.get("bright_pixel_ratio", 0.0) or 0.0)
    ent     = float(row.get("lbp_entropy", 0.0) or 0.0)
    tchange = float(row.get("temporal_change_score", 0.0) or 0.0)
    pixdiff = float(row.get("pixel_diff_mean", 0.0) or 0.0)

    T = THRESHOLDS

    # ─ Priorité 1 : PYROCLASTIQUE ─────────────────────────────────────────
    # Top 2% anomalie (patchcore absolu très élevé)
    if pc > T["pyro_patchcore_high"]:
        return "pyroclastique"
    # Événement documenté + patchcore p95 + texture chaotique
    if is_known_pyro and pc > T["pyro_patchcore_mid"]:
        return "pyroclastique"
    # Patchcore p95 + texture chaotique (p90)
    if pc > T["pyro_patchcore_mid"] and rough > T["pyro_texture_p90"]:
        return "pyroclastique"
    # Changement temporel brutal (p95) + anomalie modérée
    if tchange > T["pyro_tchange"] and pc > 40.0:
        return "pyroclastique"

    # ─ Priorité 2 : LAVE ─────────────────────────────────────────────────
    # La nuit + pixels brillants = incandescence lave
    if is_night and bright > T["lave_bright_night"]:
        return "lave"
    # Le jour + surexposition extrême (front de coulée très chaud)
    if not is_night and bright > T["lave_bright_day"]:
        return "lave"

    # ─ Priorité 3 : NUAGE ────────────────────────────────────────────────
    # Weather label explicite
    wl = str(row.get("weather_label", "")).lower()
    if "cloud" in wl or "nuage" in wl or "fog" in wl:
        return "nuage"
    # Scène très ordinaire : patchcore bas + entropy basse + peu de changement
    if (pc < T["nuage_patchcore_max"]
            and ent < T["nuage_entropy_max"]
            and pixdiff < T["nuage_pixdiff_max"]):
        return "nuage"

    # ─ Par défaut : NORMAL ───────────────────────────────────────────────
    return "normal"


def step_autolabel(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Fusionne features + index, applique l'auto-labellisation,
    sauvegarde outputs/models/labels.csv, retourne df_labeled.
    """
    logger.info("━" * 60)
    logger.info("ÉTAPE 2 — Auto-labellisation")

    # Charger l'index avec patchcore_score et weather_label
    df_index = pd.read_csv(INDEX_CSV, low_memory=False)

    # Construire l'ensemble des (year, month) pyroclastiques connus
    known_pyro_periods = set(KNOWN_PYRO_EVENTS)

    # Colonnes nécessaires depuis l'index
    idx_cols = ["filename", "year", "month", "patchcore_score", "weather_label",
                "is_night", "quality_flag"]
    idx_cols_ok = [c for c in idx_cols if c in df_index.columns]

    # Fusion features + index sur filename
    df_merged = df_features.merge(
        df_index[idx_cols_ok],
        on="filename",
        how="left",
    )
    logger.info("Fusion index+features : %d lignes.", len(df_merged))

    # Exclure les images de mauvaise qualité pour l'entraînement
    bad_quality = df_merged["quality_flag"].isin(["bad", "unusable", "dark"])
    logger.info("Exclusion %d images mauvaise qualité.", bad_quality.sum())
    df_usable = df_merged[~bad_quality].copy()

    # Créer la colonne is_known_pyro
    if df_usable.empty:
        logger.error(
            "Aucune image trouvée après fusion features+index. "
            "Vérifier que physical_features.csv n'est pas vide."
        )
        raise RuntimeError("df_usable vide — recalculer les features avec --force-features")

    def _is_pyro_period(r: pd.Series) -> bool:
        try:
            return (int(r.get("year") or 0), int(r.get("month") or 0)) in known_pyro_periods
        except (TypeError, ValueError):
            return False

    df_usable["is_known_pyro"] = df_usable.apply(_is_pyro_period, axis=1).astype(bool)

    # is_night : booléen robuste (peut être True/False/1/0/NaN)
    def _to_bool(v) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        return str(v).lower() in ("true", "1", "yes")

    # Appliquer le labeleur
    df_usable["label"] = df_usable.apply(
        lambda r: _label_row(
            r,
            bool(r["is_known_pyro"]),
            _to_bool(r.get("is_night", False)),
        ),
        axis=1,
    )

    # Rapport
    dist = df_usable["label"].value_counts()
    logger.info("Distribution des labels auto :")
    for cls, n in dist.items():
        logger.info("  %-18s : %d images", cls, n)

    # Sauvegarder
    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_usable[["filename", "year", "month", "label", "patchcore_score"]].to_csv(LABELS_CSV, index=False)
    logger.info("Labels sauvegardés → %s", LABELS_CSV)

    return df_usable


# ══════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Entraînement VolcanoClassifier
# ══════════════════════════════════════════════════════════════════════════

def step_train(df_labeled: pd.DataFrame) -> None:
    """Entraîne et sauvegarde le VolcanoClassifier."""
    logger.info("━" * 60)
    logger.info("ÉTAPE 3 — Entraînement VolcanoClassifier")

    from src.models.volcano_classifier import VolcanoClassifier, CLASSIFIER_FEATURES, VOLCANO_CLASSES

    # Préparer X et y
    feature_cols = [c for c in CLASSIFIER_FEATURES if c in df_labeled.columns]
    logger.info("Features utilisées pour l'entraînement : %s", feature_cols)

    df_train = df_labeled.dropna(subset=["label"]).copy()
    df_train[feature_cols] = df_train[feature_cols].fillna(0.0)

    X = df_train[feature_cols]
    y = df_train["label"]

    # Rapport des classes
    logger.info("Classes disponibles : %s", y.value_counts().to_dict())

    # Entraînement
    clf = VolcanoClassifier()
    clf.fit(X, y)

    if clf._is_heuristic:
        logger.warning("Modèle en mode heuristique (données insuffisantes pour RF).")
        logger.warning("Vérifier le CSV labels.csv — classes présentes : %s", y.unique().tolist())
    else:
        logger.info("Modèle RandomForest entraîné avec succès.")

    # Sauvegarde
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    clf.save(MODEL_PATH)
    logger.info("Modèle sauvegardé → %s", MODEL_PATH)

    # ── Rapport de classification ──────────────────────────────────────
    try:
        from sklearn.metrics import classification_report
        preds = clf.predict(X)
        # Utiliser les classes réellement présentes, pas les 4 classes théoriques
        labels_present = sorted(y.unique().tolist())
        report = classification_report(y, preds, labels=labels_present, zero_division=0)
        logger.info("Rapport sur données d'entraînement (indicatif — pas de test set):\n%s", report)
        REPORT_PATH.write_text(
            f"VolcanoClassifier — Rapport d'entraînement\n"
            f"{'='*60}\n"
            f"Données : {len(df_train)} images (auto-labels)\n"
            f"Features : {feature_cols}\n\n"
            f"{report}\n",
            encoding="utf-8",
        )
        logger.info("Rapport sauvegardé → %s", REPORT_PATH)
    except ImportError:
        logger.warning("scikit-learn non disponible — pas de rapport de classification.")
    except Exception as e:
        logger.warning("Rapport impossible : %s", e)


# ══════════════════════════════════════════════════════════════════════════
# Affichage des top anomalies (validation manuelle)
# ══════════════════════════════════════════════════════════════════════════

def show_top_anomalies(df_labeled: pd.DataFrame, n: int = 20) -> None:
    """Affiche les N images avec le patchcore_score le plus élevé."""
    logger.info("━" * 60)
    logger.info("TOP %d anomalies (patchcore_score) :", n)

    cols_show = ["filename", "year", "month", "patchcore_score", "label",
                 "texture_roughness", "bright_pixel_ratio", "thermal_gradient"]
    cols_ok = [c for c in cols_show if c in df_labeled.columns]

    top = (
        df_labeled.dropna(subset=["patchcore_score"])
        .sort_values("patchcore_score", ascending=False)
        .head(n)
    )
    pd.set_option("display.max_colwidth", 45)
    pd.set_option("display.width", 160)
    print(top[cols_ok].to_string(index=False))
    print()

    # Suggestion : images à labelliser manuellement si le modèle hésite
    ambiguous = df_labeled[
        (df_labeled["patchcore_score"] > 0.55) &
        (df_labeled["patchcore_score"] < 0.72)
    ]
    logger.info(
        "%d images 'ambiguës' (patchcore 0.55–0.72) — candidats à la labellisation manuelle.",
        len(ambiguous),
    )


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main(args: argparse.Namespace) -> None:
    # Étape 1 : features
    df_features = step_features(force=args.force_features)

    # Étape 2 : auto-labellisation
    df_labeled = step_autolabel(df_features)

    # Affichage optionnel
    if args.show_top > 0:
        show_top_anomalies(df_labeled, n=args.show_top)

    # Étape 3 : entraînement
    step_train(df_labeled)

    logger.info("━" * 60)
    logger.info("Pipeline terminé.")
    logger.info("  Modèle      → %s", MODEL_PATH)
    logger.info("  Labels      → %s", LABELS_CSV)
    logger.info("  Features    → %s", FEATURES_CSV)
    logger.info("  Rapport     → %s", REPORT_PATH)
    logger.info("")
    logger.info("Prochaine étape : scraper les images 2019-2025 puis relancer l'app.")
    logger.info("  USE_TF=0 USE_TORCH=1 python run_full_pipeline.py --year 2019 --max-per-month -1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement VolcanoClassifier Merapi")
    parser.add_argument(
        "--skip-features", action="store_true",
        help="Sauter le recalcul des features (utiliser le CSV existant tel quel)",
    )
    parser.add_argument(
        "--force-features", action="store_true",
        help="Forcer le recalcul des 10 features même si le CSV est complet",
    )
    parser.add_argument(
        "--show-top", type=int, default=0, metavar="N",
        help="Afficher les N images avec le patchcore_score le plus élevé (défaut: 0)",
    )
    main(parser.parse_args())
