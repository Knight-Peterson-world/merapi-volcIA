#!/usr/bin/env python3
"""
run_v1_pipeline.py — Pipeline V1 complet : DINOv2 + PatchCore + SVM + Évaluation.

Exécution complète (depuis la racine du projet) :
    python run_v1_pipeline.py

Exécution par étapes individuelles :
    python run_v1_pipeline.py --step features
    python run_v1_pipeline.py --step patchcore
    python run_v1_pipeline.py --step weather
    python run_v1_pipeline.py --step evaluate
    python run_v1_pipeline.py --step early_warning

Split temporel rigoureux (aucune fuite d'information) :
    Train : 2014–2017, usable, diurne (6h–18h)
    Test  : 2018, usable

Sorties :
    outputs/models/physical_features.csv  — features physiques (toutes images)
    outputs/models/patchcore.npz          — coreset PatchCore
    outputs/models/weather_svm.pkl        — classifieur SVM
    outputs/scores/patchcore_scores.csv   — scores PatchCore par image
    outputs/scores/weather_predictions.csv — prédictions triage météo
    outputs/scores/evaluation_report.csv  — métriques AUC-PR, F1
    outputs/scores/early_warning_*.csv    — analyse précurseurs
    outputs/figures/early_warning_timeline.png — figure principale
    data/index/index.csv                  — mis à jour avec patchcore_score, weather_label
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ─── Racine du projet ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.utils import load_config, PROJECT_ROOT as SRC_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_v1_pipeline")


# ─── Étapes du pipeline ───────────────────────────────────────────────────

def step_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Étape 1 : Extraction des features physiques sur toutes les images usable.

    Sorties :
      - outputs/models/physical_features.csv
    """
    logger.info("─" * 60)
    logger.info("ÉTAPE 1 — Features physiques (optical flow, LBP, brightness, convexité)")

    from src.features.physical_features import PhysicalFeatureExtractor

    features_path = SRC_ROOT / "outputs" / "models" / "physical_features.csv"

    if features_path.exists():
        logger.info("Features déjà calculées → chargement depuis %s", features_path)
        df_features = PhysicalFeatureExtractor.load(features_path)
        logger.info("%d lignes de features chargées", len(df_features))
        # Vérification : les images labellisées doivent être couvertes
        labeled_fn = set(df[df["quality_flag"].isin(["cloudy", "dark"])]["filename"])
        feat_fn = set(df_features["filename"])
        inter = len(labeled_fn & feat_fn)
        if inter < 20:
            logger.warning(
                "Cache features stale : seulement %d/%d images labellisées ont des features. "
                "Suppression du cache et recalcul...",
                inter, len(labeled_fn),
            )
            features_path.unlink()
            # Recalcul ci-dessous
        else:
            logger.info("Coverage labellisées : %d/%d ✓", inter, len(labeled_fn))
            return df_features

    extractor = PhysicalFeatureExtractor(max_gap_minutes=20)
    n_to_process = df["quality_flag"].isin(["usable", "cloudy", "dark"]).sum()
    logger.info("Calcul des features sur %d images (usable+cloudy+dark)...", n_to_process)
    df_features = extractor.compute_all(df, image_root=SRC_ROOT)
    extractor.save(df_features, features_path)
    logger.info("Features sauvegardées → %s (%d images)", features_path, len(df_features))
    return df_features


def step_patchcore(df: pd.DataFrame) -> pd.Series:
    """
    Étape 2 : Entraînement PatchCore (train 2014–2017) et scoring (test 2018 + all).

    Sorties :
      - outputs/models/patchcore.npz
      - outputs/scores/patchcore_scores.csv
    """
    logger.info("─" * 60)
    logger.info("ÉTAPE 2 — PatchCore (DINOv2-small features + coreset)")

    from src.models.patchcore_detector import PatchCoreDetector

    patchcore_path = SRC_ROOT / "outputs" / "models" / "patchcore.npz"
    scores_path = SRC_ROOT / "outputs" / "scores" / "patchcore_scores.csv"

    detector = PatchCoreDetector(coreset_ratio=0.15)

    if patchcore_path.exists():
        logger.info("Coreset existant → chargement depuis %s", patchcore_path)
        detector.load(patchcore_path)
    else:
        logger.info("Entraînement PatchCore sur images 2014–2017 diurnes...")
        detector.fit(df, image_root=SRC_ROOT, max_images=400)
        if detector._is_fitted:
            detector.save(patchcore_path)
        else:
            logger.warning("PatchCore non entraîné (DINOv2 indisponible) — étape sautée.")

    logger.info("Scoring sur toutes les images usable...")
    scores = detector.score_dataframe(df, image_root=SRC_ROOT, split="all")

    # Export CSV
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df = pd.DataFrame({
        "filename": df["filename"],
        "patchcore_score": scores,
    })
    scores_df.to_csv(scores_path, index=False)
    logger.info("Scores PatchCore sauvegardés → %s", scores_path)

    # Statistiques de couverture par année
    n_scored = int(scores.notna().sum())
    logger.info("Couverture globale : %d / %d images scorées", n_scored, len(df))
    if "year" in df.columns and n_scored > 0:
        df_tmp = df.copy()
        df_tmp["_score"] = scores.values
        by_year = df_tmp[df_tmp["_score"].notna()].groupby("year").size()
        for yr, cnt in by_year.items():
            logger.info("  → %d : %d images scorées", int(yr), cnt)
        n_2018_scored = int((df_tmp["year"].eq(2018) & df_tmp["_score"].notna()).sum())
        if n_2018_scored == 0:
            logger.warning(
                "⚠️  Aucune image 2018 scorée — l'évaluation aura n_test=0.\n"
                "   Téléchargez les images 2018 dans data/raw/2018/ pour corriger cela."
            )

    return scores


def step_weather(df: pd.DataFrame, df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Étape 3 : Entraînement SVM triage météo/volcanique + prédictions.

    Sorties :
      - outputs/models/weather_svm.pkl
      - outputs/scores/weather_predictions.csv
    """
    logger.info("─" * 60)
    logger.info("ÉTAPE 3 — Classifieur SVM météo/volcanique")

    from src.models.weather_classifier import WeatherClassifier

    svm_path = SRC_ROOT / "outputs" / "models" / "weather_svm.pkl"
    pred_path = SRC_ROOT / "outputs" / "scores" / "weather_predictions.csv"

    clf = WeatherClassifier()

    # Vérifier qu'il y a assez d'exemples labellisés
    n_cloudy = (df["quality_flag"] == "cloudy").sum()
    n_dark = (df["quality_flag"] == "dark").sum()
    logger.info("Exemples labellisés : %d cloudy, %d dark", n_cloudy, n_dark)

    if n_cloudy < 5 or n_dark < 5:
        logger.warning(
            "Pas assez d'exemples labellisés (cloudy=%d, dark=%d). "
            "Étape weather ignorée.", n_cloudy, n_dark,
        )
        return pd.DataFrame(columns=["filename", "weather_label", "weather_proba_volcanic"])

    if svm_path.exists():
        logger.info("SVM existant → chargement depuis %s", svm_path)
        clf.load(svm_path)
    else:
        clf.fit(df, df_features)
        clf.save(svm_path)

    metrics = clf.evaluate(df, df_features)
    logger.info(
        "Évaluation SVM : F1-macro=%.3f, F1-volcanic=%.3f, F1-weather=%.3f",
        metrics.get("f1_macro", 0), metrics.get("f1_volcanic", 0), metrics.get("f1_weather", 0),
    )

    predictions = clf.predict(df, df_features)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(pred_path, index=False)
    logger.info("Prédictions sauvegardées → %s", pred_path)

    return predictions


def step_evaluate(
    df: pd.DataFrame,
    patchcore_scores: pd.Series,
    weather_predictions: pd.DataFrame,
) -> None:
    """
    Étape 4 : Calcul de toutes les métriques et mise à jour de index.csv.

    Sorties :
      - outputs/scores/evaluation_report.csv
      - data/index/index.csv (colonnes patchcore_score, weather_label ajoutées)
    """
    logger.info("─" * 60)
    logger.info("ÉTAPE 4 — Évaluation (AUC-PR, F1, Mann-Whitney)")

    from src.evaluation.metrics import evaluate_patchcore

    # Enrichir le DataFrame avec les nouveaux scores
    df_eval = df.copy()
    df_eval["patchcore_score"] = patchcore_scores
    df_eval["year"] = pd.to_numeric(df_eval["year"], errors="coerce")
    df_eval["hour"] = pd.to_numeric(df_eval["hour"], errors="coerce")

    # ── Validation pré-évaluation ──────────────────────────────────────────
    n_scored_total = int(df_eval["patchcore_score"].notna().sum())
    n_2018_in_index = int((df_eval["year"].eq(2018) & df_eval["quality_flag"].eq("usable")).sum())
    n_2018_scored = int((df_eval["year"].eq(2018) & df_eval["patchcore_score"].notna()).sum())
    n_dark_scored = int((df_eval["quality_flag"].eq("dark") & df_eval["patchcore_score"].notna()).sum())
    logger.info("Validation pré-évaluation :")
    logger.info("  Scores PatchCore disponibles  : %d / %d images", n_scored_total, len(df_eval))
    logger.info("  Images 2018 usable (index)     : %d", n_2018_in_index)
    logger.info("  Images 2018 effectivement scorées : %d", n_2018_scored)
    logger.info("  Images 'dark' scorées (proxy+) : %d", n_dark_scored)
    if n_2018_scored == 0:
        logger.warning(
            "⚠️  Aucune image 2018 scorée.\n"
            "   Cause probable : images 2018 non téléchargées localement.\n"
            "   Dossier attendu : data/raw/2018/XX/\n"
            "   Les métriques test seront invalides (n_test=0)."
        )
    if n_dark_scored == 0:
        logger.warning(
            "⚠️  Aucune image 'dark' scorée → AUC-PR proxy invalide (n_positive=0).\n"
            "   Cause : images nocturnes (quality_flag='dark') non téléchargées."
        )

    # Merge prédictions triage
    if not weather_predictions.empty:
        df_eval = df_eval.merge(
            weather_predictions[["filename", "weather_label", "weather_proba_volcanic"]],
            on="filename", how="left",
        )

    # Évaluation
    report = evaluate_patchcore(df_eval, weather_predictions if not weather_predictions.empty else None)
    report.save(SRC_ROOT / "outputs" / "scores" / "evaluation_report.csv")

    # Mise à jour de index.csv avec les nouvelles colonnes
    index_path = SRC_ROOT / "data" / "index" / "index.csv"
    df_updated = df.copy()
    df_updated["patchcore_score"] = patchcore_scores.values

    if not weather_predictions.empty:
        # Merge sur filename
        df_updated = df_updated.merge(
            weather_predictions[["filename", "weather_label", "weather_proba_volcanic"]],
            on="filename", how="left",
        )
        # Si déjà présentes, supprimer les doublons de colonnes
        for col in ["weather_label", "weather_proba_volcanic"]:
            if f"{col}_x" in df_updated.columns:
                df_updated[col] = df_updated[f"{col}_x"].fillna(df_updated.get(f"{col}_y", np.nan))
                df_updated.drop(columns=[f"{col}_x", f"{col}_y"], errors="ignore", inplace=True)

    df_updated.to_csv(index_path, index=False)
    logger.info("index.csv mis à jour avec patchcore_score + weather_label → %s", index_path)


def step_early_warning(df: pd.DataFrame) -> None:
    """
    Étape 5 : Analyse early warning + figure timeline.

    Sorties :
      - outputs/scores/early_warning_precursors.csv
      - outputs/scores/early_warning_permutation.csv
      - outputs/figures/early_warning_timeline.png
    """
    logger.info("─" * 60)
    logger.info("ÉTAPE 5 — Early Warning (corrélation scores / événements BPPTKG)")

    from src.evaluation.early_warning import EarlyWarningAnalyzer

    score_col = "patchcore_score" if "patchcore_score" in df.columns else "anomaly_score"
    logger.info("Colonne de score utilisée : %s", score_col)

    if score_col not in df.columns or df[score_col].notna().sum() == 0:
        logger.warning("Aucun score disponible pour early warning. Exécutez l'étape patchcore.")
        return

    analyzer = EarlyWarningAnalyzer()

    try:
        # Scores précurseurs
        precursors = analyzer.compute_precursor_scores(df, score_col=score_col)
        if not precursors.empty:
            out_path = SRC_ROOT / "outputs" / "scores" / "early_warning_precursors.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            precursors.to_csv(out_path, index=False)
            logger.info("Précurseurs sauvegardés → %s", out_path)

            # Afficher résumé
            high_ratio = precursors[precursors["ratio"] > 1.3]
            logger.info(
                "Fenêtres avec ratio > 1.3 : %d / %d",
                len(high_ratio), len(precursors),
            )

        # Permutation test (fenêtre 7 jours)
        perm_result = analyzer.permutation_test(df, score_col=score_col, lead=7)
        if perm_result:
            perm_df = pd.DataFrame([perm_result])
            perm_path = SRC_ROOT / "outputs" / "scores" / "early_warning_permutation.csv"
            perm_df.to_csv(perm_path, index=False)
            logger.info("Permutation test sauvegardé → %s", perm_path)

        # Figure timeline
        try:
            fig = analyzer.plot_timeline(
                df,
                output_path=SRC_ROOT / "outputs" / "figures" / "early_warning_timeline.png",
                score_col=score_col,
            )
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as exc:
            logger.warning("Impossible de générer la figure timeline : %s", exc)

        # Analyse multi-seuils
        threshold_df = analyzer.compute_threshold_analysis(df, score_col=score_col)
        if not threshold_df.empty:
            th_path = SRC_ROOT / "outputs" / "scores" / "early_warning_thresholds.csv"
            threshold_df.to_csv(th_path, index=False)
            logger.info("Analyse seuils sauvegardée → %s", th_path)

    except FileNotFoundError as exc:
        logger.warning("Fichier événements manquant : %s", exc)
        logger.info("Créez data/events/merapi_events_2014_2018.csv pour activer l'early warning.")


def step_figures(
    df: pd.DataFrame,
    weather_predictions: pd.DataFrame | None = None,
    precursors_path: "Path | None" = None,
) -> None:
    """
    Étape 6 : Génération des figures d'analyse pour rapport/soutenance.

    Sorties :
      - outputs/figures/precision_recall.png
      - outputs/figures/score_distributions.png
      - outputs/figures/early_warning_ratios.png
      - outputs/figures/svm_confusion.png
    """
    logger.info("─" * 60)
    logger.info("ÉTAPE 6 — Génération des figures")

    from src.evaluation.figures import generate_all_figures
    from src.evaluation.early_warning import TRIGGER_THRESHOLD

    precursors = pd.DataFrame()
    if precursors_path is not None and Path(precursors_path).exists():
        precursors = pd.read_csv(precursors_path)

    score_col = "patchcore_score" if "patchcore_score" in df.columns else "anomaly_score"
    generate_all_figures(
        df=df,
        precursors=precursors if not precursors.empty else None,
        predictions=weather_predictions if (weather_predictions is not None and not weather_predictions.empty) else None,
        score_col=score_col,
        out_dir=SRC_ROOT / "outputs" / "figures",
        trigger_threshold=TRIGGER_THRESHOLD,
    )
    logger.info("Figures générées → outputs/figures/")


# ─── Pipeline complet ─────────────────────────────────────────────────────

def run_pipeline(step: str = "all") -> None:
    """Exécute le pipeline V1 complet ou une étape spécifique."""
    logger.info("=" * 60)
    logger.info("MERAPI V1 PIPELINE — step=%s", step)
    logger.info("=" * 60)

    # Chargement index
    config = load_config()
    index_path = SRC_ROOT / "data" / "index" / "index.csv"
    if not index_path.exists():
        logger.error("index.csv introuvable : %s", index_path)
        logger.error("Exécutez d'abord le scraping et l'indexation.")
        sys.exit(1)

    df = pd.read_csv(index_path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df["day"] = pd.to_numeric(df["day"], errors="coerce")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df["anomaly_score"] = pd.to_numeric(df.get("anomaly_score", pd.Series(dtype=float)), errors="coerce")

    logger.info("Index chargé : %d images (usable=%d)", len(df), (df["quality_flag"] == "usable").sum())

    # Initialiser les résultats avec des valeurs par défaut
    df_features = pd.DataFrame()
    patchcore_scores = pd.Series(dtype=float, name="patchcore_score")
    weather_predictions = pd.DataFrame()

    # ── Étape features ────────────────────────────────────────────────────
    if step in ("all", "features", "weather"):
        df_features = step_features(df)

    # ── Étape PatchCore ───────────────────────────────────────────────────
    if step in ("all", "patchcore"):
        patchcore_scores = step_patchcore(df)
        df["patchcore_score"] = patchcore_scores.values

    # ── Étape Weather SVM ─────────────────────────────────────────────────
    if step in ("all", "weather"):
        if df_features.empty:
            df_features = step_features(df)
        weather_predictions = step_weather(df, df_features)

    # ── Étape Évaluation ──────────────────────────────────────────────────
    if step in ("all", "evaluate"):
        # Charger les scores PatchCore si pas calculés dans cette session
        if patchcore_scores.empty:
            scores_path = SRC_ROOT / "outputs" / "scores" / "patchcore_scores.csv"
            if scores_path.exists():
                sc = pd.read_csv(scores_path)
                # Utiliser map (pas merge) pour éviter l'explosion de lignes
                # due aux filenames dupliqués dans l'index (ex: therm.jpg × 14)
                score_map = sc.drop_duplicates("filename").set_index("filename")["patchcore_score"]
                patchcore_scores = df["filename"].map(score_map).rename("patchcore_score")
                df["patchcore_score"] = patchcore_scores.values
        if weather_predictions.empty:
            pred_path = SRC_ROOT / "outputs" / "scores" / "weather_predictions.csv"
            if pred_path.exists():
                weather_predictions = pd.read_csv(pred_path)
        step_evaluate(df, patchcore_scores, weather_predictions)

    # ── Étape Early Warning ───────────────────────────────────────────────
    if step in ("all", "early_warning"):
        # Recharger df avec patchcore_score si déjà dans index.csv
        df_fresh = pd.read_csv(index_path)
        df_fresh["year"] = pd.to_numeric(df_fresh["year"], errors="coerce")
        df_fresh["hour"] = pd.to_numeric(df_fresh["hour"], errors="coerce")
        if "patchcore_score" not in df_fresh.columns and "patchcore_score" in df.columns:
            df_fresh["patchcore_score"] = df["patchcore_score"].values
        step_early_warning(df_fresh)

    # ── Étape Figures ─────────────────────────────────────────────────────
    if step in ("all", "figures"):
        df_fresh = pd.read_csv(index_path)
        df_fresh["year"] = pd.to_numeric(df_fresh["year"], errors="coerce")
        df_fresh["hour"] = pd.to_numeric(df_fresh["hour"], errors="coerce")
        if "patchcore_score" not in df_fresh.columns and "patchcore_score" in df.columns:
            df_fresh["patchcore_score"] = df["patchcore_score"].values
        if weather_predictions.empty:
            pred_path_fig = SRC_ROOT / "outputs" / "scores" / "weather_predictions.csv"
            if pred_path_fig.exists():
                weather_predictions = pd.read_csv(pred_path_fig)
        step_figures(
            df_fresh,
            weather_predictions=weather_predictions if not weather_predictions.empty else None,
            precursors_path=SRC_ROOT / "outputs" / "scores" / "early_warning_precursors.csv",
        )

    logger.info("=" * 60)
    logger.info("Pipeline V1 terminé.")
    logger.info("Résultats dans outputs/scores/ et outputs/figures/")
    logger.info("=" * 60)


# ─── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline V1 — Merapi Anomaly Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Étapes disponibles :
  all           Exécute tout le pipeline (défaut)
  features      Calcule les features physiques seulement
  patchcore     Entraîne PatchCore et score les images
  weather       Entraîne le classifieur SVM météo/volcanique
  evaluate      Calcule toutes les métriques d'évaluation
  early_warning Analyse les précurseurs d'événements

Exemples :
  python run_v1_pipeline.py                  # pipeline complet
  python run_v1_pipeline.py --step patchcore # seulement PatchCore
  python run_v1_pipeline.py --step evaluate  # seulement les métriques
        """,
    )
    parser.add_argument(
        "--step",
        choices=["all", "features", "patchcore", "weather", "evaluate", "early_warning", "figures"],
        default="all",
        help="Étape à exécuter (défaut: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forcer le recalcul même si les fichiers existent déjà",
    )
    args = parser.parse_args()

    if args.force:
        # Supprimer les artefacts pour forcer le recalcul
        for path in [
            SRC_ROOT / "outputs" / "models" / "physical_features.csv",
            SRC_ROOT / "outputs" / "models" / "patchcore.npz",
            SRC_ROOT / "outputs" / "models" / "weather_svm.pkl",
        ]:
            if path.exists():
                path.unlink()
                logger.info("Supprimé (--force) : %s", path)

    run_pipeline(step=args.step)


if __name__ == "__main__":
    main()
