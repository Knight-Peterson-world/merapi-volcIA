"""
metrics.py — Métriques d'évaluation scientifique pour le pipeline V1.

Métriques implémentées :
  1. AUC-PR (Average Precision) — détection d'anomalies
     Calculable sans labels parfaits via proxy quality_flag == 'dark'.
     AUC-PR est plus informatif que AUC-ROC sur des classes déséquilibrées.

  2. F1-score triage météo/volcanique
     Calculé sur les images labellisées (cloudy / dark).
     Mesure directement la réduction des faux positifs.

  3. Comparaison baseline vs PatchCore
     Delta AUC-PR entre les deux méthodes = argument quantitatif central.

Usage :
    from src.evaluation.metrics import evaluate_patchcore, EvaluationReport

    report = evaluate_patchcore(df_with_scores)
    report.print_summary()
    report.save("outputs/scores/evaluation_report.csv")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("metrics")


# ─── AUC-PR ───────────────────────────────────────────────────────────────

def compute_auc_pr(
    df: pd.DataFrame,
    score_col: str = "anomaly_score",
    positive_flags: list[str] | None = None,
) -> dict[str, float]:
    """
    Calcule l'AUC-PR avec un proxy de vérité terrain basé sur quality_flag.

    Stratégie sans labels manuels :
      - Positifs (anomalie) : images 'dark' (activité nocturne = signal thermique)
      - Négatifs  : images 'usable' diurnes (fond normal)

    Cette définition est conservative et reproductible, même sans annotation manuelle.
    Elle est défendable scientifiquement comme "proxy-labeled" evaluation.

    Args:
        df: DataFrame avec colonnes quality_flag, score_col, hour.
        score_col: nom de la colonne de score.
        positive_flags: flags considérés comme positifs. Défaut : ['dark'].

    Returns:
        dict avec auc_pr, n_positive, n_negative, random_baseline.
    """
    from sklearn.metrics import average_precision_score

    if positive_flags is None:
        positive_flags = ["dark"]

    df_eval = df[df["quality_flag"].notna()].copy()
    df_eval = df_eval[df_eval[score_col].notna()]

    # Positifs : images dark (incandescence nocturne / activité)
    # Négatifs : images usable diurnes (fond normal)
    mask_pos = df_eval["quality_flag"].isin(positive_flags)
    mask_neg = df_eval["quality_flag"].eq("usable") & df_eval.get("hour", pd.Series(12, index=df_eval.index)).between(6, 17)

    df_labeled = df_eval[mask_pos | mask_neg].copy()
    df_labeled["y_true"] = mask_pos[mask_pos | mask_neg].astype(int)

    n_pos = int(df_labeled["y_true"].sum())
    n_neg = int((df_labeled["y_true"] == 0).sum())

    if n_pos == 0 or n_neg == 0:
        logger.warning("AUC-PR non calculable : n_pos=%d, n_neg=%d", n_pos, n_neg)
        return {"auc_pr": float("nan"), "n_positive": n_pos, "n_negative": n_neg, "random_baseline": float("nan")}

    y_true = df_labeled["y_true"].values
    y_score = df_labeled[score_col].values

    auc_pr = float(average_precision_score(y_true, y_score))
    random_baseline = float(n_pos / (n_pos + n_neg))  # AUC-PR d'un classifieur aléatoire

    logger.info(
        "AUC-PR (%s) : %.4f  [baseline aléatoire : %.4f]  (n+=%-4d n-=%-4d)",
        score_col, auc_pr, random_baseline, n_pos, n_neg,
    )
    return {
        "auc_pr": auc_pr,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "random_baseline": random_baseline,
    }


# ─── F1 binaire PatchCore (proxy ground-truth quality_flag) ─────────────────

def compute_f1_patchcore_binary(
    df: pd.DataFrame,
    score_col: str = "patchcore_score",
    threshold_pct: int = 90,
) -> dict[str, float]:
    """
    Calcule le F1 binaire : anomalie (dark) vs normal (usable).

    Stratégie :
      - y_true = 1 si quality_flag == 'dark',  0 si quality_flag == 'usable'
      - y_pred = 1 si score > percentile threshold_pct,  0 sinon
      - Utilise uniquement les images qui ont un score PatchCore valide.

    Ce calcul ne dépend PAS de weather_predictions, donc toujours calculable
    après --step evaluate (tant que --step patchcore a tourné).

    Args:
        df: DataFrame avec quality_flag + score_col.
        score_col: colonne de score à seuiller.
        threshold_pct: percentile global pour binariser les scores (défaut 90).

    Returns:
        dict avec f1_macro, f1_anomaly, f1_normal, threshold, accuracy, n_samples.
    """
    from sklearn.metrics import f1_score, accuracy_score

    df_bin = df[
        df["quality_flag"].isin(["dark", "usable"]) & df[score_col].notna()
    ].copy()

    if df_bin.empty:
        logger.warning("compute_f1_patchcore_binary : aucune image dark/usable avec score.")
        return {}

    y_true = (df_bin["quality_flag"] == "dark").astype(int).values
    threshold = float(np.percentile(df_bin[score_col].values, threshold_pct))
    y_pred = (df_bin[score_col].values > threshold).astype(int)

    n_pos = int(y_true.sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        logger.warning(
            "compute_f1_patchcore_binary : une classe absente (n_pos=%d, n_neg=%d).",
            n_pos, n_neg,
        )
        return {}

    f1_macro = float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0))
    f1_anom  = float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0))
    f1_norm  = float(f1_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0))
    acc      = float(accuracy_score(y_true, y_pred))

    logger.info(
        "F1 binaire PatchCore — macro=%.4f  anomalie=%.4f  normal=%.4f  "
        "(seuil P%d=%.4f, n=%d : %d dark + %d usable)",
        f1_macro, f1_anom, f1_norm, threshold_pct, threshold, len(y_true), n_pos, n_neg,
    )
    return {
        "f1_macro": f1_macro,
        "f1_anomaly": f1_anom,
        "f1_normal": f1_norm,
        "threshold": threshold,
        "accuracy": acc,
        "n_samples": len(y_true),
        "n_positive": n_pos,
        "n_negative": n_neg,
    }


# ─── F1 triage ─────────────────────────────────────────────────────────────

def compute_f1_triage(
    df_index: pd.DataFrame,
    df_predictions: pd.DataFrame,
) -> dict[str, float]:
    """
    Calcule le F1-score du triage météo vs volcanique.

    Args:
        df_index: DataFrame avec quality_flag (cloudy/dark).
        df_predictions: DataFrame avec columns ['filename', 'weather_label'].

    Returns:
        dict avec f1_macro, f1_weather, f1_volcanic, accuracy, n_samples.

    Note sur f1_macro :
        On passe explicitement labels=[0, 1] pour que sklearn inclue TOUJOURS
        les deux classes dans la moyenne, même si l'une est absente de y_true.
        Sans cela, sklearn calcule la moyenne seulement sur les classes présentes,
        ce qui donnait f1_macro=1.0 quand seule la classe 1 (dark) était représentée.
    """
    from sklearn.metrics import f1_score, accuracy_score

    labeled = df_index[df_index["quality_flag"].isin(["cloudy", "dark"])].copy()
    labeled["y_true"] = labeled["quality_flag"].map({"cloudy": 0, "dark": 1})
    # Supprimer weather_label si déjà présent (évite les colonnes _x/_y lors du merge)
    labeled = labeled.drop(columns=["weather_label"], errors="ignore")

    merged = labeled.merge(df_predictions[["filename", "weather_label"]], on="filename", how="inner")
    if merged.empty:
        logger.warning("Aucune image labellisée avec prédiction disponible.")
        return {}

    y_true = merged["y_true"].values
    y_pred = merged["weather_label"].values

    # labels=[0, 1] force l'inclusion des deux classes même si absentes de y_true
    f1_macro_corrected = float(
        f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)
    )
    f1_w = float(f1_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0))
    f1_v = float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0))

    # Validation interne : macro doit être la moyenne des deux
    expected_macro = (f1_w + f1_v) / 2
    if abs(f1_macro_corrected - expected_macro) > 1e-6:
        logger.warning(
            "Incohérence F1-macro : calculé=%.4f, attendu=%.4f — vérifier les labels.",
            f1_macro_corrected, expected_macro,
        )

    logger.info(
        "F1 triage — macro=%.4f (weather=%.4f, volcanic=%.4f), accuracy=%.4f, n=%d",
        f1_macro_corrected, f1_w, f1_v,
        float(accuracy_score(y_true, y_pred)), len(y_true),
    )

    return {
        "f1_macro": f1_macro_corrected,
        "f1_weather": f1_w,
        "f1_volcanic": f1_v,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n_samples": len(y_true),
    }


# ─── Comparaison baseline vs PatchCore ─────────────────────────────────────

def compare_detectors(
    df: pd.DataFrame,
    baseline_col: str = "anomaly_score",
    patchcore_col: str = "patchcore_score",
) -> dict[str, float]:
    """
    Compare les AUC-PR de deux détecteurs.

    Returns:
        dict avec baseline_auc_pr, patchcore_auc_pr, delta, improvement_pct.
    """
    baseline_metrics = compute_auc_pr(df, score_col=baseline_col)
    patchcore_metrics = compute_auc_pr(df, score_col=patchcore_col)

    delta = patchcore_metrics["auc_pr"] - baseline_metrics["auc_pr"]
    improvement_pct = (
        delta / baseline_metrics["auc_pr"] * 100
        if baseline_metrics["auc_pr"] > 0 else float("nan")
    )

    logger.info(
        "Comparaison — Baseline AUC-PR: %.4f | PatchCore AUC-PR: %.4f | Δ: %+.4f (%.1f%%)",
        baseline_metrics["auc_pr"], patchcore_metrics["auc_pr"], delta, improvement_pct,
    )
    return {
        "baseline_auc_pr": baseline_metrics["auc_pr"],
        "patchcore_auc_pr": patchcore_metrics["auc_pr"],
        "delta": delta,
        "improvement_pct": improvement_pct,
        "n_positive": baseline_metrics["n_positive"],
        "n_negative": baseline_metrics["n_negative"],
        "random_baseline": baseline_metrics["random_baseline"],
    }


# ─── AUC-ROC ─────────────────────────────────────────────────────────

def compute_auc_roc(
    df: pd.DataFrame,
    score_col: str = "patchcore_score",
    positive_flags: list[str] | None = None,
) -> dict[str, float]:
    """
    Calcule l'AUC-ROC avec le même proxy de vérité terrain que compute_auc_pr.

    Moins sensible au déséquilibre de classes que l'AUC-PR, mais complémentaire :
    AUC-ROC > 0.7 confirme que le modèle discrimine bien les deux classes.

    Args:
        df: DataFrame avec colonnes quality_flag, score_col, hour.
        score_col: colonne de score.
        positive_flags: flags positifs. Défaut : ['dark'].

    Returns:
        dict avec auc_roc, n_positive, n_negative.
    """
    from sklearn.metrics import roc_auc_score

    if positive_flags is None:
        positive_flags = ["dark"]

    df_eval = df[df["quality_flag"].notna()].copy()
    df_eval = df_eval[df_eval[score_col].notna()]

    mask_pos = df_eval["quality_flag"].isin(positive_flags)
    mask_neg = (
        df_eval["quality_flag"].eq("usable")
        & df_eval.get("hour", pd.Series(12, index=df_eval.index)).between(6, 17)
    )
    df_labeled = df_eval[mask_pos | mask_neg].copy()
    df_labeled["y_true"] = mask_pos[mask_pos | mask_neg].astype(int)

    n_pos = int(df_labeled["y_true"].sum())
    n_neg = int((df_labeled["y_true"] == 0).sum())

    if n_pos == 0 or n_neg == 0:
        logger.warning("AUC-ROC non calculable : n_pos=%d, n_neg=%d", n_pos, n_neg)
        return {"auc_roc": float("nan"), "n_positive": n_pos, "n_negative": n_neg}

    auc_roc = float(roc_auc_score(df_labeled["y_true"].values, df_labeled[score_col].values))
    logger.info("AUC-ROC (%s) : %.4f  (n+=%-4d n-=%-4d)", score_col, auc_roc, n_pos, n_neg)
    return {"auc_roc": auc_roc, "n_positive": n_pos, "n_negative": n_neg}


# ─── Mann-Whitney test (validation du signal vs bruit) ────────────────────

def test_signal_significance(
    df: pd.DataFrame,
    score_col: str = "patchcore_score",
) -> dict[str, float]:
    """
    Test de Mann-Whitney pour vérifier que la distribution des scores
    des images 'dark' est significativement différente des images 'usable'.

    Défendable comme validation statistique non-paramétrique.

    Returns:
        dict avec statistic, p_value, effect_size (rank-biserial correlation).
    """
    from scipy.stats import mannwhitneyu

    dark_scores = df[df["quality_flag"] == "dark"][score_col].dropna().values
    usable_scores = df[
        (df["quality_flag"] == "usable") &
        df.get("hour", pd.Series(12, index=df.index)).between(6, 17)
    ][score_col].dropna().values

    if len(dark_scores) < 5 or len(usable_scores) < 5:
        logger.warning("Pas assez de données pour le test Mann-Whitney.")
        return {}

    stat, p_value = mannwhitneyu(dark_scores, usable_scores, alternative="greater")
    n1, n2 = len(dark_scores), len(usable_scores)
    # Rank-biserial correlation (convention standard) :
    #   +1 = dark toujours > usable  |  -1 = dark toujours < usable  |  0 = équivalent
    # Note : avec alternative='greater', U est le comptage des paires (dark > usable).
    # Un effet positif confirme que PatchCore attribue bien des scores plus hauts aux anomalies.
    effect_size = (2 * stat) / (n1 * n2) - 1  # standard rank-biserial correlation

    logger.info(
        "Mann-Whitney — U=%.0f, p=%.4f, effect_size=%.3f (n_dark=%d, n_usable=%d)",
        stat, p_value, effect_size, n1, n2,
    )
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "effect_size": float(effect_size),
        "n_dark": n1,
        "n_usable": n2,
        "significant": bool(p_value < 0.05),
    }


# ─── Rapport d'évaluation complet ─────────────────────────────────────────

@dataclass
class EvaluationReport:
    """
    Rapport d'évaluation complet regroupant toutes les métriques V1.

    Attributs sauvegardés en CSV et affichables dans Streamlit.
    """
    auc_pr_baseline: float = float("nan")
    auc_pr_patchcore: float = float("nan")
    auc_roc_patchcore: float = float("nan")  # complément AUC-PR, moins sensible au déséquilibre
    delta_auc_pr: float = float("nan")
    improvement_pct: float = float("nan")
    # F1 binaire PatchCore (dark vs usable, seuil P90) — toujours calculable
    f1_macro_evaluate: float = float("nan")
    f1_anomaly: float = float("nan")
    f1_normal: float = float("nan")
    # F1 triage météo (legacy — dépend de weather_predictions, souvent NaN)
    f1_macro_triage: float = float("nan")
    f1_volcanic: float = float("nan")
    f1_weather: float = float("nan")
    mann_whitney_p: float = float("nan")
    effect_size: float = float("nan")
    n_train: int = 0
    n_test: int = 0
    n_positive_proxy: int = 0
    extra: dict = field(default_factory=dict)

    def print_summary(self) -> None:
        """Affiche un résumé formaté dans les logs."""
        logger.info("=" * 60)
        logger.info("RAPPORT D'ÉVALUATION V1")
        logger.info("=" * 60)
        logger.info("DÉTECTION D'ANOMALIES")
        logger.info("  Baseline AUC-PR    : %.4f", self.auc_pr_baseline)
        logger.info("  PatchCore AUC-PR   : %.4f", self.auc_pr_patchcore)
        logger.info("  PatchCore AUC-ROC  : %.4f", self.auc_roc_patchcore)
        logger.info("  Amélioration       : %+.4f (%.1f%%)", self.delta_auc_pr, self.improvement_pct)
        logger.info("")
        logger.info("F1 BINAIRE PATCHCORE (evaluate)")
        logger.info("  F1-macro           : %.4f", self.f1_macro_evaluate)
        logger.info("  F1 anomalie (dark) : %.4f", self.f1_anomaly)
        logger.info("  F1 normal (usable) : %.4f", self.f1_normal)
        logger.info("TRIAGE MÉTÉO / VOLCANIQUE (legacy)")
        logger.info("  F1-macro (corrigé) : %.4f  [= (f1_w + f1_v) / 2]", self.f1_macro_triage)
        logger.info("  F1 volcanique      : %.4f", self.f1_volcanic)
        logger.info("  F1 météo           : %.4f", self.f1_weather)
        logger.info("")
        logger.info("SIGNIFICANCE STATISTIQUE")
        logger.info("  Mann-Whitney p     : %.4e %s", self.mann_whitney_p,
                    "✓ significatif" if self.mann_whitney_p < 0.05 else "✗ non-significatif")
        logger.info("  Effect size        : %.3f", self.effect_size)
        logger.info("=" * 60)

    def to_dataframe(self) -> pd.DataFrame:
        """Retourne un DataFrame 1 ligne pour export CSV."""
        return pd.DataFrame([{
            "auc_pr_baseline": self.auc_pr_baseline,
            "auc_pr_patchcore": self.auc_pr_patchcore,
            "auc_roc_patchcore": self.auc_roc_patchcore,
            "delta_auc_pr": self.delta_auc_pr,
            "improvement_pct": self.improvement_pct,
            "f1_macro_evaluate": self.f1_macro_evaluate,
            "f1_anomaly": self.f1_anomaly,
            "f1_normal": self.f1_normal,
            "f1_macro_triage": self.f1_macro_triage,
            "f1_volcanic": self.f1_volcanic,
            "f1_weather": self.f1_weather,
            "mann_whitney_p": self.mann_whitney_p,
            "effect_size": self.effect_size,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_positive_proxy": self.n_positive_proxy,
        }])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)
        logger.info("Rapport sauvegardé → %s", path)


def evaluate_patchcore(
    df_with_scores: pd.DataFrame,
    df_predictions: pd.DataFrame | None = None,
) -> EvaluationReport:
    """
    Calcule toutes les métriques V1 et retourne un EvaluationReport.

    Args:
        df_with_scores: DataFrame avec colonnes anomaly_score et patchcore_score.
        df_predictions: DataFrame des prédictions triage (optionnel).
    """
    report = EvaluationReport()

    # 1. AUC-PR baseline = classifieur aléatoire (AUC-PR = prévalence des positifs).
    #
    # Raisonnement : l'anomaly_score existant est une copie de patchcore_score
    # (voir load_index), donc utiliser anomaly_score comme baseline biaiserait
    # le delta à 0. La vraie baseline de référence est le classifieur aléatoire,
    # dont l'AUC-PR = prévalence = n_positifs / n_total (théorème de Davis & Goadrich).
    patchcore_metrics_for_baseline = compute_auc_pr(df_with_scores, "patchcore_score")
    report.n_positive_proxy = patchcore_metrics_for_baseline["n_positive"]
    random_baseline_auc_pr = patchcore_metrics_for_baseline["random_baseline"]
    if not np.isnan(random_baseline_auc_pr):
        report.auc_pr_baseline = random_baseline_auc_pr
        logger.info(
            "Baseline aléatoire AUC-PR = %.4f (prévalence : %d / %d)",
            random_baseline_auc_pr,
            patchcore_metrics_for_baseline["n_positive"],
            patchcore_metrics_for_baseline["n_positive"] + patchcore_metrics_for_baseline["n_negative"],
        )
    else:
        logger.warning(
            "Baseline aléatoire non calculable (pas assez d'images dark/usable)."
        )

    # 2. AUC-PR PatchCore
    if "patchcore_score" in df_with_scores.columns:
        patchcore = compute_auc_pr(df_with_scores, "patchcore_score")
        report.auc_pr_patchcore = patchcore["auc_pr"]
        # n_positive_proxy prioritise le score PatchCore si le baseline est absent
        if report.n_positive_proxy == 0:
            report.n_positive_proxy = patchcore["n_positive"]

    # 3. AUC-ROC PatchCore (complément à l'AUC-PR, moins sensible au déséquilibre)
    if "patchcore_score" in df_with_scores.columns:
        roc = compute_auc_roc(df_with_scores, "patchcore_score")
        report.auc_roc_patchcore = roc.get("auc_roc", float("nan"))

    # 4. Delta
    if not (np.isnan(report.auc_pr_baseline) or np.isnan(report.auc_pr_patchcore)):
        report.delta_auc_pr = report.auc_pr_patchcore - report.auc_pr_baseline
        if report.auc_pr_baseline > 0:
            report.improvement_pct = report.delta_auc_pr / report.auc_pr_baseline * 100

    # 5. Significance
    score_col = "patchcore_score" if "patchcore_score" in df_with_scores.columns else "anomaly_score"
    sig = test_signal_significance(df_with_scores, score_col=score_col)
    if sig:
        report.mann_whitney_p = sig.get("p_value", float("nan"))
        report.effect_size = sig.get("effect_size", float("nan"))

    # 6a. F1 binaire PatchCore — toujours calculable (dark vs usable, seuil P90)
    score_col_f1 = "patchcore_score" if "patchcore_score" in df_with_scores.columns else "anomaly_score"
    f1_bin = compute_f1_patchcore_binary(df_with_scores, score_col=score_col_f1)
    if f1_bin:
        report.f1_macro_evaluate = f1_bin.get("f1_macro", float("nan"))
        report.f1_anomaly = f1_bin.get("f1_anomaly", float("nan"))
        report.f1_normal  = f1_bin.get("f1_normal",  float("nan"))

    # 6b. F1 triage météo (legacy — dépend du merge weather_predictions × quality_flag)
    if df_predictions is not None and not df_predictions.empty:
        triage = compute_f1_triage(df_with_scores, df_predictions)
        report.f1_macro_triage = triage.get("f1_macro", float("nan"))
        report.f1_volcanic = triage.get("f1_volcanic", float("nan"))
        report.f1_weather = triage.get("f1_weather", float("nan"))

    # 7. Split stats
    report.n_train = int(
        (df_with_scores["year"].isin(range(2014, 2018)) & df_with_scores["quality_flag"].eq("usable")).sum()
    )

    # n_test = images 2018 ayant un score PatchCore (images effectivement scorées,
    # pas seulement "usable dans l'index"). Reflète la couverture réelle du pipeline.
    _best_score_col = next(
        (c for c in ["patchcore_score", "anomaly_score"]
         if c in df_with_scores.columns and df_with_scores[c].notna().any()),
        None,
    )
    if _best_score_col:
        report.n_test = int(
            (df_with_scores["year"].eq(2018) & df_with_scores[_best_score_col].notna()).sum()
        )
    else:
        report.n_test = 0

    if report.n_test == 0:
        logger.warning(
            "⚠️  n_test = 0 : aucune image 2018 n'a de score PatchCore.\n"
            "   Cause probable : images 2018 non téléchargées dans data/raw/2018/.\n"
            "   Pour les scorer : téléchargez les images et relancez\n"
            "     python run_v1_pipeline.py --step patchcore"
        )

    if report.n_positive_proxy == 0:
        logger.warning(
            "⚠️  n_positive_proxy = 0 : aucune image 'dark' scorée trouvée.\n"
            "   AUC-PR non significative — proxy d'anomalies insuffisant."
        )

    report.print_summary()
    return report
