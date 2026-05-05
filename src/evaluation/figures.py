"""
figures.py — Génération des figures d'analyse pour le rapport / soutenance.

Figures produites :
  1. precision_recall.png  : courbe Precision-Recall PatchCore vs baseline aléatoire
  2. score_distributions.png : histogramme des scores dark vs usable
  3. early_warning_ratios.png : distribution des ratios early warning par fenêtre
  4. svm_confusion.png     : matrice de confusion SVM (optionnelle)

Usage :
    from src.evaluation.figures import generate_all_figures
    generate_all_figures(df, precursors, predictions, out_dir="outputs/figures/")
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("figures")


# ─── 1. Courbe Precision-Recall ───────────────────────────────────────────

def plot_precision_recall(
    df: pd.DataFrame,
    score_cols: list[str] | None = None,
    output_path: str | Path | None = None,
) -> "matplotlib.figure.Figure":
    """
    Courbe Precision-Recall pour un ou plusieurs détecteurs.

    Positifs : quality_flag == 'dark' (proxy activité volcanique)
    Négatifs : quality_flag == 'usable', heure 6–17 (diurne, fond normal)

    Args:
        df: DataFrame avec quality_flag, score_cols, hour.
        score_cols: colonnes de score à tracer. Défaut : ['patchcore_score', 'anomaly_score'].
        output_path: chemin de sauvegarde (None = pas de sauvegarde).
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    if score_cols is None:
        score_cols = [c for c in ["patchcore_score", "anomaly_score"] if c in df.columns]

    # Construire le masque positif/négatif
    hour_col = pd.to_numeric(df.get("hour", pd.Series(12, index=df.index)), errors="coerce").fillna(12)
    mask_pos = df["quality_flag"].eq("dark")
    mask_neg = df["quality_flag"].eq("usable") & hour_col.between(6, 17)
    df_labeled = df[mask_pos | mask_neg].copy()
    df_labeled["y_true"] = mask_pos[mask_pos | mask_neg].astype(int)

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    labels_map = {
        "patchcore_score": "PatchCore (DINOv2)",
        "anomaly_score": "Baseline (anomaly_score)",
    }

    for i, col in enumerate(score_cols):
        sub = df_labeled[df_labeled[col].notna()].copy()
        if len(sub) == 0 or sub["y_true"].sum() == 0:
            logger.warning("Pas de données pour la courbe PR de '%s'.", col)
            continue
        precision, recall, _ = precision_recall_curve(sub["y_true"], sub[col])
        auc = average_precision_score(sub["y_true"], sub[col])
        label = labels_map.get(col, col)
        ax.plot(recall, precision, color=colors[i % len(colors)],
                lw=2, label=f"{label} (AUC-PR={auc:.3f})")

    # Ligne de référence aléatoire
    n_pos = int(df_labeled["y_true"].sum())
    n_total = len(df_labeled)
    if n_total > 0:
        random_ap = n_pos / n_total
        ax.axhline(random_ap, color="gray", ls="--", lw=1,
                   label=f"Aléatoire (AP={random_ap:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Courbe Precision-Recall — Détection d'anomalies volcaniques\n"
                 "Positifs : images dark (proxy incandescence nocturne)", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    _save(fig, output_path, "precision_recall.png")
    return fig


# ─── 2. Histogramme des distributions de scores ───────────────────────────

def plot_score_distributions(
    df: pd.DataFrame,
    score_col: str = "patchcore_score",
    output_path: str | Path | None = None,
) -> "matplotlib.figure.Figure":
    """
    Histogramme comparatif : distribution des scores pour dark vs usable diurne.

    Permet de visualiser la séparabilité (justification de l'AUC-PR).
    """
    import matplotlib.pyplot as plt

    hour_col = pd.to_numeric(df.get("hour", pd.Series(12, index=df.index)), errors="coerce").fillna(12)
    dark_scores = df[df["quality_flag"].eq("dark") & df[score_col].notna()][score_col].values
    usable_scores = df[
        df["quality_flag"].eq("usable") &
        hour_col.between(6, 17) &
        df[score_col].notna()
    ][score_col].values

    if len(dark_scores) == 0 or len(usable_scores) == 0:
        logger.warning("plot_score_distributions : données insuffisantes pour '%s'.", score_col)
        return plt.figure()

    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(
        min(dark_scores.min(), usable_scores.min()),
        max(dark_scores.max(), usable_scores.max()),
        40,
    )
    ax.hist(usable_scores, bins=bins, alpha=0.6, color="#3498db",
            label=f"Usable diurne (N={len(usable_scores)})", density=True)
    ax.hist(dark_scores, bins=bins, alpha=0.7, color="#e74c3c",
            label=f"Dark / activité (N={len(dark_scores)})", density=True)

    # Lignes médianes
    ax.axvline(np.median(usable_scores), color="#3498db", ls="--", lw=1.5,
               label=f"Médiane usable={np.median(usable_scores):.3f}")
    ax.axvline(np.median(dark_scores), color="#e74c3c", ls="--", lw=1.5,
               label=f"Médiane dark={np.median(dark_scores):.3f}")

    ax.set_xlabel(f"Score d'anomalie ({score_col})", fontsize=12)
    ax.set_ylabel("Densité", fontsize=12)
    ax.set_title(f"Distribution des scores — dark vs usable diurne\n({score_col})", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    _save(fig, output_path, "score_distributions.png")
    return fig


# ─── 3. Distribution des ratios early warning ─────────────────────────────

def plot_early_warning_ratios(
    precursors: pd.DataFrame,
    trigger_threshold: float = 1.1,
    output_path: str | Path | None = None,
) -> "matplotlib.figure.Figure":
    """
    Boîte à moustaches + strip plot des ratios par fenêtre temporelle.

    Visualise la concentration des ratios élevés sur certains événements
    et compare les différentes durées de fenêtre.
    """
    import matplotlib.pyplot as plt

    if precursors.empty or "ratio" not in precursors.columns:
        logger.warning("plot_early_warning_ratios : DataFrame vide ou colonne 'ratio' absente.")
        return plt.figure()

    lead_vals = sorted(precursors["lead_days"].unique())
    data_by_lead = [precursors[precursors["lead_days"] == l]["ratio"].dropna().values for l in lead_vals]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Gauche : boîte à moustaches par fenêtre ──
    ax = axes[0]
    bp = ax.boxplot(data_by_lead, labels=[f"{l}j" for l in lead_vals],
                    patch_artist=True, notch=False, showfliers=True)
    colors_box = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c"]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(trigger_threshold, color="red", ls="--", lw=1.5,
               label=f"Seuil {trigger_threshold}")
    ax.set_xlabel("Fenêtre temporelle", fontsize=11)
    ax.set_ylabel("Ratio score précurseur / background", fontsize=11)
    ax.set_title("Distribution des ratios early warning\npar fenêtre pré-événement", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ── Droite : histogramme des ratios (toutes fenêtres confondues) ──
    ax2 = axes[1]
    all_ratios = precursors["ratio"].dropna().values
    ax2.hist(all_ratios, bins=20, color="#9b59b6", alpha=0.75, edgecolor="white")
    ax2.axvline(trigger_threshold, color="red", ls="--", lw=1.5,
                label=f"Seuil {trigger_threshold}")
    ax2.axvline(float(np.median(all_ratios)), color="orange", ls="-", lw=1.5,
                label=f"Médiane={np.median(all_ratios):.2f}")
    n_above = int((all_ratios >= trigger_threshold).sum())
    ax2.set_title(f"Histogramme des ratios (toutes fenêtres)\n"
                  f"{n_above}/{len(all_ratios)} triggers ≥ {trigger_threshold}", fontsize=11)
    ax2.set_xlabel("Ratio", fontsize=11)
    ax2.set_ylabel("Nombre de fenêtres", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, output_path, "early_warning_ratios.png")
    return fig


# ─── 4. Matrice de confusion SVM ──────────────────────────────────────────

def plot_confusion_matrix(
    df_index: pd.DataFrame,
    predictions: pd.DataFrame,
    output_path: str | Path | None = None,
) -> "matplotlib.figure.Figure":
    """
    Matrice de confusion normalisée du classifieur météo/volcanique.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    labeled = df_index[df_index["quality_flag"].isin(["cloudy", "dark"])].copy()
    labeled["y_true"] = labeled["quality_flag"].map({"cloudy": 0, "dark": 1})
    merged = labeled.merge(predictions[["filename", "weather_label"]], on="filename", how="inner")

    if merged.empty:
        logger.warning("plot_confusion_matrix : aucune donnée labellisée avec prédiction.")
        return plt.figure()

    cm = confusion_matrix(merged["y_true"], merged["weather_label"], normalize="true")
    disp = ConfusionMatrixDisplay(cm, display_labels=["Météo (cloudy)", "Volcanique (dark)"])

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2f")
    ax.set_title("Matrice de confusion — SVM triage météo/volcanique\n(normalisée par classe réelle)",
                 fontsize=11)
    plt.tight_layout()
    _save(fig, output_path, "svm_confusion.png")
    return fig


# ─── Entrée unique ─────────────────────────────────────────────────────────

def generate_all_figures(
    df: pd.DataFrame,
    precursors: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
    score_col: str = "patchcore_score",
    out_dir: str | Path = "outputs/figures/",
    trigger_threshold: float = 1.1,
) -> None:
    """
    Génère et sauvegarde toutes les figures d'analyse dans out_dir.

    Args:
        df: DataFrame principal avec scores et quality_flag.
        precursors: DataFrame des précurseurs early warning (optionnel).
        predictions: DataFrame des prédictions SVM (optionnel).
        score_col: colonne de score principale.
        out_dir: répertoire de sortie.
        trigger_threshold: seuil ratio early warning pour annotations.
    """
    import matplotlib
    matplotlib.use("Agg")  # backend non-interactif (sans affichage)
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Génération des figures dans %s ...", out_dir)

    # 1. Precision-Recall
    try:
        cols = [c for c in [score_col, "anomaly_score"] if c in df.columns]
        if cols:
            fig = plot_precision_recall(df, score_cols=cols, output_path=out_dir)
            plt.close(fig)
            logger.info("✓ precision_recall.png")
    except Exception as exc:
        logger.warning("Erreur figure PR : %s", exc)

    # 2. Score distributions
    if score_col in df.columns:
        try:
            fig = plot_score_distributions(df, score_col=score_col, output_path=out_dir)
            plt.close(fig)
            logger.info("✓ score_distributions.png")
        except Exception as exc:
            logger.warning("Erreur figure distributions : %s", exc)

    # 3. Early warning ratios
    if precursors is not None and not precursors.empty:
        try:
            fig = plot_early_warning_ratios(precursors, trigger_threshold=trigger_threshold,
                                            output_path=out_dir)
            plt.close(fig)
            logger.info("✓ early_warning_ratios.png")
        except Exception as exc:
            logger.warning("Erreur figure early warning ratios : %s", exc)

    # 4. Confusion matrix
    if predictions is not None and not predictions.empty:
        try:
            fig = plot_confusion_matrix(df, predictions, output_path=out_dir)
            plt.close(fig)
            logger.info("✓ svm_confusion.png")
        except Exception as exc:
            logger.warning("Erreur figure confusion matrix : %s", exc)

    logger.info("Figures sauvegardées dans %s", out_dir)


# ─── helper ───────────────────────────────────────────────────────────────

def _save(fig: "matplotlib.figure.Figure", output_path, default_name: str) -> None:
    """Sauvegarde la figure dans output_path (répertoire ou chemin complet)."""
    if output_path is None:
        return
    p = Path(output_path)
    if p.is_dir() or not p.suffix:
        p = p / default_name
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    logger.info("Figure sauvegardée → %s", p)
