"""
early_warning.py — Protocole d'alerte précoce (early warning).

Principe :
  Pour chaque événement volcanique documenté, calculer le score d'anomalie
  moyen dans les fenêtres précédant l'événement (J-1, J-3, J-7).
  Comparer avec le score de fond (background) sur des fenêtres aléatoires.
  → Si ratio > 1.5 et significatif (permutation test) : signal précurseur détecté.

Données événementielles :
  Fichier CSV : data/events/merapi_events_2014_2018.csv
  Colonnes : date, type, intensity, source, description

Usage :
    from src.evaluation.early_warning import EarlyWarningAnalyzer

    analyzer = EarlyWarningAnalyzer()
    results = analyzer.compute_precursor_scores(df_index)
    analyzer.permutation_test(df_index)
    analyzer.plot_timeline(df_index, "outputs/figures/early_warning_timeline.png")
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("early_warning")

# Chemin par défaut du fichier d'événements
EVENTS_CSV_DEFAULT = Path(__file__).resolve().parents[2] / "data" / "events" / "merapi_events_2014_2018.csv"

# Fenêtres de précurseur à analyser (en jours)
LEAD_DAYS = [3, 7, 14]

# Seuil de détection d'un précurseur (ratio score_précurseur / background)
# Abaissé à 1.1 pour être sensible aux signaux faibles
TRIGGER_THRESHOLD = 1.1

# Nombre de permutations pour le test de significativité
N_PERMUTATIONS = 1000


class EarlyWarningAnalyzer:
    """
    Analyse la capacité des scores d'anomalie à détecter les précurseurs d'événements.
    """

    def __init__(self, events_path: str | Path | None = None) -> None:
        if events_path is None:
            events_path = EVENTS_CSV_DEFAULT
        self.events_path = Path(events_path)
        self._events: pd.DataFrame | None = None

    def load_events(self) -> pd.DataFrame:
        """Charge le CSV des événements volcaniques documentés."""
        if self._events is not None:
            return self._events

        if not self.events_path.exists():
            raise FileNotFoundError(
                f"Fichier d'événements introuvable : {self.events_path}\n"
                "Créez data/events/merapi_events_2014_2018.csv ou fournissez events_path."
            )

        df = pd.read_csv(self.events_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        self._events = df
        logger.info("Événements chargés : %d événements (2014–2018)", len(df))
        return df

    def compute_precursor_scores(
        self,
        df_index: pd.DataFrame,
        score_col: str = "patchcore_score",
        lead_days: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Pour chaque événement, calcule le score moyen dans la fenêtre pré-événement.

        Args:
            df_index: DataFrame avec colonnes year, month, day, hour, score_col, quality_flag.
            score_col: colonne de score à analyser.
            lead_days: liste des horizons en jours (défaut: [1, 3, 7]).

        Returns:
            DataFrame avec colonnes :
              event, event_date, event_type, lead_days,
              mean_score, max_score, n_images, background_score, ratio
        """
        if lead_days is None:
            lead_days = LEAD_DAYS

        events = self.load_events()
        df = self._prepare_df(df_index, score_col)

        if df.empty:
            logger.warning("Aucune image avec score '%s' disponible.", score_col)
            return pd.DataFrame()

        # Score de fond : médiane globale sur images diurnes usable
        bg_mask = df["quality_flag"].eq("usable") & df["hour"].between(6, 17)
        background_score = float(df[bg_mask][score_col].median())
        logger.info("Score background (médiane diurne) : %.4f", background_score)

        # Distribution globale pour référence
        bg_all = df[bg_mask][score_col].dropna()
        p50 = float(bg_all.quantile(0.50)) if len(bg_all) > 0 else float("nan")
        p90 = float(bg_all.quantile(0.90)) if len(bg_all) > 0 else float("nan")
        logger.info("Distribution background — P50=%.4f, P90=%.4f, N=%d", p50, p90, len(bg_all))

        records = []
        for _, event in events.iterrows():
            event_date = pd.to_datetime(event["date"])

            for lead in lead_days:
                window_start = event_date - pd.Timedelta(days=lead)
                window_end = event_date

                window_mask = (
                    (df["datetime"] >= window_start) &
                    (df["datetime"] < window_end) &
                    df["quality_flag"].eq("usable")
                )
                window_data = df[window_mask][score_col].dropna()

                if len(window_data) < 2:
                    continue

                # Quantile 0.9 plutôt que la moyenne : capte les pics d'activité
                # sans être lissé par les images normales de la fenêtre
                q90_window = float(window_data.quantile(0.90))
                ratio = float(q90_window / background_score) if background_score > 0 else float("nan")

                records.append({
                    "event": str(event.get("description", event.get("type", "unknown"))),
                    "event_date": event["date"],
                    "event_type": str(event.get("type", "unknown")),
                    "event_intensity": str(event.get("intensity", "unknown")),
                    "lead_days": lead,
                    "mean_score": float(window_data.mean()),
                    "q90_score": q90_window,
                    "max_score": float(window_data.max()),
                    "n_images": len(window_data),
                    "background_score": background_score,
                    "ratio": ratio,
                    "triggered": ratio >= TRIGGER_THRESHOLD,
                    "score_col": score_col,
                })

        if not records:
            logger.warning("Aucune fenêtre pré-événement avec données suffisantes.")
            return pd.DataFrame()

        results = pd.DataFrame(records)
        n_triggers = int(results["triggered"].sum())
        ratio_max = float(results["ratio"].max())
        logger.info(
            "Précurseurs calculés : %d entrées | ratio max=%.3f | triggers (ratio>=%.1f) : %d/%d",
            len(results), ratio_max, TRIGGER_THRESHOLD, n_triggers, len(results),
        )
        if n_triggers > 0:
            logger.info(
                "Fenêtres déclenchées :\n%s",
                results[results["triggered"]][["event_date", "lead_days", "ratio", "n_images"]].to_string(index=False),
            )
        return results

    def compute_threshold_analysis(
        self,
        df_index: pd.DataFrame,
        score_col: str = "patchcore_score",
        thresholds: list | None = None,
        lead_days: list | None = None,
    ) -> pd.DataFrame:
        """
        Analyse le taux de déclenchement pour plusieurs seuils de ratio.

        Utile pour calibrer le seuil optimal selon le compromis
        sensibilité / faux positifs.

        Returns:
            DataFrame avec colonnes :
              threshold, lead_days, n_events, n_triggers, trigger_rate,
              ratio_mean, ratio_median, ratio_max
        """
        if thresholds is None:
            thresholds = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        if lead_days is None:
            lead_days = LEAD_DAYS

        precursors = self.compute_precursor_scores(df_index, score_col=score_col, lead_days=lead_days)
        if precursors.empty:
            logger.warning("compute_threshold_analysis : aucun précurseur disponible.")
            return pd.DataFrame()

        records = []
        for thr in thresholds:
            for lead in lead_days:
                sub = precursors[precursors["lead_days"] == lead]
                if sub.empty:
                    continue
                n_trig = int((sub["ratio"] >= thr).sum())
                records.append({
                    "threshold":    thr,
                    "lead_days":    lead,
                    "n_events":     len(sub),
                    "n_triggers":   n_trig,
                    "trigger_rate": round(n_trig / len(sub), 3) if len(sub) > 0 else 0.0,
                    "ratio_mean":   round(float(sub["ratio"].mean()), 4),
                    "ratio_median": round(float(sub["ratio"].median()), 4),
                    "ratio_max":    round(float(sub["ratio"].max()), 4),
                })

        result = pd.DataFrame(records)
        logger.info(
            "Analyse des seuils (score=%s) :\n%s",
            score_col,
            result[["threshold", "lead_days", "n_triggers", "trigger_rate", "ratio_max"]].to_string(index=False),
        )
        return result

    def permutation_test(
        self,
        df_index: pd.DataFrame,
        score_col: str = "patchcore_score",
        lead: int = 7,
        n_permutations: int = N_PERMUTATIONS,
    ) -> dict[str, float]:
        """
        Valide que les scores pré-événements sont significativement plus élevés
        que des fenêtres aléatoires de même durée.

        Protocole :
          H0 : le score moyen pré-événement ≈ score d'une fenêtre aléatoire
          H1 : le score moyen pré-événement est plus élevé (one-sided)

        Returns:
            dict avec observed_mean, null_distribution_mean, p_value, significant.
        """
        events = self.load_events()
        df = self._prepare_df(df_index, score_col)
        if df.empty:
            return {}

        # Scores pré-événements observés (quantile 0.9 pour cohérence)
        pre_scores = []
        for _, event in events.iterrows():
            event_date = pd.to_datetime(event["date"])
            mask = (
                (df["datetime"] >= event_date - pd.Timedelta(days=lead)) &
                (df["datetime"] < event_date) &
                df["quality_flag"].eq("usable")
            )
            vals = df[mask][score_col].dropna().values
            if len(vals) >= 2:
                pre_scores.append(float(np.quantile(vals, 0.90)))

        if not pre_scores:
            logger.warning("Aucune fenêtre pré-événement valide pour le permutation test.")
            return {}

        observed_mean = float(np.mean(pre_scores))

        # Distribution nulle : fenêtres aléatoires de même longueur
        all_scores = df[df["quality_flag"].eq("usable")][score_col].dropna().values
        if len(all_scores) < 10:
            return {}

        rng = np.random.default_rng(42)
        n_event_images = len(pre_scores) * max(1, int(lead * 2))  # approx n images/fenêtre
        n_event_images = min(n_event_images, len(all_scores))

        null_means = np.array([
            float(rng.choice(all_scores, size=n_event_images, replace=False).mean())
            for _ in range(n_permutations)
        ])

        p_value = float((null_means >= observed_mean).mean())
        logger.info(
            "Permutation test (lead=%dd) : observed=%.4f, null_mean=%.4f, p=%.4f %s",
            lead, observed_mean, null_means.mean(), p_value,
            "✓ significatif" if p_value < 0.05 else "✗ non-significatif",
        )

        return {
            "lead_days": lead,
            "observed_mean": observed_mean,
            "null_distribution_mean": float(null_means.mean()),
            "null_distribution_std": float(null_means.std()),
            "p_value": p_value,
            "significant": bool(p_value < 0.05),
            "n_permutations": n_permutations,
        }

    def plot_timeline(
        self,
        df_index: pd.DataFrame,
        output_path: str | Path | None = None,
        score_col: str = "patchcore_score",
        figsize: tuple = (14, 5),
    ) -> "matplotlib.figure.Figure":
        """
        Génère la figure principale : timeline des scores + marqueurs d'événements.

        Args:
            df_index: DataFrame avec score_col, year, month, day, hour, quality_flag.
            output_path: chemin de sauvegarde (None = pas de sauvegarde).
            score_col: colonne de score à tracer.
            figsize: taille de la figure.

        Returns:
            matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        events = self.load_events()
        df = self._prepare_df(df_index, score_col)
        df_usable = df[df["quality_flag"].eq("usable")].copy()

        if df_usable.empty:
            raise ValueError("Aucune image usable avec score disponible.")

        # Agrégation hebdomadaire pour lisibilité
        df_usable = df_usable.set_index("datetime").sort_index()
        weekly = df_usable[score_col].resample("W").agg(["mean", "max", "count"])
        weekly = weekly[weekly["count"] >= 2]

        fig, ax = plt.subplots(figsize=figsize)

        # Zone de fond (écart-type glissant)
        ax.fill_between(
            weekly.index,
            weekly["mean"] - weekly["mean"].rolling(4, min_periods=1).std(),
            weekly["mean"] + weekly["mean"].rolling(4, min_periods=1).std(),
            alpha=0.15, color="#3498db", label="_nolegend_",
        )

        # Score moyen hebdomadaire
        ax.plot(weekly.index, weekly["mean"], color="#3498db", lw=1.5, label=f"{score_col} (moy. hebdo)")

        # Score max hebdomadaire
        ax.plot(weekly.index, weekly["max"], color="#e74c3c", lw=0.8, alpha=0.5, ls="--", label="max hebdo")

        # Ligne de seuil (percentile 90)
        threshold_90 = float(df_usable[score_col].quantile(0.90))
        ax.axhline(threshold_90, color="#e74c3c", ls=":", lw=1.2, label=f"P90 ({threshold_90:.3f})")

        # Marqueurs d'événements
        cmap_intensity = {
            "major": ("#e74c3c", 12, "v"),
            "moderate": ("#e67e22", 9, "^"),
            "minor": ("#f1c40f", 7, "D"),
        }
        for _, event in events.iterrows():
            edate = pd.to_datetime(event["date"])
            intensity = str(event.get("intensity", "minor")).lower()
            color, ms, marker = cmap_intensity.get(intensity, ("#9b59b6", 8, "o"))
            ax.axvline(edate, color=color, alpha=0.6, lw=1.2, ls="-")
            ax.scatter(edate, threshold_90 * 1.05, color=color, s=ms**2, marker=marker,
                       zorder=5, label=f"{event.get('type', '?')} ({event['date']})")

        ax.set_xlabel("Date")
        ax.set_ylabel("Score d'anomalie")
        ax.set_title(f"Timeline scores d'anomalie — Merapi 2014–2018\n({score_col})")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
        plt.tight_layout()

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info("Timeline sauvegardée → %s", output_path)

        return fig

    # ─── helpers privés ───────────────────────────────────────────────────

    @staticmethod
    def _prepare_df(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
        """Ajoute une colonne 'datetime' et filtre sur le score disponible.

        Stratégie de nettoyage des dates :
          - year/month invalides : suppression (non récupérable)
          - day invalide (0 ou NaN) : imputation à J1 (premier du mois)
          - errors='coerce' sur la conversion finale
        """
        import logging
        _log = logging.getLogger("early_warning")
        df = df.copy()
        for col in ["year", "month", "day", "hour", "minute", "second"]:
            df[col] = pd.to_numeric(df.get(col, 0), errors="coerce")

        # Supprimer year/month invalides (non récupérables)
        invalid_base = (
            df["year"].isna() |
            df["month"].isna() | df["month"].le(0) | df["month"].gt(12)
        )
        n_dropped = int(invalid_base.sum())
        if n_dropped > 0:
            _log.warning("_prepare_df : %d lignes supprimées (year/month invalide).", n_dropped)
            df = df[~invalid_base].copy()

        # Imputer les jours invalides à J1 (préserve les données)
        bad_day = df["day"].isna() | df["day"].le(0) | df["day"].gt(31)
        n_imputed = int(bad_day.sum())
        if n_imputed > 0:
            years_aff = sorted(df.loc[bad_day, "year"].dropna().astype(int).unique().tolist())
            df.loc[bad_day, "day"] = 1
            _log.warning(
                "_prepare_df : %d jour(s) invalide(s) imputés à J1. Années : %s",
                n_imputed, years_aff,
            )

        for col in ["hour", "minute", "second"]:
            df[col] = df[col].fillna(0)
        for col in ["year", "month", "day", "hour", "minute", "second"]:
            df[col] = df[col].astype(int)

        df["datetime"] = pd.to_datetime(
            df[["year", "month", "day", "hour", "minute", "second"]],
            errors="coerce",
        )
        n_nat = df["datetime"].isna().sum()
        if n_nat > 0:
            _log.warning("_prepare_df : %d NaT résiduels supprimés.", n_nat)
            df = df[df["datetime"].notna()].copy()

        if score_col in df.columns:
            df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
            return df[df[score_col].notna()].copy()
        return pd.DataFrame()
