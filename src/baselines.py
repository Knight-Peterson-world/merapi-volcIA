"""
baselines.py — Méthodes baseline de détection d'anomalies (Phase 4).

Méthodes implémentées :
  1. Différence inter-images (pixel-level absolute difference)
  2. SSIM — Structural Similarity Index entre paires d'images
  3. Comparaison heure similaire (même créneau horaire, jours consécutifs)
  4. Score de luminosité nocturne (détection d'incandescence)

Chaque méthode produit un score d'anomalie par image
stocké dans l'index CSV via la colonne `anomaly_score`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging

    class _FallbackLogger:
        def __init__(self):
            self._l = logging.getLogger("baselines")
            self._l.setLevel(logging.INFO)
            if not self._l.handlers:
                h = logging.StreamHandler(sys.stderr)
                h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
                self._l.addHandler(h)
        def info(self, m, *a, **k): self._l.info(m)
        def warning(self, m, *a, **k): self._l.warning(m)
        def debug(self, m, *a, **k): self._l.debug(m)
        def error(self, m, *a, **k): self._l.error(m)

    logger = _FallbackLogger()

from src.utils import load_config, PROJECT_ROOT, parse_filename_datetime
from src.preprocessing import MerapiPreprocessor


# ============================================================
# Classe principale
# ============================================================

class BaselineDetector:
    """
    Détecteur d'anomalies par méthodes simples (baselines statistiques).

    Toutes les méthodes attendent des images prétraitées (PNG, float32,
    normalisées [0, 1], niveaux de gris, taille target_size).

    Attributes:
        config: configuration du projet.
        night_threshold: seuil de luminosité moyenne sous lequel
                         l'image est considérée « nuit ».
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            config = load_config()
        self.config = config
        pp_cfg = config.get("preprocessing", {})
        self.night_threshold = pp_cfg.get("night_brightness_threshold", 30) / 255.0
        self.scores_dir = PROJECT_ROOT / config["paths"]["scores"]
        self.scores_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # 1. Différence inter-images (Mean Absolute Difference)
    # ----------------------------------------------------------

    @staticmethod
    def absolute_difference(img_a: np.ndarray, img_b: np.ndarray) -> float:
        """
        Différence absolue moyenne entre deux images.

        Un score élevé signifie un changement important entre les deux
        clichés (éruption, panache, éclairage brutal, artefact).

        Args:
            img_a, img_b: arrays float32 de même shape (H, W).

        Returns:
            Score MAD ∈ [0, 1].
        """
        return float(np.mean(np.abs(img_a.astype(np.float64) - img_b.astype(np.float64))))

    # ----------------------------------------------------------
    # 2. SSIM — Structural Similarity Index
    # ----------------------------------------------------------

    @staticmethod
    def ssim_score(img_a: np.ndarray, img_b: np.ndarray) -> float:
        """
        Calcule 1 − SSIM entre deux images.

        SSIM mesure la similarité structurelle ; 1 − SSIM donne donc
        un score d'anomalie : 0 = identiques, 1 = totalement différentes.

        Args:
            img_a, img_b: arrays float32 de même shape (H, W).

        Returns:
            Score 1 − SSIM ∈ [0, 1], ou NaN si skimage absent.
        """
        if ssim is None:
            logger.warning("scikit-image non installé — SSIM indisponible.")
            return float("nan")
        data_range = max(img_a.max() - img_a.min(), img_b.max() - img_b.min())
        if data_range < 1e-8:
            data_range = 1.0
        similarity = ssim(img_a, img_b, data_range=data_range)
        return float(1.0 - similarity)

    # ----------------------------------------------------------
    # 3. Score de luminosité nocturne (incandescence)
    # ----------------------------------------------------------

    def night_brightness_score(self, img: np.ndarray) -> float:
        """
        Détecte une luminosité anormale sur une image nocturne.

        Pour une image de nuit (luminosité globale < seuil), on calcule
        la fraction de pixels « brillants » (> 2× seuil). Une valeur
        élevée peut indiquer une incandescence volcanique.

        Args:
            img: array float32 (H, W), normalisé [0, 1].

        Returns:
            Fraction de pixels lumineux ∈ [0, 1], ou 0.0 si jour.
        """
        mean_brightness = float(np.mean(img))
        if mean_brightness >= self.night_threshold:
            # Image de jour → pas de score nocturne
            return 0.0
        bright_threshold = self.night_threshold * 2.0
        bright_pixels = np.sum(img > bright_threshold)
        total_pixels = img.size
        return float(bright_pixels / total_pixels) if total_pixels > 0 else 0.0

    # ----------------------------------------------------------
    # 4. Comparaison heure similaire
    # ----------------------------------------------------------

    def compare_same_hour(
        self,
        df: pd.DataFrame,
        target_hour: int,
        year: int,
        month: int,
    ) -> pd.DataFrame:
        """
        Compare des images prises à la même heure sur des jours consécutifs.

        Pour chaque paire (jour_i, jour_i+1) au créneau `target_hour`,
        calcule MAD et SSIM.

        Args:
            df: index filtré (doit contenir 'day', 'hour', 'local_path').
            target_hour: heure cible (0–23).
            year / month: pour le filtrage.

        Returns:
            DataFrame avec colonnes :
              day_a, day_b, hour, mad_score, ssim_score, path_a, path_b
        """
        subset = df[
            (df["year"] == year) & (df["month"] == month) & (df["hour"] == target_hour)
        ].sort_values("day").reset_index(drop=True)
        rows = []
        for i in range(len(subset) - 1):
            row_a = subset.iloc[i]
            row_b = subset.iloc[i + 1]

            # Charger les images prétraitées
            path_a = self._resolve_processed_path(row_a["local_path"])
            path_b = self._resolve_processed_path(row_b["local_path"])
            img_a = MerapiPreprocessor.load_processed_image(path_a)
            img_b = MerapiPreprocessor.load_processed_image(path_b)

            if img_a is None or img_b is None:
                continue

            rows.append({
                "day_a": int(row_a["day"]),
                "day_b": int(row_b["day"]),
                "hour": target_hour,
                "mad_score": self.absolute_difference(img_a, img_b),
                "ssim_score": self.ssim_score(img_a, img_b),
                "path_a": str(path_a),
                "path_b": str(path_b),
            })

        return pd.DataFrame(rows)

    # ----------------------------------------------------------
    # Pipeline complet sur un mois
    # ----------------------------------------------------------

    def score_month(
        self,
        df: pd.DataFrame,
        year: int,
        month: int,
    ) -> pd.DataFrame:
        """
        Calcule les scores baseline pour toutes les images d'un mois.

        Pour chaque image téléchargée et prétraitée :
          - score de luminosité nocturne
          - MAD et SSIM par rapport à l'image précédente (tri chronologique)

        Args:
            df: index complet chargé depuis MerapiIndexer.
            year, month: mois cible.

        Returns:
            DataFrame avec colonnes :
              filename, day, hour, night_score, mad_prev, ssim_prev, combined_score
        """
        subset = df[
            (df["year"] == year)
            & (df["month"] == month)
            & (df["downloaded"] == True)
        ].copy()

        # Fallback : reconstruire day/hour/minute depuis le nom de fichier si absent
        if subset["day"].isna().all() or subset["hour"].isna().all():
            logger.warning("day/hour absents de l'index — extraction depuis les noms de fichiers.")
            for col in ["day", "hour", "minute"]:
                if subset[col].isna().all() and "filename" in subset.columns:
                    parsed = subset["filename"].apply(
                        lambda fn: (parse_filename_datetime(str(fn)) or {}).get(col)
                    )
                    subset[col] = pd.to_numeric(parsed, errors="coerce")

        subset = subset.sort_values(["day", "hour", "minute"]).reset_index(drop=True)

        logger.info(f"Scoring baseline {year}/{month:02d} — {len(subset)} images")

        results = []
        prev_img = None

        for idx, row in tqdm(subset.iterrows(), total=len(subset), desc="Baselines"):
            proc_path = self._resolve_processed_path(row["local_path"])
            img = MerapiPreprocessor.load_processed_image(proc_path)

            if img is None:
                logger.debug(f"Image prétraitée introuvable : {proc_path}")
                continue

            night_score = self.night_brightness_score(img)

            mad_prev = float("nan")
            ssim_prev = float("nan")
            if prev_img is not None and prev_img.shape == img.shape:
                mad_prev = self.absolute_difference(prev_img, img)
                ssim_prev = self.ssim_score(prev_img, img)

            # Score combiné simple : moyenne des scores disponibles
            scores = [s for s in [night_score, mad_prev, ssim_prev] if not np.isnan(s)]
            combined = float(np.mean(scores)) if scores else 0.0

            results.append({
                "filename": row["filename"],
                "day": int(row["day"]) if pd.notna(row.get("day")) else 0,
                "hour": int(row["hour"]) if pd.notna(row.get("hour")) else 0,
                "night_score": night_score,
                "mad_prev": mad_prev,
                "ssim_prev": ssim_prev,
                "combined_score": combined,
            })

            prev_img = img

        scores_df = pd.DataFrame(results)

        # --- Seuil statistique μ + 2σ ---
        if not scores_df.empty and "combined_score" in scores_df.columns:
            mean_score = scores_df["combined_score"].mean()
            std_score = scores_df["combined_score"].std()
            threshold = mean_score + 2 * std_score
            scores_df["threshold"] = threshold
            scores_df["is_anomaly"] = scores_df["combined_score"] > threshold
            scores_df["distance_to_threshold"] = scores_df["combined_score"] - threshold

            n_anomalies = int(scores_df["is_anomaly"].sum())
            logger.info(
                f"Seuil μ+2σ = {threshold:.6f} — {n_anomalies} anomalies détectées "
                f"sur {len(scores_df)} images"
            )

        if not scores_df.empty:
            out_path = self.scores_dir / f"baselines_{year}_{month:02d}.csv"
            scores_df.to_csv(out_path, index=False)
            logger.info(f"Scores sauvegardés → {out_path}")

        return scores_df

    # ----------------------------------------------------------
    # Mise à jour de l'index
    # ----------------------------------------------------------

    @staticmethod
    def update_index_scores(
        scores_df: pd.DataFrame,
        indexer: Any,
        score_col: str = "combined_score",
    ) -> None:
        """
        Met à jour la colonne `anomaly_score` de l'index CSV.

        Args:
            scores_df: résultats de score_month().
            indexer: instance de MerapiIndexer.
            score_col: nom de la colonne à copier dans anomaly_score.
        """
        if scores_df.empty:
            logger.warning("Aucun score à écrire dans l'index.")
            return

        df_index = indexer.load()
        for _, row in scores_df.iterrows():
            mask = df_index["filename"] == row["filename"]
            if mask.any():
                df_index.loc[mask, "anomaly_score"] = row[score_col]

        indexer._save(df_index)
        logger.info(f"{len(scores_df)} scores écrits dans l'index.")

    # ----------------------------------------------------------
    # Utilitaires internes
    # ----------------------------------------------------------

    def _resolve_processed_path(self, raw_local_path: str | Path) -> Path:
        """
        Dérive le chemin de l'image prétraitée à partir du chemin raw.

        data/raw/2014/11/img.jpg → data/processed/2014/11/img.png
        (rétro-compatible : cherchera aussi .jpg si .png absent)
        """
        raw_rel = str(raw_local_path)
        proc_rel = raw_rel.replace(
            self.config["paths"]["data_raw"],
            self.config["paths"]["data_processed"],
        )
        # Priorité au format PNG (lossless)
        proc_path = PROJECT_ROOT / proc_rel
        png_path = proc_path.with_suffix(".png")
        if png_path.exists():
            return png_path
        return proc_path  # fallback pour rétro-compatibilité (.jpg / .npy)


# ============================================================
# Point d'entrée CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baselines anomalie Merapi (Phase 4)")
    parser.add_argument("--year", type=int, default=2014)
    parser.add_argument("--month", type=int, default=11)
    args = parser.parse_args()

    cfg = load_config()

    from src.utils import setup_logger
    setup_logger(cfg)

    from src.indexer import MerapiIndexer

    detector = BaselineDetector(cfg)
    indexer = MerapiIndexer(cfg)
    df = indexer.load()

    scores = detector.score_month(df, args.year, args.month)
    if not scores.empty:
        detector.update_index_scores(scores, indexer)
        print(f"\n--- Top 10 anomalies (score combiné) ---")
        print(scores.nlargest(10, "combined_score")[
            ["filename", "day", "hour", "night_score", "mad_prev", "ssim_prev",
             "combined_score", "is_anomaly", "distance_to_threshold"]
        ].to_string(index=False))
