"""
quality_filter.py — Filtrage et évaluation de la qualité des images Merapi.

Responsabilités :
- Classifier chaque image : 'usable', 'cloudy', 'dark', 'corrupted', 'unknown'
- Calculer des indicateurs de qualité (variance, entropie, luminosité)
- Mettre à jour l'index via MerapiIndexer
- Produire un rapport de qualité

Logique de classification (ordre de priorité) :
    1. corrupted  → image illisible ou taille nulle
    2. dark       → luminosité moyenne < seuil nuit (image nocturne)
    3. cloudy     → variance < seuil (image uniforme = nuage/brouillard)
    4. usable     → reste

Justification scientifique :
    La variance est un proxy robuste pour détecter les images nuageuses :
    un ciel couvert produit une image quasi-uniforme (variance faible),
    tandis qu'une image du dôme volcanique contient des textures variées.
    Ce critère est simple, rapide, et ne nécessite pas de modèle IA.

Usage :
    from src.quality_filter import QualityFilter
    from src.utils import load_config, setup_logger

    config = load_config()
    setup_logger(config)
    qf = QualityFilter(config)

    results = qf.classify_month(year=2014, month=11)
    qf.update_index(results)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
try:
    from loguru import logger
except ModuleNotFoundError:
    import logging as _logging
    import sys as _sys
    class _FallbackLogger:
        def __init__(self):
            self._l = _logging.getLogger("quality_filter")
            self._l.setLevel(_logging.INFO)
            if not self._l.handlers:
                _h = _logging.StreamHandler(_sys.stderr)
                _h.setFormatter(_logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
                self._l.addHandler(_h)
        def info(self, m, *a, **k): self._l.info(m)
        def warning(self, m, *a, **k): self._l.warning(m)
        def debug(self, m, *a, **k): self._l.debug(m)
        def error(self, m, *a, **k): self._l.error(m)
        def exception(self, m, *a, **k): self._l.exception(m)
    logger = _FallbackLogger()
from PIL import Image, UnidentifiedImageError

from src.utils import get_raw_image_dir, load_config, setup_logger, PROJECT_ROOT


# ============================================================
# Constantes — labels de qualité
# ============================================================

QUALITY_USABLE = "usable"
QUALITY_CLOUDY = "cloudy"
QUALITY_DARK = "dark"
QUALITY_CORRUPTED = "corrupted"
QUALITY_UNKNOWN = "unknown"


# ============================================================
# Filtre de qualité
# ============================================================

class QualityFilter:
    """
    Classifieur de qualité d'images basé sur des indicateurs simples.

    Tous les seuils sont paramétrables via config/settings.yaml.
    Cette approche constitue la première baseline qualité du projet.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.pp_cfg = config["preprocessing"]

        # Seuils depuis la configuration
        self.night_thresh = self.pp_cfg.get("night_brightness_threshold", 30)
        self.cloud_var_thresh = self.pp_cfg.get("cloud_variance_threshold", 50.0)

        # Taille minimale de fichier considérée comme valide
        self.min_file_bytes = config["download"].get("min_file_size_bytes", 1000)

    # ----------------------------------------------------------
    # Classification d'une image
    # ----------------------------------------------------------

    def classify_image(self, image_path: Path) -> dict[str, Any]:
        """
        Classifie une image selon sa qualité.

        Args:
            image_path: chemin vers le fichier image.

        Returns:
            dict avec les champs :
                - quality_flag : str (usable / cloudy / dark / corrupted / unknown)
                - is_night     : bool | None
                - mean_brightness, std_brightness, variance, entropy : float | None
                - error        : str | None
        """
        result = {
            "path": str(image_path),
            "quality_flag": QUALITY_UNKNOWN,
            "is_night": None,
            "mean_brightness": None,
            "std_brightness": None,
            "variance": None,
            "entropy": None,
            "error": None,
        }

        # 1. Vérification existence et taille
        if not image_path.exists():
            result["quality_flag"] = QUALITY_CORRUPTED
            result["error"] = "Fichier inexistant"
            return result

        file_size = image_path.stat().st_size
        if file_size < self.min_file_bytes:
            result["quality_flag"] = QUALITY_CORRUPTED
            result["error"] = f"Fichier trop petit : {file_size} octets"
            return result

        # 2. Chargement
        try:
            img = Image.open(image_path).convert("L")
        except (UnidentifiedImageError, Exception) as e:
            result["quality_flag"] = QUALITY_CORRUPTED
            result["error"] = f"Impossible d'ouvrir l'image : {e}"
            return result

        # 3. Calcul des indicateurs
        arr = np.array(img, dtype=np.float32)
        mean_b = float(arr.mean())
        std_b = float(arr.std())
        var_b = float(arr.var())
        entropy = self._compute_entropy(arr)

        result["mean_brightness"] = mean_b
        result["std_brightness"] = std_b
        result["variance"] = var_b
        result["entropy"] = entropy

        # 4. Classification par ordre de priorité
        if mean_b < self.night_thresh:
            # Image sombre → nocturne (pas "inutilisable" — traitable séparément)
            result["quality_flag"] = QUALITY_DARK
            result["is_night"] = True

        elif var_b < self.cloud_var_thresh:
            # Image trop uniforme → nuageuse ou brumeuse
            result["quality_flag"] = QUALITY_CLOUDY
            result["is_night"] = False

        else:
            result["quality_flag"] = QUALITY_USABLE
            result["is_night"] = False

        logger.debug(
            f"{image_path.name} → {result['quality_flag']} "
            f"| lum={mean_b:.1f} | var={var_b:.1f} | H={entropy:.2f}"
        )
        return result

    # ----------------------------------------------------------
    # Traitement par lot
    # ----------------------------------------------------------

    def classify_month(
        self,
        year: int,
        month: int,
        max_images: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Classifie toutes les images brutes d'un mois.

        Args:
            year: année.
            month: mois (1–12).
            max_images: plafond d'images (None = toutes).

        Returns:
            Liste de dicts résultats (un par image).
        """
        from tqdm import tqdm

        raw_dir = get_raw_image_dir(self.config, year, month)
        extensions = set(self.config["source"]["image_extensions"])

        images = sorted([
            p for p in raw_dir.glob("*")
            if p.suffix in extensions
        ])

        if not images:
            logger.warning(f"Aucune image dans {raw_dir}")
            return []

        if max_images is not None and len(images) > max_images:
            logger.info(f"Plafond appliqué : {max_images}/{len(images)} images")
            images = images[:max_images]

        logger.info(f"Classification qualité de {len(images)} images — {year}/{month:02d}")

        results = []
        for img_path in tqdm(images, desc=f"Qualité {year}/{month:02d}", unit="img"):
            try:
                r = self.classify_image(img_path)
            except Exception as exc:
                logger.error(f"Erreur classify_image {img_path.name}: {exc}")
                r = {
                    "path": str(img_path),
                    "quality_flag": QUALITY_CORRUPTED,
                    "is_night": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            r["filename"] = img_path.name
            results.append(r)

        self._log_quality_summary(results, year, month)

        # Avertissement si aucune image utilisable
        n_usable = sum(1 for r in results if r.get("quality_flag") == QUALITY_USABLE)
        n_corrupted = sum(1 for r in results if r.get("quality_flag") == QUALITY_CORRUPTED)
        if n_usable == 0 and results:
            n_non_corrupted = len(results) - n_corrupted
            logger.warning(
                f"⚠  Aucune image 'usable' pour {year}/{month:02d} "
                f"({n_non_corrupted} images non corrompues disponibles). "
                "Vérifiez les seuils night_brightness_threshold et cloud_variance_threshold "
                "dans config/settings.yaml."
            )

        return results

    def classify_all_months(
        self,
        year_start: int = 2014,
        year_end: int = 2018,
        max_per_month: int | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Classifie tous les mois d'une plage d'années.

        Args:
            year_start, year_end: bornes incluses.
            max_per_month: plafond par mois (None = toutes).

        Returns:
            Dict {"YYYY-MM": [résultats]}.
        """
        all_results = {}
        for year in range(year_start, year_end + 1):
            for month in range(1, 13):
                key = f"{year}-{month:02d}"
                results = self.classify_month(
                    year, month, max_images=max_per_month
                )
                if results:
                    all_results[key] = results
        return all_results

    def classify_from_paths(self, paths: list[Path]) -> list[dict[str, Any]]:
        """
        Classifie une liste arbitraire de fichiers images.

        Args:
            paths: liste de chemins locaux.

        Returns:
            Liste de dicts résultats.
        """
        from tqdm import tqdm

        results = []
        for p in tqdm(paths, desc="Classification qualité", unit="img"):
            r = self.classify_image(p)
            r["filename"] = p.name
            results.append(r)
        return results

    # ----------------------------------------------------------
    # Réconciliation de l'index
    # ----------------------------------------------------------

    @staticmethod
    def reconcile_index(df, config) -> pd.DataFrame:
        """
        Réconcilie l'index CSV avec les fichiers réellement présents
        sur disque. Corrige le flag 'downloaded' et met à jour
        file_size_bytes.

        Args:
            df: DataFrame de l'index.
            config: configuration du projet.

        Returns:
            DataFrame mis à jour.
        """
        import pandas as pd
        raw_base = PROJECT_ROOT / config["paths"]["data_raw"]

        updated = 0
        for idx, row in df.iterrows():
            try:
                y, m = int(row["year"]), int(row["month"])
            except (ValueError, TypeError):
                continue
            raw_path = raw_base / str(y) / f"{m:02d}" / row["filename"]
            exists = raw_path.exists()

            if exists:
                df.at[idx, "downloaded"] = True
                try:
                    df.at[idx, "file_size_bytes"] = raw_path.stat().st_size
                except OSError:
                    pass
                updated += 1
            else:
                df.at[idx, "downloaded"] = False

        logger.info(f"Index réconcilié : {updated} fichiers trouvés sur disque")
        return df

    # ----------------------------------------------------------
    # Mise à jour de l'index
    # ----------------------------------------------------------

    def update_index_from_results(
        self,
        results: list[dict[str, Any]],
        indexer,
    ) -> None:
        """
        Met à jour les champs quality_flag, is_night et indicateurs
        dans l'index.

        Args:
            results: sortie de classify_month() ou classify_from_paths().
            indexer: instance de MerapiIndexer.
        """
        import pandas as pd

        df = indexer.load()

        if df.empty:
            logger.warning("Index vide — impossible de mettre à jour les qualités.")
            return

        # Construire un DataFrame de résultats pour le merge
        df_results = pd.DataFrame(results)
        if df_results.empty:
            logger.warning("Aucun résultat de classification à intégrer.")
            return

        # Merge par filename (plus fiable que par URL)
        merge_cols = ["quality_flag", "is_night"]
        # Ajouter les colonnes d'indicateurs si présentes
        for col in ["mean_brightness", "std_brightness", "variance", "entropy"]:
            if col in df_results.columns and col not in df.columns:
                df[col] = np.nan
            if col in df_results.columns:
                merge_cols.append(col)

        updates = 0
        for _, r in df_results.iterrows():
            filename = r.get("filename", "")
            if not filename:
                continue
            mask = df["filename"] == filename
            if not mask.any():
                continue
            for col in merge_cols:
                if col in r and pd.notna(r[col]):
                    df.loc[mask, col] = r[col]
            updates += mask.sum()

        indexer._save(df)
        logger.info(f"Index mis à jour : {updates} entrées quality_flag enrichies.")

    # ----------------------------------------------------------
    # Rapport de qualité
    # ----------------------------------------------------------

    def quality_report(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Génère un rapport de qualité statistique.

        Args:
            results: sortie de classify_month().

        Returns:
            dict avec les statistiques de qualité.
        """
        import pandas as pd

        if not results:
            return {}

        df = pd.DataFrame(results)
        n_total = len(df)

        report = {
            "total": n_total,
            "by_quality": df["quality_flag"].value_counts().to_dict(),
            "usable_rate": float((df["quality_flag"] == QUALITY_USABLE).sum() / n_total),
            "night_rate": float((df["quality_flag"] == QUALITY_DARK).sum() / n_total),
            "cloudy_rate": float((df["quality_flag"] == QUALITY_CLOUDY).sum() / n_total),
            "corrupted_rate": float((df["quality_flag"] == QUALITY_CORRUPTED).sum() / n_total),
        }

        for metric in ["mean_brightness", "variance", "entropy"]:
            if metric not in df.columns:
                continue
            vals = df[metric].dropna()
            if not vals.empty:
                report[f"{metric}_mean"] = float(vals.mean())
                report[f"{metric}_std"] = float(vals.std())
                report[f"{metric}_min"] = float(vals.min())
                report[f"{metric}_max"] = float(vals.max())

        return report

    # ----------------------------------------------------------
    # Indicateurs de qualité
    # ----------------------------------------------------------

    @staticmethod
    def _compute_entropy(arr: np.ndarray) -> float:
        """
        Entropie de Shannon de l'histogramme de l'image (en bits).

        Une entropie élevée → image riche en information (textures).
        Une entropie faible → image uniforme (nuage, brouillard, nuit).

        Complémentaire à la variance pour discriminer les images nuageuses.
        """
        hist, _ = np.histogram(arr.flatten(), bins=256, range=(0, 255))
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total == 0:
            return 0.0
        probs = hist[hist > 0] / total
        return float(-np.sum(probs * np.log2(probs)))

    def _log_quality_summary(
        self,
        results: list[dict[str, Any]],
        year: int,
        month: int,
    ) -> None:
        """Affiche un résumé de la classification dans les logs."""
        report = self.quality_report(results)
        n = report.get("total", 0)
        logger.info(f"Qualité {year}/{month:02d} ({n} images) :")
        for flag, count in report.get("by_quality", {}).items():
            pct = count / n * 100 if n > 0 else 0
            logger.info(f"  {flag:12s} : {count:4d}  ({pct:.1f}%)")
        logger.info(f"  Taux utilisables : {report.get('usable_rate', 0)*100:.1f}%")


# ============================================================
# Point d'entrée CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filtrage qualité images Merapi")
    parser.add_argument("--year", type=int, default=2014)
    parser.add_argument("--month", type=int, default=11)
    args = parser.parse_args()

    cfg = load_config()
    setup_logger(cfg)

    from src.indexer import MerapiIndexer
    qf = QualityFilter(cfg)
    indexer = MerapiIndexer(cfg)

    results = qf.classify_month(args.year, args.month)
    if results:
        qf.update_index_from_results(results, indexer)
        print("\nRapport de qualité :")
        import json
        print(json.dumps(qf.quality_report(results), indent=2))
