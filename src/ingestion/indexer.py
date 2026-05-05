"""
indexer.py — IndexManager : source unique de vérité (index.csv).

Extension de MerapiIndexer avec :
  - sync_all()           : réconciliation raw ↔ processed ↔ index.csv
  - update_quality_bulk(): mise à jour en bloc des quality_flags
  - fix_unknown_flags()  : corrige les 'unknown' pour lesquels on a des stats
  - La colonne quality_flag ne contient JAMAIS 'unknown' de façon persistante
    pour des images dont on dispose des statistiques de pixels.

Usage :
    from src.ingestion.indexer import IndexManager
    from src.utils import load_config

    im = IndexManager(load_config())
    im.sync_all()                        # réconciliation complète
    im.fix_unknown_flags()               # corrige les unknown résiduels
    im.update_quality_bulk(results)      # met à jour après preprocessing
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.indexer import MerapiIndexer, INDEX_COLUMNS
from src.utils import PROJECT_ROOT

try:
    from loguru import logger
except ImportError:
    import logging as _l

    logger = _l.getLogger("ingestion.indexer")  # type: ignore[assignment]


class IndexManager(MerapiIndexer):
    """
    Gestionnaire avancé de l'index avec synchronisation et gestion qualité.

    Hérite de MerapiIndexer (load, upsert, update_fields, build_from_disk, …).
    Ajoute :
      - sync_all()           : réconcile raw / processed / index
      - update_quality_bulk(): met à jour quality_flag + stats en bloc
      - fix_unknown_flags()  : corrige les 'unknown' résiduels
    """

    # ----------------------------------------------------------
    # Synchronisation complète
    # ----------------------------------------------------------

    def sync_all(self) -> pd.DataFrame:
        """
        Réconciliation complète raw ↔ processed ↔ index.csv.

        Étapes :
            1. build_from_disk() — scan data/raw/ et upsert dans l'index
            2. Mise à jour de la colonne 'processed' pour chaque entrée
            3. Resynchronise 'downloaded' avec la présence réelle des fichiers

        Returns:
            DataFrame mis à jour sauvegardé dans index.csv.
        """
        logger.info("sync_all : réconciliation index ↔ raw ↔ processed…")

        # 1. Reconstruire depuis raw/ (idempotent)
        df = self.build_from_disk()
        if df.empty:
            logger.warning("sync_all : aucune image dans data/raw/")
            return df

        # 2. Ajouter colonne 'processed' si absente
        if "processed" not in df.columns:
            df["processed"] = False

        proc_base = PROJECT_ROOT / self.config["paths"]["data_processed"]
        n_proc = 0
        for idx, row in df.iterrows():
            try:
                y = int(row["year"])
                m = int(row["month"])
            except (TypeError, ValueError):
                continue
            fn       = str(row.get("filename", ""))
            png_name = Path(fn).stem + ".png"
            proc_path = proc_base / str(y) / f"{m:02d}" / png_name
            is_proc   = proc_path.exists()
            if df.at[idx, "processed"] != is_proc:
                df.at[idx, "processed"] = is_proc
            if is_proc:
                n_proc += 1

        self._save(df)

        n_dl = int((df["downloaded"] == True).sum())
        logger.info(
            f"sync_all terminé : {len(df)} total | "
            f"{n_dl} téléchargées | {n_proc} traitées"
        )
        return df

    # ----------------------------------------------------------
    # Mise à jour qualité en bloc
    # ----------------------------------------------------------

    def update_quality_bulk(
        self,
        results: list[dict[str, Any]],
    ) -> int:
        """
        Met à jour quality_flag, is_night et les métriques pixel dans l'index
        pour tous les résultats fournis.

        Matching par filename (robuste aux changements de chemin/URL).

        Args:
            results: liste de dicts avec au minimum 'filename' et 'quality_flag'.
                     Typiquement : sortie de ImageProcessor.process_month().

        Returns:
            Nombre d'entrées effectivement mises à jour.
        """
        if not results:
            return 0

        df = self.load()
        if df.empty:
            logger.warning("update_quality_bulk : index vide, rien à mettre à jour")
            return 0

        update_cols = [
            "quality_flag", "is_night",
            "mean_brightness", "std_brightness", "variance",
        ]
        # Ajouter les colonnes manquantes à l'index
        for col in update_cols:
            if col not in df.columns:
                df[col] = pd.NA

        updated = 0
        for r in results:
            filename = (
                r.get("filename")
                or Path(r.get("raw_path", "")).name
                or Path(r.get("processed_path", "")).stem + ".jpg"
            )
            if not filename:
                continue

            mask = df["filename"] == filename
            if not mask.any():
                continue

            for col in update_cols:
                val = r.get(col)
                if val is not None:
                    df.loc[mask, col] = val

            updated += int(mask.sum())

        self._save(df)
        logger.info(f"update_quality_bulk : {updated} entrées mises à jour.")
        return updated

    # ----------------------------------------------------------
    # Correction des flags 'unknown' résiduels
    # ----------------------------------------------------------

    def fix_unknown_flags(
        self,
        night_thresh: float = 30.0,
        cloud_var_thresh: float = 50.0,
    ) -> int:
        """
        Recalcule quality_flag pour les entrées 'unknown' qui ont déjà
        mean_brightness + variance dans l'index.

        Args:
            night_thresh:     seuil de luminosité nuit (0–255).
            cloud_var_thresh: seuil de variance nuages.

        Returns:
            Nombre d'entrées corrigées.
        """
        df = self.load()
        if df.empty:
            return 0

        if "quality_flag" not in df.columns or "mean_brightness" not in df.columns:
            return 0

        mask_unknown = df["quality_flag"].fillna("unknown").isin(["unknown", ""])
        mb_series    = pd.to_numeric(df["mean_brightness"], errors="coerce")
        has_stats    = mb_series.notna()
        to_fix       = mask_unknown & has_stats

        if not to_fix.any():
            logger.info("fix_unknown_flags : aucun flag 'unknown' à corriger.")
            return 0

        vb_series = pd.to_numeric(df.get("variance", pd.Series(dtype=float)), errors="coerce")

        fixed = 0
        for idx in df[to_fix].index:
            try:
                mb = float(mb_series.at[idx])
                vb = float(vb_series.at[idx]) if pd.notna(vb_series.at[idx]) else 0.0
            except (TypeError, ValueError):
                continue

            if mb < night_thresh:
                flag     = "dark"
                is_night = True
            elif vb < cloud_var_thresh:
                flag     = "cloudy"
                is_night = False
            else:
                flag     = "usable"
                is_night = False

            df.at[idx, "quality_flag"] = flag
            df.at[idx, "is_night"]     = is_night
            fixed += 1

        if fixed:
            self._save(df)
            logger.info(f"fix_unknown_flags : {fixed} flags corrigés.")
        return fixed

    # ----------------------------------------------------------
    # Rebuild : recalcul qualité depuis les PNGs traités
    # ----------------------------------------------------------

    def rebuild_quality_from_processed(self) -> int:
        """
        Recalcule quality_flag + is_night pour TOUTES les images dont le PNG
        traité existe dans data/processed/.

        Utile après un rebuild-processed ou pour corriger un index incohérent.

        Returns:
            Nombre d'entrées mises à jour.
        """
        from src.ingestion.quality import QualityClassifier

        cfg  = self.config
        qc   = QualityClassifier.from_config(cfg)
        proc_base = PROJECT_ROOT / cfg["paths"]["data_processed"]
        df   = self.load()

        if df.empty:
            logger.warning("rebuild_quality_from_processed : index vide")
            return 0

        update_cols = [
            "quality_flag", "is_night",
            "mean_brightness", "std_brightness", "variance",
        ]
        for col in update_cols:
            if col not in df.columns:
                df[col] = pd.NA

        updated = 0
        for idx, row in df.iterrows():
            try:
                y = int(row["year"])
                m = int(row["month"])
            except (TypeError, ValueError):
                continue

            fn       = str(row.get("filename", ""))
            png_path = proc_base / str(y) / f"{m:02d}" / (Path(fn).stem + ".png")

            if not png_path.exists():
                continue

            try:
                qr = qc.classify(png_path)
            except Exception as exc:
                logger.error(f"rebuild_quality : erreur sur {png_path.name}: {exc}")
                continue

            for col in update_cols:
                val = qr.get(col)
                if val is not None:
                    df.at[idx, col] = val
            updated += 1

        if updated:
            self._save(df)
            logger.info(f"rebuild_quality_from_processed : {updated} entrées mises à jour.")
        return updated
