"""
downloader.py — Téléchargement d'images Kalor du Merapi.

Règles métier strictes :
  - Seules les images KALOR (filename ∈ {kalor_*, ech_kalor_*})
  - Maximum 30 images par mois (configurable via config['pipeline']['max_images_per_month'])
  - Plage temporelle 2014–2025
  - Idempotent : skip si déjà téléchargé et valide

Usage :
    from src.ingestion.downloader import KalorDownloader
    from src.utils import load_config

    dl = KalorDownloader(load_config())
    records = dl.download_month(2019, 6)     # ≤ 30 images Kalor
    all_r   = dl.download_range(year=2019)   # tous les mois 2019
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.utils import get_raw_image_dir, load_config
from src.scraper import MerapiScraper

try:
    from loguru import logger
except ImportError:
    import logging as _l

    logger = _l.getLogger("ingestion.downloader")  # type: ignore[assignment]

# ── Règles métier ─────────────────────────────────────────────────────────
MAX_IMAGES_PER_MONTH: int = 30
YEAR_START: int = 2014
YEAR_END:   int = 2025


class KalorDownloader:
    """
    Téléchargeur d'images Kalor avec règles métier strictes.

    Propriétés :
      - Filtre caméra : uniquement kalor_* et ech_kalor_*
      - Plafond       : max_images_per_month (défaut 30)
      - Plage         : 2014 → 2025
      - Skip          : ne re-télécharge pas si fichier valide présent
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config         = config
        self.scraper        = MerapiScraper(config)
        self.max_per_month  = int(
            config.get("pipeline", {}).get("max_images_per_month", MAX_IMAGES_PER_MONTH)
        )
        self._dl_cfg        = config["download"]

    # ----------------------------------------------------------
    # Téléchargement d'un mois
    # ----------------------------------------------------------

    def download_month(
        self,
        year: int,
        month: int,
    ) -> list[dict[str, Any]]:
        """
        Scrape et télécharge les images Kalor d'un mois.

        Ne re-télécharge pas les images déjà présentes et valides.

        Args:
            year:  année (YEAR_START ≤ year ≤ YEAR_END).
            month: mois (1–12).

        Returns:
            Liste de dicts : url, filename, local_path, year, month,
                             downloaded (bool), file_size_bytes (int|None),
                             error (str|None).
        """
        if not (YEAR_START <= year <= YEAR_END):
            logger.warning(f"Année hors plage [{YEAR_START}–{YEAR_END}] : {year}")
            return []

        # 1. Scraping
        records = self.scraper.scrape_month(year, month)
        if not records:
            logger.info(f"Aucune image scrapée pour {year}/{month:02d}")
            return []

        # 2. Filtre Kalor (double vérification — scraper filtre déjà)
        records = [r for r in records if self._is_kalor(r.get("filename", ""))]
        if not records:
            logger.warning(
                f"{year}/{month:02d} : aucune image Kalor après filtrage"
            )
            return []

        # 3. Plafond mensuel
        if len(records) > self.max_per_month:
            logger.info(
                f"{year}/{month:02d} : {len(records)} images → plafond {self.max_per_month}"
            )
            records = records[: self.max_per_month]

        # 4. Téléchargement
        raw_dir  = get_raw_image_dir(self.config, year, month)
        delay    = float(self._dl_cfg.get("image_delay_s", 0.5))
        min_size = int(self._dl_cfg.get("min_file_size_bytes", 1000))
        results: list[dict[str, Any]] = []

        for rec in records:
            url      = rec.get("url", "")
            filename = rec.get("filename", "")
            local_path = raw_dir / filename

            result: dict[str, Any] = {
                **rec,
                "local_path":      str(local_path),
                "downloaded":      False,
                "file_size_bytes": None,
                "error":           None,
            }

            # Skip si déjà présent et valide
            if local_path.exists() and local_path.stat().st_size >= min_size:
                result["downloaded"]      = True
                result["file_size_bytes"] = local_path.stat().st_size
                results.append(result)
                continue

            # Téléchargement
            try:
                resp = self.scraper._get_with_retry(url)
                if resp is None:
                    result["error"] = "Téléchargement échoué (toutes tentatives)"
                elif len(resp.content) < min_size:
                    result["error"] = (
                        f"Fichier trop petit : {len(resp.content)} octets"
                    )
                else:
                    local_path.write_bytes(resp.content)
                    result["downloaded"]      = True
                    result["file_size_bytes"] = local_path.stat().st_size
                    logger.debug(f"✓ {filename}")
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
                logger.error(f"Erreur téléchargement {filename}: {exc}")

            results.append(result)
            time.sleep(delay)

        n_ok = sum(bool(r["downloaded"]) for r in results)
        logger.info(f"Download {year}/{month:02d} : {n_ok}/{len(results)} OK")
        return results

    # ----------------------------------------------------------
    # Téléchargement d'une plage
    # ----------------------------------------------------------

    def download_range(
        self,
        year: int | None = None,
        month: int | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Télécharge la plage complète 2014–2025 (ou un sous-ensemble).

        Args:
            year:  si fourni, restreint à cette année uniquement.
            month: si fourni avec year, restreint à ce mois.

        Returns:
            Dict {"YYYY-MM": [résultats]}.
        """
        years  = range(year, year + 1)  if year  else range(YEAR_START, YEAR_END + 1)
        months = range(month, month + 1) if month else range(1, 13)

        all_results: dict[str, list[dict[str, Any]]] = {}
        for y in years:
            for m in months:
                key     = f"{y}-{m:02d}"
                results = self.download_month(y, m)
                if results:
                    all_results[key] = results

        total_ok = sum(
            sum(bool(r["downloaded"]) for r in v)
            for v in all_results.values()
        )
        total    = sum(len(v) for v in all_results.values())
        logger.info(f"Download terminé : {total_ok}/{total} images OK")
        return all_results

    # ----------------------------------------------------------
    # Utilitaire
    # ----------------------------------------------------------

    @staticmethod
    def _is_kalor(filename: str) -> bool:
        """Retourne True si le fichier appartient à la caméra Kalor."""
        low = Path(filename).name.lower()
        return low.startswith("kalor") or low.startswith("ech_kalor")
