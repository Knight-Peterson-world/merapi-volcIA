"""
run_full_pipeline.py — Script orchestrateur pour le pipeline complet sur 10 ans.

Exécute séquentiellement :
  1. Scraping + indexation de tous les mois (2014 → 2024)
  2. Téléchargement progressif des images
  3. Prétraitement (resize, normalisation, grayscale)
  4. Classification qualité
  5. Scores baseline (MAD, SSIM, luminosité nocturne)

Usage :
    # Pipeline complet (scraping + download + preprocessing + quality + baselines)
    python run_full_pipeline.py

    # Scraping + indexation seulement (pas de téléchargement)
    python run_full_pipeline.py --no-download

    # Uniquement preprocessing + quality + baselines (images déjà téléchargées)
    python run_full_pipeline.py --skip-scraping

    # Limiter le nombre d'images téléchargées par mois
    python run_full_pipeline.py --max-per-month 100

    # Année unique
    python run_full_pipeline.py --year 2014
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Ajouter la racine du projet au PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, setup_logger
from src.scraper import MerapiScraper
from src.indexer import MerapiIndexer
from src.preprocessing import MerapiPreprocessor
from src.quality_filter import QualityFilter
from src.baselines import BaselineDetector

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stderr))


def generate_periods(year_start: int, year_end: int, single_year: int | None = None) -> list[tuple[int, int]]:
    """Génère la liste (year, month) pour la plage donnée."""
    if single_year is not None:
        return [(single_year, m) for m in range(1, 13)]
    periods = []
    for y in range(year_start, year_end + 1):
        for m in range(1, 13):
            periods.append((y, m))
    return periods


def run_pipeline(args: argparse.Namespace) -> None:
    config = load_config()
    setup_logger(config)

    scraper = MerapiScraper(config)
    indexer = MerapiIndexer(config)
    preprocessor = MerapiPreprocessor(config)
    quality = QualityFilter(config)
    detector = BaselineDetector(config)

    year_start = config["source"]["years_available"]["start"]
    year_end = config["source"]["years_available"]["end"]
    periods = generate_periods(year_start, year_end, args.year)

    logger.info(f"Pipeline complet — {len(periods)} mois de {year_start} à {year_end}")
    logger.info(f"Options : download={not args.no_download}, max_per_month={args.max_per_month}")

    # ==========================================================
    # PHASE 1 — Scraping + Indexation + Téléchargement
    # ==========================================================
    if not args.skip_scraping:
        logger.info("=" * 60)
        logger.info("PHASE 1 — Scraping et indexation")
        logger.info("=" * 60)

        phase1_success = 0
        phase1_empty = 0
        phase1_errors = 0

        for i, (year, month) in enumerate(periods, 1):
            logger.info(f"[{i}/{len(periods)}] Scraping {year}/{month:02d}...")
            try:
                records = scraper.scrape_month(year, month)
                if records:
                    indexer.upsert(records)
                    logger.info(f"  → {len(records)} images indexées")
                    phase1_success += 1

                    if not args.no_download:
                        scraper.download_images(
                            records,
                            max_images=args.max_per_month,
                        )
                        # Mettre à jour l'index avec le statut de téléchargement
                        indexer.upsert(records)
                else:
                    logger.info(f"  → Aucune image trouvée pour {year}/{month:02d}")
                    phase1_empty += 1
            except Exception as e:
                logger.error(f"  Erreur sur {year}/{month:02d} : {e}")
                phase1_errors += 1
                continue

        logger.info(
            f"Phase 1 terminée — {phase1_success} mois avec images, "
            f"{phase1_empty} mois vides, {phase1_errors} erreurs"
        )
        indexer.print_summary()

        # Synchroniser le statut downloaded avec les fichiers réellement sur disque
        # (indispensable quand --no-download : les fichiers existent déjà mais
        # le scraper les marque downloaded=False)
        logger.info("Synchronisation index ↔ fichiers sur disque...")
        indexer.sync_file_status()

    # Si --skip-scraping OU si l'index est vide après scraping,
    # construire/enrichir l'index depuis les fichiers sur disque
    df_check = indexer.load()
    dl_count = int((df_check["downloaded"] == True).sum()) if not df_check.empty else 0
    if dl_count == 0:
        logger.info(
            "Aucune image marquée 'downloaded' dans l'index — "
            "construction depuis les fichiers sur disque..."
        )
        indexer.build_from_disk()

    # ==========================================================
    # PHASE 3 — Prétraitement + Qualité
    # ==========================================================
    logger.info("=" * 60)
    logger.info("PHASE 3 — Prétraitement et classification qualité")
    logger.info("=" * 60)

    processed_periods = set()
    df = indexer.load()
    dl_mask = df["downloaded"].astype(str).str.lower() == "true"
    for _, row in df[dl_mask].iterrows():
        y = row.get("year")
        m = row.get("month")
        if pd.notna(y) and pd.notna(m):
            processed_periods.add((int(y), int(m)))

    for i, (year, month) in enumerate(sorted(processed_periods), 1):
        logger.info(f"[{i}/{len(processed_periods)}] Prétraitement {year}/{month:02d}...")
        try:
            preprocessor.process_month(year, month, overwrite=False)
        except Exception as e:
            logger.error(f"  Erreur prétraitement {year}/{month:02d} : {e}")

        logger.info(f"  Classification qualité {year}/{month:02d}...")
        try:
            results = quality.classify_month(year, month)
            if results:
                quality.update_index_from_results(results, indexer)
        except Exception as e:
            logger.error(f"  Erreur qualité {year}/{month:02d} : {e}")

    # ==========================================================
    # PHASE 4 — Baselines
    # ==========================================================
    logger.info("=" * 60)
    logger.info("PHASE 4 — Scores baseline")
    logger.info("=" * 60)

    df = indexer.load()
    for i, (year, month) in enumerate(sorted(processed_periods), 1):
        logger.info(f"[{i}/{len(processed_periods)}] Baselines {year}/{month:02d}...")
        try:
            scores = detector.score_month(df, year, month)
            if not scores.empty:
                detector.update_index_scores(scores, indexer)
                logger.info(f"  → {len(scores)} images scorées")
        except Exception as e:
            logger.error(f"  Erreur baselines {year}/{month:02d} : {e}")

    # ==========================================================
    # Résumé final
    # ==========================================================
    logger.info("=" * 60)
    logger.info("PIPELINE TERMINÉ")
    logger.info("=" * 60)
    indexer.print_summary()

    df_final = indexer.load()
    n_total = len(df_final)
    n_dl = (df_final["downloaded"].astype(str).str.lower() == "true").sum()
    n_scored = df_final["anomaly_score"].notna().sum()
    logger.info(f"Total : {n_total} indexées, {n_dl} téléchargées, {n_scored} scorées")
    logger.info(f"Application Streamlit : streamlit run app/streamlit_app.py")


# ==============================================================
# CLI
# ==============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline complet Merapi — scraping, preprocessing, baselines (10 ans)"
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help="Année unique à traiter (par défaut : toutes de 2014 à 2024)"
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Scraper et indexer sans télécharger les images"
    )
    parser.add_argument(
        "--skip-scraping", action="store_true",
        help="Sauter le scraping (preprocessing + baselines seulement)"
    )
    parser.add_argument(
        "--max-per-month", type=int, default=None,
        help="Nombre max d'images à télécharger par mois (None = toutes)"
    )

    run_pipeline(parser.parse_args())
