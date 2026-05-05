#!/usr/bin/env python3
"""
scrape_missing.py — Téléchargement des images manquantes pour une plage d'années.

Ce script scrape, indexe et télécharge les images pour les années qui n'ont
pas encore de données dans l'index, ou pour les mois sans images téléchargées.

Usage :
    # Télécharger tout 2019-2025 (toutes les images)
    USE_TF=0 USE_TORCH=1 python scrape_missing.py --start 2019 --end 2025

    # Télécharger seulement 2021 complet
    USE_TF=0 USE_TORCH=1 python scrape_missing.py --start 2021 --end 2021

    # Limiter à 50 images par mois (test)
    USE_TF=0 USE_TORCH=1 python scrape_missing.py --start 2019 --end 2025 --max 50

    # Scraper sans télécharger (indexation seule)
    USE_TF=0 USE_TORCH=1 python scrape_missing.py --start 2019 --end 2025 --no-download
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.utils import load_config
from src.scraper import MerapiScraper
from src.indexer import MerapiIndexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scrape_missing")

INDEX_CSV = PROJECT_ROOT / "data" / "index" / "index.csv"


def load_existing_index() -> pd.DataFrame:
    if INDEX_CSV.exists():
        df = pd.read_csv(INDEX_CSV, low_memory=False)
        dl = df["downloaded"].astype(str).str.lower() == "true"
        logger.info(
            "Index existant : %d entrées, %d téléchargées (2014-2018 conservées).",
            len(df), dl.sum(),
        )
        return df
    return pd.DataFrame()


def get_already_scraped_periods(df: pd.DataFrame) -> set[tuple[int, int]]:
    """Retourne les (year, month) déjà présents dans l'index avec des images."""
    if df.empty or "year" not in df.columns:
        return set()
    dl = df["downloaded"].astype(str).str.lower() == "true"
    df_dl = df[dl].dropna(subset=["year", "month"])
    return {
        (int(r["year"]), int(r["month"]))
        for _, r in df_dl.iterrows()
    }


def run_scrape(args: argparse.Namespace) -> None:
    config = load_config()
    scraper = MerapiScraper(config)
    indexer = MerapiIndexer(config)

    df_existing = load_existing_index()
    already_done = get_already_scraped_periods(df_existing)

    # Générer la liste de tous les (year, month) à traiter
    periods_all = [
        (y, m)
        for y in range(args.start, args.end + 1)
        for m in range(1, 13)
    ]

    # Filtrer les mois déjà traités (sauf si --force)
    if args.force:
        periods = periods_all
        logger.info("Mode --force : traitement de tous les %d mois.", len(periods))
    else:
        periods = [(y, m) for y, m in periods_all if (y, m) not in already_done]
        skipped = len(periods_all) - len(periods)
        logger.info(
            "%d mois à traiter (%d déjà indexés → ignorés).",
            len(periods), skipped,
        )

    if not periods:
        logger.info("Rien à faire — tous les mois demandés sont déjà téléchargés.")
        return

    # Statistiques globales
    total_found   = 0
    total_dl_ok   = 0
    total_dl_fail = 0
    months_empty  = 0
    months_error  = 0

    for i, (year, month) in enumerate(periods, 1):
        logger.info(
            "[%d/%d] ── %d/%02d ──────────────────────────────────────────",
            i, len(periods), year, month,
        )
        try:
            records = scraper.scrape_month(year, month)
        except Exception as e:
            logger.error("  Scraping %d/%02d échoué : %s", year, month, e)
            months_error += 1
            continue

        if not records:
            logger.info("  → Aucune image trouvée pour %d/%02d", year, month)
            months_empty += 1
            continue

        total_found += len(records)
        logger.info("  → %d images trouvées", len(records))

        # Indexation
        try:
            indexer.upsert(records)
        except Exception as e:
            logger.error("  Indexation %d/%02d échouée : %s", year, month, e)
            months_error += 1
            continue

        # Téléchargement
        if not args.no_download:
            try:
                scraper.download_images(records, max_images=args.max)
                indexer.upsert(records)  # mise à jour statut downloaded
                ok = sum(1 for r in records if r.get("downloaded"))
                fail = len(records) - ok
                total_dl_ok   += ok
                total_dl_fail += fail
                logger.info(
                    "  → Téléchargement : %d OK, %d échecs",
                    ok, fail,
                )
            except Exception as e:
                logger.error("  Téléchargement %d/%02d échoué : %s", year, month, e)
                months_error += 1

        # Petite pause pour ne pas surcharger le serveur
        time.sleep(0.3)

    # ── Résumé ────────────────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("SCRAPING TERMINÉ")
    logger.info("  Mois traités  : %d", len(periods))
    logger.info("  Images trouvées: %d", total_found)
    if not args.no_download:
        logger.info("  Téléchargées  : %d OK, %d échecs", total_dl_ok, total_dl_fail)
    logger.info("  Mois vides    : %d", months_empty)
    logger.info("  Erreurs       : %d", months_error)
    logger.info("═" * 60)

    # Synchronisation finale index ↔ disque
    logger.info("Synchronisation index ↔ fichiers sur disque...")
    try:
        indexer.sync_file_status()
    except Exception as e:
        logger.warning("sync_file_status a échoué : %s", e)

    indexer.print_summary()

    df_final = indexer.load()
    dl_final = (df_final["downloaded"].astype(str).str.lower() == "true").sum()
    logger.info("Index final : %d entrées, %d téléchargées.", len(df_final), dl_final)

    if not args.no_download:
        logger.info("")
        logger.info("Prochaine étape : recalculer les features + réentraîner le classifieur")
        logger.info("  USE_TF=0 USE_TORCH=1 python train_volcano_pipeline.py --force-features")
        logger.info("Puis relancer PatchCore sur les nouvelles images :")
        logger.info("  USE_TF=0 USE_TORCH=1 python run_v1_pipeline.py --step patchcore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scraping + download des images Merapi manquantes"
    )
    parser.add_argument("--start", type=int, default=2019, help="Première année (défaut: 2019)")
    parser.add_argument("--end",   type=int, default=2025, help="Dernière année (défaut: 2025)")
    parser.add_argument(
        "--max", type=int, default=None, metavar="N",
        help="Nombre max d'images à télécharger par mois (None = toutes)",
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Scraper et indexer sans télécharger",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Retraiter même les mois déjà présents dans l'index",
    )
    run_scrape(parser.parse_args())
