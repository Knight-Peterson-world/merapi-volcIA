"""
test_phase1.py — Script de test rapide pour la Phase 1.

Ce script illustre le workflow complet de la Phase 1 :
  1. Scraping d'un mois (novembre 2014)
  2. Indexation des métadonnées
  3. Téléchargement d'un sous-ensemble limité
  4. Vérification de la structure produite

Usage :
    # Depuis la racine du projet :
    python test_phase1.py

    # Pour ne tester que le scraping sans télécharger :
    python test_phase1.py --no-download

    # Pour choisir le nombre d'images à télécharger :
    python test_phase1.py --max-images 20
"""

import argparse
import sys
from pathlib import Path

# Ajouter la racine au PYTHONPATH si exécuté directement
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import load_config, setup_logger, get_monthly_page_url
from src.scraper import MerapiScraper
from src.indexer import MerapiIndexer


def main(year: int, month: int, max_images: int, no_download: bool) -> None:
    # --------------------------------------------------------
    # 1. Initialisation
    # --------------------------------------------------------
    config = load_config()
    setup_logger(config)

    from loguru import logger

    logger.info("=" * 60)
    logger.info("PHASE 1 — Test local scraping + indexation")
    logger.info(f"Cible : {year}/{month:02d}  |  Max téléchargements : {max_images}")
    logger.info("=" * 60)

    scraper = MerapiScraper(config)
    indexer = MerapiIndexer(config)

    # --------------------------------------------------------
    # 2. Scraping de la page mensuelle
    # --------------------------------------------------------
    page_url = get_monthly_page_url(config, year, month)
    logger.info(f"URL cible : {page_url}")

    records = scraper.scrape_month(year=year, month=month)

    if not records:
        logger.error(
            "Aucune image trouvée. Vérifiez :\n"
            "  - votre connexion internet\n"
            "  - que l'URL est accessible depuis votre réseau\n"
            f"  - URL : {page_url}"
        )
        sys.exit(1)

    logger.info(f"✓ {len(records)} image(s) indexée(s)")

    # Afficher quelques exemples
    logger.info("Exemples de records :")
    for r in records[:3]:
        logger.info(
            f"  {r['filename']} | jour={r['day']} | heure={r['hour']}"
        )

    # --------------------------------------------------------
    # 3. Indexation
    # --------------------------------------------------------
    logger.info("Indexation dans data/index/index.csv...")
    df = indexer.upsert(records)
    logger.info(f"✓ Index créé : {len(df)} entrées")

    # --------------------------------------------------------
    # 4. Téléchargement (optionnel)
    # --------------------------------------------------------
    if not no_download and max_images > 0:
        logger.info(f"Téléchargement des {max_images} premières images...")
        scraper.download_images(records, max_images=max_images)

        # Resynchroniser l'index avec les fichiers réels
        indexer.sync_file_status()
    else:
        logger.info("Téléchargement ignoré (--no-download ou max-images=0).")

    # --------------------------------------------------------
    # 5. Résumé final
    # --------------------------------------------------------
    indexer.print_summary()

    # Vérification de la structure des dossiers
    logger.info("Structure data/ :")
    data_dir = Path("data")
    for p in sorted(data_dir.rglob("*"))[:20]:
        logger.info(f"  {p}")

    logger.info("Phase 1 terminée avec succès.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Phase 1 — Merapi Scraping")
    parser.add_argument("--year", type=int, default=2014)
    parser.add_argument("--month", type=int, default=11)
    parser.add_argument(
        "--max-images", type=int, default=10,
        help="Nombre d'images à télécharger (défaut : 10)"
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Scraper et indexer uniquement, sans télécharger"
    )
    args = parser.parse_args()

    main(
        year=args.year,
        month=args.month,
        max_images=args.max_images,
        no_download=args.no_download,
    )
