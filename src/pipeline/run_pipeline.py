"""
run_pipeline.py — Orchestrateur unique du pipeline Merapi.

Phases disponibles :
    --download         Scrape + télécharge les images Kalor
    --preprocess       Prétraite raw/ → processed/  (quality_flag calculé)
    --quality          Re-classifie les processed/ et met à jour l'index
    --sync             Réconcilie raw / processed / index.csv
    --full             Toutes les phases dans l'ordre

Filtres :
    --year YYYY        Restreint à une année
    --month MM         Restreint à un mois (nécessite --year)

Options de traitement :
    --force-reprocess  Overwrite les PNG déjà dans processed/
    --rebuild-processed  Supprime processed/ puis relance le preprocessing
                         (confirme avec --yes)
    --max-per-month N  Plafond d'images par mois (défaut : 30)
    --no-index-update  Désactive la mise à jour de l'index après preprocessing

Usage :
    # Pipeline complet
    python -m src.pipeline.run_pipeline --full

    # Preprocessing d'une année
    python -m src.pipeline.run_pipeline --preprocess --year 2019

    # Preprocessing d'un mois + mise à jour index
    python -m src.pipeline.run_pipeline --preprocess --year 2019 --month 6

    # Rebuild total
    python -m src.pipeline.run_pipeline --preprocess --rebuild-processed --yes

    # Download uniquement
    python -m src.pipeline.run_pipeline --download --year 2020
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# ── Racine projet ─────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from src.utils import load_config, setup_logger, safe_sum, PROJECT_ROOT

try:
    from loguru import logger
except ImportError:
    import logging as _l

    logger = _l.getLogger("pipeline")  # type: ignore[assignment]


# ============================================================
# Phases du pipeline
# ============================================================

def phase_sync(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """Phase SYNC : réconcilie raw / processed / index.csv."""
    from src.ingestion.indexer import IndexManager

    print("\n── SYNC ──────────────────────────────────────────────────")
    im = IndexManager(cfg)
    df = im.sync_all()
    print(f"Index synchronisé : {len(df)} entrées")


def phase_download(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """Phase DOWNLOAD : scrape + télécharge les images Kalor."""
    from src.ingestion.downloader import KalorDownloader
    from src.ingestion.indexer import IndexManager

    print("\n── DOWNLOAD ──────────────────────────────────────────────")
    dl = KalorDownloader(cfg)
    im = IndexManager(cfg)

    all_results = dl.download_range(year=args.year, month=args.month)

    if not all_results:
        print("Aucune image téléchargée.")
        return

    # Upsert dans l'index
    flat: list[dict[str, Any]] = [r for v in all_results.values() for r in v]
    n_ok  = sum(bool(r["downloaded"]) for r in flat)
    n_err = len(flat) - n_ok
    im.upsert(flat)

    print(f"✔ {n_ok} images téléchargées | ❌ {n_err} erreurs")
    print(f"Index mis à jour : {len(flat)} records traités")


def phase_preprocess(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """Phase PREPROCESS : raw/ → processed/ avec quality_flag garanti."""
    from src.ingestion.processor import ImageProcessor
    from src.ingestion.indexer import IndexManager

    print("\n── PREPROCESS ────────────────────────────────────────────")

    # Option rebuild : supprime processed/ avant de relancer
    if getattr(args, "rebuild_processed", False):
        proc_base = PROJECT_ROOT / cfg["paths"]["data_processed"]
        if proc_base.exists():
            if not getattr(args, "yes", False):
                confirm = input(
                    f"\n⚠  Supprimer {proc_base} et tout son contenu ? [oui/N] "
                ).strip().lower()
                if confirm not in ("oui", "o", "yes", "y"):
                    print("Annulé.")
                    return
            shutil.rmtree(proc_base)
            print(f"processed/ supprimé : {proc_base}")

    proc = ImageProcessor(cfg)
    im   = IndexManager(cfg)

    max_per = getattr(args, "max_per_month", None) or cfg.get("pipeline", {}).get(
        "max_images_per_month", 30
    )

    all_results = proc.process_all(
        year          = args.year,
        month         = args.month,
        overwrite     = getattr(args, "force_reprocess", False),
        max_per_month = max_per,
    )

    if not all_results:
        print("Aucun résultat de preprocessing.")
        return

    # Mise à jour de l'index
    if not getattr(args, "no_index_update", False):
        flat = [r for v in all_results.values() for r in v]
        n_updated = im.update_quality_bulk(flat)
        print(f"Index mis à jour : {n_updated} entrées quality_flag")


def phase_quality(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """
    Phase QUALITY : re-classifie tous les PNGs traités et met à jour l'index.

    Utile si les seuils ont changé dans settings.yaml ou si l'index
    contient des 'unknown' résiduels.
    """
    from src.ingestion.indexer import IndexManager

    print("\n── QUALITY ───────────────────────────────────────────────")
    im  = IndexManager(cfg)

    # 1. Recalcul depuis les PNGs
    n1 = im.rebuild_quality_from_processed()
    print(f"  Qualité recalculée depuis processed/ : {n1} images")

    # 2. Correction des 'unknown' résiduels qui ont des stats
    pp = cfg["preprocessing"]
    n2 = im.fix_unknown_flags(
        night_thresh     = float(pp.get("night_brightness_threshold", 30)),
        cloud_var_thresh = float(pp.get("cloud_variance_threshold", 50)),
    )
    print(f"  Flags 'unknown' corrigés via stats : {n2}")

    # 3. Résumé
    df = im.load()
    if not df.empty and "quality_flag" in df.columns:
        counts = df["quality_flag"].fillna("unknown").value_counts()
        print("\n  Distribution quality_flag :")
        for flag in ["usable", "dark", "cloudy", "corrupted", "unknown"]:
            c = counts.get(flag, 0)
            print(f"    • {flag:<12}: {c}")


# ============================================================
# CLI
# ============================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline.run_pipeline",
        description=(
            "Pipeline Merapi — orchestrateur unique.\n"
            "Sans phase → affiche un résumé de l'état actuel.\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Phases ────────────────────────────────────────────────
    phase_grp = parser.add_argument_group("Phases")
    phase_grp.add_argument(
        "--full", action="store_true",
        help="Exécute SYNC + DOWNLOAD + PREPROCESS + QUALITY.",
    )
    phase_grp.add_argument(
        "--sync", action="store_true",
        help="Réconcilie raw / processed / index.csv.",
    )
    phase_grp.add_argument(
        "--download", action="store_true",
        help="Scrape et télécharge les images Kalor.",
    )
    phase_grp.add_argument(
        "--preprocess", action="store_true",
        help="Prétraite les images raw/ → processed/.",
    )
    phase_grp.add_argument(
        "--quality", action="store_true",
        help="Re-classifie les images traitées et met à jour l'index.",
    )

    # ── Filtres ───────────────────────────────────────────────
    filter_grp = parser.add_argument_group("Filtres")
    filter_grp.add_argument(
        "--year", type=int, default=None,
        help="Année à traiter (ex: 2019). Défaut : toutes.",
    )
    filter_grp.add_argument(
        "--month", type=int, default=None,
        help="Mois à traiter (1–12). Nécessite --year.",
    )

    # ── Options ───────────────────────────────────────────────
    opt_grp = parser.add_argument_group("Options")
    opt_grp.add_argument(
        "--force-reprocess", action="store_true",
        help="Overwrite les PNG déjà dans processed/.",
    )
    opt_grp.add_argument(
        "--rebuild-processed", action="store_true",
        help="Supprime processed/ et relance le preprocessing depuis zéro.",
    )
    opt_grp.add_argument(
        "--yes", action="store_true",
        help="Confirme automatiquement les opérations destructrices (--rebuild-processed).",
    )
    opt_grp.add_argument(
        "--max-per-month", type=int, default=None, metavar="N",
        help="Plafond d'images par mois (défaut : 30).",
    )
    opt_grp.add_argument(
        "--no-index-update", action="store_true",
        help="Désactive la mise à jour de l'index après preprocessing.",
    )

    return parser


def _print_status(cfg: dict[str, Any]) -> None:
    """Affiche l'état actuel du projet (sans modifier quoi que ce soit)."""
    from src.ingestion.indexer import IndexManager

    im = IndexManager(cfg)
    df = im.load()

    print(f"\n{'='*55}")
    print("Merapi Pipeline — état actuel")
    print(f"{'='*55}")

    if df.empty:
        print("Index vide — lancez --sync ou --full pour initialiser.")
        return

    print(f"Total indexé     : {len(df)}")

    n_dl = int((df["downloaded"] == True).sum())
    print(f"Téléchargées     : {n_dl}")

    if "processed" in df.columns:
        n_proc = int(df["processed"].fillna(False).astype(bool).sum())
        print(f"Traitées (PNG)   : {n_proc}")

    if "quality_flag" in df.columns:
        counts = df["quality_flag"].fillna("unknown").value_counts()
        print("\nquality_flag :")
        for flag in ["usable", "dark", "cloudy", "corrupted", "unknown"]:
            c = counts.get(flag, 0)
            if c:
                print(f"  • {flag:<12}: {c}")

    if "year" in df.columns:
        years = sorted(df["year"].dropna().astype(int).unique())
        if years:
            print(f"\nAnnées couvertes : {years[0]}–{years[-1]}")

    print(f"{'='*55}\n")


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    # Validation --month sans --year
    if args.month is not None and args.year is None:
        parser.error("--month nécessite --year (ex: --year 2019 --month 6)")

    cfg = load_config()
    setup_logger(cfg)

    # ── Aucune phase → afficher le statut ─────────────────────
    no_phase = not any([
        args.full, args.sync, args.download,
        args.preprocess, args.quality,
    ])
    if no_phase:
        _print_status(cfg)
        return

    print(f"\n{'='*55}")
    print("Merapi Pipeline")
    if args.year:
        print(f"  Scope : {args.year}" + (f"/{args.month:02d}" if args.month else ""))
    print(f"{'='*55}")

    # ── Exécution des phases ───────────────────────────────────
    if args.full or args.sync:
        phase_sync(cfg, args)

    if args.full or args.download:
        phase_download(cfg, args)

    if args.full or args.preprocess:
        phase_preprocess(cfg, args)

    if args.full or args.quality:
        phase_quality(cfg, args)

    # ── Résumé final ──────────────────────────────────────────
    _print_status(cfg)


if __name__ == "__main__":
    main()
