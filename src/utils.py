"""
utils.py — Utilitaires partagés du projet Merapi Anomaly Detection.

Contenu :
- Chargement de la configuration YAML
- Initialisation du logger (loguru)
- Création des dossiers du projet
- Helpers de chemins et de dates
"""

from __future__ import annotations

import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
try:
    from loguru import logger
except ModuleNotFoundError:
    class _FallbackLogger:
        """Compatibilite minimale avec loguru si le package n'est pas installe."""

        def __init__(self) -> None:
            self._logger = logging.getLogger("merapi")
            self._logger.setLevel(logging.INFO)
            if not self._logger.handlers:
                handler = logging.StreamHandler(sys.stderr)
                handler.setFormatter(
                    logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s")
                )
                self._logger.addHandler(handler)

        def remove(self, *args: Any, **kwargs: Any) -> None:
            return

        def add(self, *args: Any, **kwargs: Any) -> None:
            return

        def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
            self._logger.debug(message)

        def info(self, message: str, *args: Any, **kwargs: Any) -> None:
            self._logger.info(message)

        def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
            self._logger.warning(message)

        def error(self, message: str, *args: Any, **kwargs: Any) -> None:
            self._logger.error(message)

        def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
            self._logger.exception(message)

    logger = _FallbackLogger()


# ============================================================
# Racine du projet — détectée dynamiquement
# ============================================================

def get_project_root() -> Path:
    """
    Retourne la racine du projet en remontant depuis ce fichier.
    Fonctionne quelle que soit la façon dont le script est appelé.
    """
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()


# ============================================================
# Configuration
# ============================================================

def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Charge le fichier settings.yaml et retourne un dictionnaire.

    Args:
        config_path: chemin vers le YAML. Si None, cherche dans config/settings.yaml.

    Returns:
        dict contenant toute la configuration.

    Raises:
        FileNotFoundError si le fichier est introuvable.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "settings.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Fichier de configuration introuvable : {config_path}\n"
            f"Vérifiez que config/settings.yaml existe à la racine du projet."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


# ============================================================
# Logging
# ============================================================

_logger_initialized = False


def setup_logger(config: dict[str, Any] | None = None) -> None:
    """
    Initialise le logger loguru à partir de la configuration.
    Peut être appelé plusieurs fois sans duplication (idempotent).

    Args:
        config: dictionnaire de configuration (section 'logging').
                Si None, charge la configuration par défaut.
    """
    global _logger_initialized
    if _logger_initialized:
        return

    if config is None:
        config = load_config()

    log_cfg = config.get("logging", {})
    level = log_cfg.get("level", "INFO")
    log_to_file = log_cfg.get("log_to_file", True)
    log_dir = PROJECT_ROOT / log_cfg.get("log_dir", "logs")
    log_filename = log_cfg.get("log_filename", "merapi_{date}.log")
    rotation = log_cfg.get("rotation", "10 MB")
    retention = log_cfg.get("retention", "30 days")

    # Remplacer le handler par défaut par un format propre
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{line}</cyan> — {message}",
    )

    if log_to_file:
        ensure_dir(log_dir)
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / log_filename.replace("{date}", date_str)
        logger.add(
            str(log_file),
            level=level,
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
        )

    _logger_initialized = True
    logger.info(f"Logger initialisé — niveau : {level}")


# ============================================================
# Gestion des dossiers
# ============================================================

def ensure_dir(path: str | Path) -> Path:
    """
    Crée un dossier (et ses parents) s'il n'existe pas.

    Args:
        path: chemin du dossier à créer.

    Returns:
        Path résolu du dossier créé ou existant.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_project_dirs(config: dict[str, Any]) -> dict[str, Path]:
    """
    Crée tous les dossiers du projet définis dans config['paths'].

    Args:
        config: dictionnaire de configuration complet.

    Returns:
        dict {nom_clé: Path} pour chaque dossier créé.
    """
    paths_cfg = config.get("paths", {})
    created = {}

    # Dossiers (pas les fichiers comme index_file)
    dir_keys = [k for k, v in paths_cfg.items() if not str(v).endswith(".csv")]

    for key in dir_keys:
        rel_path = paths_cfg[key]
        abs_path = ensure_dir(PROJECT_ROOT / rel_path)
        created[key] = abs_path

    logger.info(f"Dossiers projet vérifiés/créés : {len(created)} répertoires")
    return created


# ============================================================
# Helpers de chemins
# ============================================================

def get_raw_image_dir(config: dict[str, Any], year: int, month: int) -> Path:
    """
    Retourne le chemin local de stockage des images brutes pour un mois donné.
    Structure : data/raw/{year}/{month:02d}/

    Args:
        config: configuration du projet.
        year: année (ex. 2014).
        month: mois (1–12).

    Returns:
        Path du dossier, créé si nécessaire.
    """
    base = PROJECT_ROOT / config["paths"]["data_raw"]
    dir_path = base / str(year) / f"{month:02d}"
    return ensure_dir(dir_path)


def get_processed_image_dir(config: dict[str, Any], year: int, month: int) -> Path:
    """
    Retourne le chemin local des images prétraitées pour un mois donné.
    Structure : data/processed/{year}/{month:02d}/
    """
    base = PROJECT_ROOT / config["paths"]["data_processed"]
    dir_path = base / str(year) / f"{month:02d}"
    return ensure_dir(dir_path)


def get_index_path(config: dict[str, Any]) -> Path:
    """Retourne le chemin absolu de l'index CSV."""
    return PROJECT_ROOT / config["paths"]["index_file"]


def get_monthly_page_url(config: dict[str, Any], year: int, month: int) -> str:
    """
    Construit l'URL de la page mensuelle d'images.

    Pattern observé sur le site :
        {base_url}/{year}_{mm:02d}/{year}_{mon_abbr}.html
    Exemple :
        .../domerapi/2014_11/2014_nov.html

    Args:
        config: configuration du projet.
        year: année.
        month: mois (1–12).

    Returns:
        URL complète de la page mensuelle.
    """
    base_url = config["source"]["base_url"]
    month_abbrs = config["source"]["month_abbrs"]
    mon_abbr = month_abbrs[month]
    return f"{base_url}/{year}_{month:02d}/{year}_{mon_abbr}.html"


# ============================================================
# Helpers de dates / nommage
# ============================================================

def parse_filename_datetime(filename: str) -> dict[str, Any] | None:
    """
    Tente d'extraire une date/heure depuis un nom de fichier image.

    Patterns courants observés sur le site Merapi :
        - merapi_YYYYMMDD_HHMMSS.jpg
        - YYYYMMDD_HHMMSS.jpg
        - merapi_YYYYMMDD_HH.jpg
        (les patterns réels devront être validés sur les données)

    Args:
        filename: nom du fichier (sans chemin).

    Returns:
        dict avec keys {year, month, day, hour, minute} ou None si non parsable.
    """
    import re

    stem = Path(filename).stem

    # Pattern 1 : YYYYMMDD_HHMMSS (ex. 20141115_143022)
    m = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})?", stem)
    if m:
        return {
            "year": int(m.group(1)),
            "month": int(m.group(2)),
            "day": int(m.group(3)),
            "hour": int(m.group(4)),
            "minute": int(m.group(5)),
            "second": int(m.group(6)) if m.group(6) else 0,
        }

    # Pattern 2 : YYYYMMDD_HH (ex. 20141115_14)
    m = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})", stem)
    if m:
        return {
            "year": int(m.group(1)),
            "month": int(m.group(2)),
            "day": int(m.group(3)),
            "hour": int(m.group(4)),
            "minute": 0,
            "second": 0,
        }

    # Pattern 3 : YYYY-MM-DD_HH-MM
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})", stem)
    if m:
        return {
            "year": int(m.group(1)),
            "month": int(m.group(2)),
            "day": int(m.group(3)),
            "hour": int(m.group(4)),
            "minute": int(m.group(5)),
            "second": 0,
        }

    # Pattern 4 : YYYY-MM-DD_HH.MM.SS (ex. Kalor_Canon_2014-11-12_17.13.07)
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})\.(\d{2})(?:\.(\d{2}))?", stem)
    if m:
        return {
            "year": int(m.group(1)),
            "month": int(m.group(2)),
            "day": int(m.group(3)),
            "hour": int(m.group(4)),
            "minute": int(m.group(5)),
            "second": int(m.group(6)) if m.group(6) else 0,
        }

    return None  # pattern non reconnu → à enrichir après inspection des vraies données


# ============================================================
# Agrégations sûres (robustesse aux valeurs None / NaN)
# ============================================================

def safe_sum(iterable) -> int:
    """
    Somme d'une séquence de booléens/entiers en ignorant silencieusement
    les valeurs None et NaN.

    Exemple :
        safe_sum([True, False, None, True])  →  2
        safe_sum([1, None, 0, 1])            →  2

    Args:
        iterable: séquence de valeurs bool | int | None.

    Returns:
        Somme entière (0 si tout est None/vide).
    """
    import math

    total = 0
    for v in iterable:
        if v is None:
            continue
        try:
            fv = float(v)
            if math.isnan(fv):
                continue
            total += int(fv)
        except (TypeError, ValueError):
            continue
    return total


def safe_mean(values, default: float = 0.0) -> float:
    """
    Moyenne d'une séquence en ignorant silencieusement None et NaN.

    Args:
        values:  séquence de valeurs numériques, None acceptés.
        default: valeur retournée si la séquence valide est vide.

    Returns:
        Moyenne (float) ou `default`.
    """
    import math

    valid: list[float] = []
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
            if not math.isnan(fv):
                valid.append(fv)
        except (TypeError, ValueError):
            continue
    return sum(valid) / len(valid) if valid else default
