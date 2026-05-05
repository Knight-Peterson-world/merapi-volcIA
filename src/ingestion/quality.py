"""
quality.py — Classificateur de qualité d'image, déterministe et robuste.

Classifie chaque image parmi : usable / dark / cloudy / corrupted
à partir de ses statistiques de pixels (aucun modèle IA).

Toujours appelé sur l'image **processée** (PNG 256×256 niveaux de gris).

Garanties :
  - quality_flag  : jamais None, jamais 'unknown'
  - is_night      : toujours bool (jamais None)
  - Robuste aux images corrompues (retourne 'corrupted', pas de crash)

Usage :
    from src.ingestion.quality import QualityClassifier
    from src.utils import load_config

    qc = QualityClassifier.from_config(load_config())
    result = qc.classify(Path("data/processed/2019/06/kalor_foo.png"))
    # → {'quality_flag': 'usable', 'is_night': False, 'mean_brightness': 143.2, ...}
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

try:
    from loguru import logger
except ImportError:
    import logging as _l

    logger = _l.getLogger("ingestion.quality")  # type: ignore[assignment]

# ── Labels canoniques ──────────────────────────────────────────────────────
QUALITY_USABLE    = "usable"
QUALITY_DARK      = "dark"
QUALITY_CLOUDY    = "cloudy"
QUALITY_CORRUPTED = "corrupted"

VALID_FLAGS = {QUALITY_USABLE, QUALITY_DARK, QUALITY_CLOUDY, QUALITY_CORRUPTED}


class QualityClassifier:
    """
    Classifieur de qualité déterministe basé sur les statistiques de pixels.

    Ordre de priorité :
        1. corrupted → image illisible ou fichier trop petit
        2. dark      → luminosité moyenne < night_thresh   (image nocturne)
        3. cloudy    → variance < cloud_var_thresh          (image uniforme = nuage)
        4. usable    → reste
    """

    def __init__(
        self,
        night_thresh: float = 30.0,
        cloud_var_thresh: float = 50.0,
        min_file_bytes: int = 1000,
    ) -> None:
        self.night_thresh     = night_thresh
        self.cloud_var_thresh = cloud_var_thresh
        self.min_file_bytes   = min_file_bytes

    # ----------------------------------------------------------
    # Classification d'une image
    # ----------------------------------------------------------

    def classify(self, image_path: Path) -> dict[str, Any]:
        """
        Classifie une image et retourne ses indicateurs de qualité.

        Args:
            image_path: chemin vers le fichier image (brut ou traité).

        Returns:
            dict avec les champs :
                quality_flag    : str  — jamais None, jamais 'unknown'
                is_night        : bool — jamais None
                mean_brightness : float | None
                std_brightness  : float | None
                variance        : float | None
                error           : str  | None — None si succès
        """
        result: dict[str, Any] = {
            "quality_flag":    QUALITY_CORRUPTED,
            "is_night":        False,
            "mean_brightness": None,
            "std_brightness":  None,
            "variance":        None,
            "error":           None,
        }

        # 1. Existence et taille minimale
        if not image_path.exists():
            result["error"] = f"Fichier inexistant : {image_path}"
            return result

        file_size = image_path.stat().st_size
        if file_size < self.min_file_bytes:
            result["error"] = f"Fichier trop petit : {file_size} octets"
            return result

        # 2. Chargement
        try:
            arr = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        except (UnidentifiedImageError, Exception) as exc:
            result["error"] = f"Impossible d'ouvrir l'image : {exc}"
            logger.debug(f"classify — image corrompue {image_path.name}: {exc}")
            return result

        # 3. Statistiques de pixels
        mean_b = float(arr.mean())
        std_b  = float(arr.std())
        var_b  = float(arr.var())

        result["mean_brightness"] = mean_b
        result["std_brightness"]  = std_b
        result["variance"]        = var_b

        # 4. Classification (ordre de priorité strict)
        if mean_b < self.night_thresh:
            result["quality_flag"] = QUALITY_DARK
            result["is_night"]     = True
        elif var_b < self.cloud_var_thresh:
            result["quality_flag"] = QUALITY_CLOUDY
            result["is_night"]     = False
        else:
            result["quality_flag"] = QUALITY_USABLE
            result["is_night"]     = False

        logger.debug(
            f"{image_path.name} → {result['quality_flag']} "
            f"| lum={mean_b:.1f} | var={var_b:.1f}"
        )
        return result

    # ----------------------------------------------------------
    # Classification par lot
    # ----------------------------------------------------------

    def classify_batch(
        self,
        paths: list[Path],
    ) -> list[dict[str, Any]]:
        """
        Classifie une liste d'images.

        Args:
            paths: liste de chemins.

        Returns:
            Liste de dicts (un par image), chacun avec 'filename' et 'path' en plus.
        """
        results = []
        for p in paths:
            try:
                r = self.classify(p)
            except Exception as exc:
                logger.error(f"classify_batch — erreur inattendue sur {p.name}: {exc}")
                r = {
                    "quality_flag":    QUALITY_CORRUPTED,
                    "is_night":        False,
                    "mean_brightness": None,
                    "std_brightness":  None,
                    "variance":        None,
                    "error":           f"{type(exc).__name__}: {exc}",
                }
            r["filename"] = p.name
            r["path"]     = str(p)
            results.append(r)
        return results

    # ----------------------------------------------------------
    # Construction depuis la configuration
    # ----------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QualityClassifier":
        """Construit un QualityClassifier depuis un dict de configuration."""
        pp = config.get("preprocessing", {})
        dl = config.get("download", {})
        return cls(
            night_thresh     = float(pp.get("night_brightness_threshold", 30.0)),
            cloud_var_thresh = float(pp.get("cloud_variance_threshold", 50.0)),
            min_file_bytes   = int(dl.get("min_file_size_bytes", 1000)),
        )
