"""
anomaly.py — Stub AnomalyDetector (PatchCore / DINOv2).

Ce module est un placeholder qui ne crashe pas.
Il enregistre un warning indiquant qu'il n'est pas encore implémenté
et retourne des scores nuls par défaut.

Usage (futur) :
    from src.features.anomaly import AnomalyDetector
    detector = AnomalyDetector(config)
    scores = detector.score_batch(image_paths)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging as _l

    logger = _l.getLogger("features.anomaly")  # type: ignore[assignment]

_NOT_IMPLEMENTED_MSG = (
    "AnomalyDetector : module non encore implémenté. "
    "Les scores retournés sont des valeurs nulles (placeholder). "
    "Implémentation PatchCore/DINOv2 à venir."
)


class AnomalyDetector:
    """
    Détecteur d'anomalies volcaniques (stub).

    Placeholder pour l'intégration PatchCore + DINOv2.
    Toutes les méthodes logguent un warning et retournent des valeurs neutres.

    Args:
        config: dictionnaire de configuration (settings.yaml).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        logger.warning(_NOT_IMPLEMENTED_MSG)

    def fit(self, image_paths: list[Path]) -> None:
        """Entraîne le modèle sur un ensemble d'images normales (non impl.)."""
        logger.warning(
            f"AnomalyDetector.fit() : non implémenté ({len(image_paths)} images ignorées)."
        )

    def score(self, image_path: Path) -> float:
        """
        Retourne un score d'anomalie pour une image.

        Returns:
            0.0 (placeholder).
        """
        logger.warning(
            f"AnomalyDetector.score() : non implémenté. Retourne 0.0 pour {image_path.name}"
        )
        return 0.0

    def score_batch(
        self,
        image_paths: list[Path],
    ) -> list[dict[str, Any]]:
        """
        Retourne des scores d'anomalie pour une liste d'images.

        Returns:
            Liste de dicts : {'filename': str, 'anomaly_score': 0.0}
        """
        logger.warning(
            f"AnomalyDetector.score_batch() : non implémenté. "
            f"Retourne 0.0 pour {len(image_paths)} images."
        )
        return [
            {"filename": p.name, "anomaly_score": 0.0}
            for p in image_paths
        ]

    def is_available(self) -> bool:
        """Retourne False — modèle non encore disponible."""
        return False
