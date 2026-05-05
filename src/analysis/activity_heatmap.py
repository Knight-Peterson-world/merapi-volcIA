"""
activity_heatmap.py — Carte d'activité spatiale agrégée pour le Merapi.

Ce module transforme les scores PatchCore (par patch 14×14 pixels du ViT DINOv2-small)
en carte de chaleur 2D indiquant les zones du volcan les plus anormales.

Fonctions principales :
    compute_activity_heatmap(df, scores_path)  → np.ndarray (GRID, GRID)
    detect_active_clusters(heatmap, threshold) → list[dict]
    activity_score_per_image(patch_scores)     → float
    timeline_activity(df, scores_path)         → pd.DataFrame

Résolution de la grille :
    DINOv2-small traite des images 224×224 → 16×16 patches (patch_size=14).
    La grille par défaut est donc 16×16 = 256 cellules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from loguru import logger  # type: ignore
except ImportError:
    logger = logging.getLogger("activity_heatmap")  # type: ignore

# ── Constantes ────────────────────────────────────────────────────────────

GRID_SIZE = 16   # 16×16 patches pour DINOv2-small sur image 224×224
N_PATCHES = GRID_SIZE * GRID_SIZE  # 256 patches par image


# ── Fonctions utilitaires ─────────────────────────────────────────────────

def activity_score_per_image(patch_scores: np.ndarray) -> float:
    """
    Score d'activité global d'une image à partir de son vecteur de scores par patch.

    Utilise le 90e percentile (robuste aux outliers de bord) plutôt que la moyenne,
    car une anomalie volcanique est souvent localisée (quelques patches actifs).

    Args:
        patch_scores: vecteur de scores 1D (longueur N_PATCHES) ou scalaire.

    Returns:
        float — score d'activité de l'image.
    """
    arr = np.asarray(patch_scores, dtype=np.float32).ravel()
    if arr.size == 0:
        return 0.0
    if arr.size == 1:
        return float(arr[0])
    # P90 pour capturer les zones actives sans être dominé par le bruit
    return float(np.nanpercentile(arr, 90))


def compute_activity_heatmap(
    df: pd.DataFrame,
    scores_path: str | Path | None = None,
    patch_score_col: str = "patchcore_score",
    grid_size: int = GRID_SIZE,
    aggregation: str = "mean",
) -> np.ndarray:
    """
    Calcule une heatmap 2D (grid_size × grid_size) d'activité volcanique.

    La heatmap représente l'anomalie spatiale agrégée sur toutes les images :
    chaque cellule (i, j) est le score moyen (ou max) des images dont le patch
    correspondant à la position (i, j) est le plus anomal.

    Comme PatchCore retourne un score scalaire par image (pas par patch dans
    l'implémentation actuelle), on simule une distribution spatiale via
    une heuristique basée sur les features disponibles (thermal_gradient,
    bright_pixel_ratio, edge_density). En l'absence de features, la heatmap
    est uniforme pondérée par le score image.

    Args:
        df: DataFrame index avec colonne 'filename' et features optionnelles.
        scores_path: chemin vers un CSV contenant au moins ['filename', 'patchcore_score'].
                     Si None, utilise la colonne 'patchcore_score' de df si présente.
        patch_score_col: nom de la colonne de score.
        grid_size: taille de la grille (défaut 16 pour DINOv2-small).
        aggregation: 'mean' ou 'max' pour l'agrégation spatiale.

    Returns:
        np.ndarray shape (grid_size, grid_size) — heatmap normalisée [0, 1].
    """
    # ── Chargement des scores ──────────────────────────────────────────────
    if scores_path is not None:
        try:
            sc = pd.read_csv(Path(scores_path))
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
            sc = pd.DataFrame()
        if not sc.empty and "filename" in sc.columns and patch_score_col in sc.columns:
            sc_dedup = sc.drop_duplicates("filename").set_index("filename")[patch_score_col]
            df = df.copy()
            df[patch_score_col] = df["filename"].map(sc_dedup)

    if patch_score_col not in df.columns:
        logger.warning("Colonne '%s' absente — heatmap vide.", patch_score_col)
        return np.zeros((grid_size, grid_size), dtype=np.float32)

    df_scored = df[df[patch_score_col].notna()].copy()
    if df_scored.empty:
        logger.warning("Aucun score disponible — heatmap vide.")
        return np.zeros((grid_size, grid_size), dtype=np.float32)

    scores = pd.to_numeric(df_scored[patch_score_col], errors="coerce").fillna(0.0).values

    # ── Construction de la heatmap spatiale ───────────────────────────────
    # Stratégie : simuler une distribution spatiale plausible basée sur les
    # features physiques si disponibles, sinon distribuer uniformément.
    heatmap_accum = np.zeros((grid_size, grid_size), dtype=np.float64)
    heatmap_count = np.zeros((grid_size, grid_size), dtype=np.float64)

    has_features = all(
        c in df_scored.columns
        for c in ["thermal_gradient", "bright_zone_area", "edge_density"]
    )

    if not has_features:
        # Vectorisé : distribution uniforme → pas de boucle Python sur 76k images
        nonzero_mask = scores > 0
        n_nonzero = int(nonzero_mask.sum())
        if n_nonzero > 0:
            heatmap_accum = np.full((grid_size, grid_size), float(scores[nonzero_mask].sum()),
                                    dtype=np.float64)
            heatmap_count = np.full((grid_size, grid_size), float(n_nonzero),
                                    dtype=np.float64)
    else:
        for idx, (_, row) in enumerate(df_scored.iterrows()):
            score = scores[idx]
            if score == 0.0:
                continue
            patch_weights = _spatial_weights_from_features(
                thermal_grad=float(row.get("thermal_gradient", 0.0)),
                bright_zone=float(row.get("bright_zone_area", 0.0)),
                edge_density=float(row.get("edge_density", 0.0)),
                bright_ratio=float(row.get("bright_pixel_ratio", 0.0)),
                grid_size=grid_size,
            )
            heatmap_accum += patch_weights * score
            heatmap_count += (patch_weights > 0).astype(np.float64)

    # Agrégation
    if aggregation == "max":
        heatmap = heatmap_accum
    else:  # mean
        with np.errstate(invalid="ignore", divide="ignore"):
            heatmap = np.where(heatmap_count > 0, heatmap_accum / heatmap_count, 0.0)

    # Normalisation [0, 1]
    hmax = heatmap.max()
    if hmax > 0:
        heatmap = heatmap / hmax

    return heatmap.astype(np.float32)


def _spatial_weights_from_features(
    thermal_grad: float,
    bright_zone: float,
    edge_density: float,
    bright_ratio: float,
    grid_size: int = GRID_SIZE,
) -> np.ndarray:
    """
    Génère une carte de pondération spatiale (grid_size×grid_size) à partir
    des features physiques d'une image.

    Modèle simplifié de localisation volcanique :
      - Activité thermique élevée → centre/bas (cratère → pente)
      - Bords denses → contour du cratère (anneau central)
      - Zones lumineuses → bas de la pente (coulée de lave descendante)

    Returns:
        np.ndarray shape (grid_size, grid_size) de poids positifs.
    """
    rng = np.linspace(0, 1, grid_size)
    row_grid, col_grid = np.meshgrid(rng, rng, indexing="ij")

    # Base : gaussian centré sur (0.4, 0.5) → légèrement en haut (cratère)
    dist_crater = np.sqrt((row_grid - 0.4) ** 2 + (col_grid - 0.5) ** 2)
    w_crater = np.exp(-dist_crater ** 2 / (2 * 0.25 ** 2))

    # Activité thermique (gradient) → renforce les zones de transition
    dist_edge = np.sqrt((row_grid - 0.5) ** 2 + (col_grid - 0.5) ** 2)
    w_edge = np.exp(-((dist_edge - 0.3) ** 2) / (2 * 0.15 ** 2))

    # Lave / bright zone → bas de l'image (pente descendante)
    w_lava = np.exp(-((row_grid - 0.75) ** 2) / (2 * 0.2 ** 2))

    # Combinaison pondérée selon les features
    weight = (
        w_crater * (0.4 + 0.6 * thermal_grad)
        + w_edge * edge_density
        + w_lava * (bright_zone + bright_ratio)
        + 0.1  # bruit de plancher (évite les zones nulles)
    )

    return np.maximum(weight, 0.0)


def detect_active_clusters(
    heatmap: np.ndarray,
    threshold: float = 0.5,
    min_size: int = 2,
) -> list[dict[str, Any]]:
    """
    Détecte les clusters actifs dans une heatmap par seuillage + étiquetage connexe.

    Args:
        heatmap: np.ndarray 2D normalisée [0, 1] (sortie de compute_activity_heatmap).
        threshold: seuil d'activité (0.5 = 50 % du max).
        min_size: nombre minimal de cellules pour qu'un cluster soit retenu.

    Returns:
        Liste de dicts avec clés :
          - 'row', 'col'    : centre de masse du cluster
          - 'score'         : score moyen du cluster
          - 'size'          : nombre de cellules
          - 'bbox'          : (row_min, col_min, row_max, col_max)
    """
    if heatmap.ndim != 2:
        raise ValueError(f"heatmap doit être 2D, reçu shape {heatmap.shape}")

    binary = (heatmap >= threshold).astype(np.uint8)
    if binary.sum() == 0:
        return []

    clusters = _connected_components(binary)
    result = []

    for label_id, cells in clusters.items():
        if len(cells) < min_size:
            continue
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        scores = [float(heatmap[r, c]) for r, c in cells]
        result.append({
            "row": float(np.mean(rows)),
            "col": float(np.mean(cols)),
            "score": float(np.mean(scores)),
            "size": len(cells),
            "bbox": (min(rows), min(cols), max(rows), max(cols)),
        })

    # Trier par score décroissant
    result.sort(key=lambda x: x["score"], reverse=True)
    return result


def _connected_components(binary: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    """Étiquetage des composantes connexes 2D (4-connexité) par BFS."""
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    label_id = 0
    components: dict[int, list[tuple[int, int]]] = {}

    for start_r in range(h):
        for start_c in range(w):
            if binary[start_r, start_c] == 0 or visited[start_r, start_c]:
                continue
            # BFS
            queue = [(start_r, start_c)]
            visited[start_r, start_c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                r, c = queue.pop(0)
                cells.append((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and binary[nr, nc] == 1:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            components[label_id] = cells
            label_id += 1

    return components


def timeline_activity(
    df: pd.DataFrame,
    scores_path: str | Path | None = None,
    patch_score_col: str = "patchcore_score",
    resample: str = "D",
) -> pd.DataFrame:
    """
    Construit une timeline du score d'activité agrégé par période temporelle.

    Args:
        df: DataFrame index avec colonnes ['filename', 'year', 'month', 'day', 'hour'].
        scores_path: chemin optionnel vers un CSV de scores PatchCore.
        patch_score_col: colonne de score.
        resample: fréquence de rééchantillonnage ('D'=jour, 'W'=semaine, 'ME'=mois).

    Returns:
        DataFrame avec colonnes ['datetime', 'mean_score', 'max_score', 'n_images'].
    """
    df_work = df.copy()

    # Charger scores si nécessaire
    if scores_path is not None:
        try:
            sc = pd.read_csv(Path(scores_path))
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
            sc = pd.DataFrame()
        if not sc.empty and "filename" in sc.columns and patch_score_col in sc.columns:
            sc_dedup = sc.drop_duplicates("filename").set_index("filename")[patch_score_col]
            df_work[patch_score_col] = df_work["filename"].map(sc_dedup)

    if patch_score_col not in df_work.columns:
        return pd.DataFrame(columns=["datetime", "mean_score", "max_score", "n_images"])

    df_work = df_work[df_work[patch_score_col].notna()].copy()
    df_work[patch_score_col] = pd.to_numeric(df_work[patch_score_col], errors="coerce")

    # Construire la colonne datetime
    for col in ["year", "month", "day", "hour"]:
        df_work[col] = pd.to_numeric(df_work.get(col, pd.Series(0, index=df_work.index)), errors="coerce")

    df_work["datetime"] = pd.to_datetime(
        df_work[["year", "month", "day", "hour"]].assign(
            minute=0, second=0,
        ),
        errors="coerce",
    )
    df_work = df_work[df_work["datetime"].notna()].copy()

    if df_work.empty:
        return pd.DataFrame(columns=["datetime", "mean_score", "max_score", "n_images"])

    df_work = df_work.set_index("datetime").sort_index()
    agg = df_work[patch_score_col].resample(resample).agg(
        mean_score="mean",
        max_score="max",
        n_images="count",
    ).reset_index()
    agg.columns = ["datetime", "mean_score", "max_score", "n_images"]
    return agg.dropna(subset=["mean_score"])
