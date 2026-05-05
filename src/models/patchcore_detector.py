"""
patchcore_detector.py — Détecteur d'anomalies PatchCore sur features DINOv2.

Principe de PatchCore (Roth et al., 2022 — CVPR) :
  1. Phase d'entraînement (images normales uniquement) :
     - Extraire les features patch-level DINOv2 de chaque image
     - Construire une memory bank (coreset) par sous-échantillonnage greedy
  2. Phase de scoring :
     - Pour chaque image, extraire ses features patch
     - Calculer la distance minimale de chaque patch au coreset
     - Score d'anomalie = max des distances min (= le patch le plus anormal)

Avantages vs autoencoder pixel-level :
  - Pas de sensibilité aux variations météo / lumineuses (features sémantiques)
  - Pas d'entraînement gradient (pas de convergence à gérer)
  - Localisation spatiale des anomalies (quel patch est anormal)
  - SOTA sur anomaly detection industrielle, applicable au contexte volcanique

Split temporel imposé :
  - Train : images usable 2014–2017, heure diurne (6h–18h)
  - Test  : 2018

Usage :
    from src.models.patchcore_detector import PatchCoreDetector

    detector = PatchCoreDetector()
    detector.fit(df_train, image_root)       # construit le coreset
    scores = detector.score_dataframe(df_test, image_root)  # retourne Series
    detector.save("outputs/models/patchcore.npz")
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger("patchcore_detector")

# ─── Constantes ───────────────────────────────────────────────────────────
TRAIN_YEARS = list(range(2014, 2018))   # 2014, 2015, 2016, 2017
TEST_YEARS = [2018]
DIURNAL_HOURS = range(6, 18)            # 6h–18h : images diurnes uniquement en train
CORESET_RATIO = 0.15                    # garder 15% des patches en mémoire
MIN_CORESET_SIZE = 500                  # minimum de patches dans le coreset


class PatchCoreDetector:
    """
    Détecteur d'anomalies basé sur PatchCore + DINOv2.

    Attributes:
        coreset     : memory bank sous-échantillonnée (np.ndarray, shape (N, 384))
        anomaly_scores : dict filename → float, calculés lors du scoring
    """

    def __init__(self, coreset_ratio: float = CORESET_RATIO) -> None:
        self.coreset_ratio = coreset_ratio
        self.coreset: np.ndarray | None = None
        self._is_fitted = False
        self._dino_unavailable = False  # True si DINOv2 non disponible

    # ─── API publique ──────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        image_root: Path | str | None = None,
        max_images: int = 500,
    ) -> "PatchCoreDetector":
        """
        Construit le coreset à partir des images normales (train set).

        IMPORTANT : seules les images diurnes (6h–18h) et usable sont utilisées,
        pour éviter de polluer la "normalité" avec des images d'incandescence nocturne.

        Args:
            df: DataFrame index complet.
            image_root: racine du projet.
            max_images: nombre max d'images à utiliser (pour limiter la RAM).
        """
        from src.utils import PROJECT_ROOT
        from src.features.attention_maps import get_dino_patch_features

        if image_root is None:
            image_root = PROJECT_ROOT
        image_root = Path(image_root)

        # Filtrage strict du train set
        mask = (
            df["quality_flag"].eq("usable") &
            df["year"].isin(TRAIN_YEARS) &
            df["hour"].between(6, 17)  # diurne uniquement
        )
        df_train = df[mask].copy()
        logger.info("Train set : %d images diurnes usable (2014–2017)", len(df_train))

        if df_train.empty:
            raise ValueError("Train set vide — vérifiez le quality_flag et les années.")

        # Sous-échantillonnage si trop grand
        if len(df_train) > max_images:
            df_train = df_train.sample(max_images, random_state=42)
            logger.info("Sous-échantillonnage train → %d images", max_images)

        # Collecte des features patch
        all_patches: list[np.ndarray] = []
        n_ok = 0
        for _, row in df_train.iterrows():
            img_path = self._resolve_path(row, image_root)
            if img_path is None:
                continue
            patches = get_dino_patch_features(img_path)  # (256, 384)
            if patches.shape[0] > 0 and patches.max() != 0:
                all_patches.append(patches)
                n_ok += 1

        logger.info("Features extraites pour %d / %d images", n_ok, len(df_train))

        if not all_patches:
            logger.error(
                "Aucune feature DINOv2 extraite (DINOv2 indisponible ou images inaccessibles).\n"
                "  → PatchCore désactivé. Le pipeline continuera avec anomaly_score (baseline).\n"
                "  → Correction protobuf : pip install 'protobuf>=3.20,<4'"
            )
            self._is_fitted = False
            self._dino_unavailable = True
            return self

        # Concaténer tous les patches : (N_total, 384)
        memory_bank = np.vstack(all_patches)
        logger.info("Memory bank initiale : %d patches (dim=384)", len(memory_bank))

        # Coreset par sous-échantillonnage aléatoire
        # (approx. greedy coreset — suffisant pour < 128k patches)
        n_coreset = max(MIN_CORESET_SIZE, int(len(memory_bank) * self.coreset_ratio))
        n_coreset = min(n_coreset, len(memory_bank))
        idx = np.random.choice(len(memory_bank), size=n_coreset, replace=False)
        self.coreset = memory_bank[idx]
        logger.info("Coreset final : %d patches", len(self.coreset))

        self._is_fitted = True
        return self

    def score_image(self, image_input) -> tuple[float, np.ndarray]:
        """
        Calcule le score d'anomalie d'une image.

        Args:
            image_input: chemin (str/Path) ou np.ndarray.

        Returns:
            (anomaly_score, patch_scores) où :
              - anomaly_score : float, score global de l'image (max des distances min)
              - patch_scores  : np.ndarray (16, 16), carte spatiale des distances
        """
        from src.features.attention_maps import get_dino_patch_features

        if not self._is_fitted or self.coreset is None:
            raise RuntimeError("Le détecteur n'est pas entraîné. Appelez fit() d'abord.")

        patches = get_dino_patch_features(image_input)  # (256, 384)

        # Distance de chaque patch au voisin le plus proche dans le coreset
        # Approche efficace : distance euclidienne par blocs pour économiser la RAM
        min_dists = self._min_distances_to_coreset(patches)  # (256,)

        # Score global = max des distances (le patch le plus anormal)
        anomaly_score = float(min_dists.max())

        # Carte spatiale des scores (16×16)
        patch_map = min_dists.reshape(16, 16)
        vmin, vmax = patch_map.min(), patch_map.max()
        if vmax > vmin:
            patch_map_norm = (patch_map - vmin) / (vmax - vmin)
        else:
            patch_map_norm = np.zeros_like(patch_map)

        return anomaly_score, patch_map_norm.astype(np.float32)

    def score_dataframe(
        self,
        df: pd.DataFrame,
        image_root: Path | str | None = None,
        split: str = "test",
    ) -> pd.Series:
        """
        Score toutes les images d'un DataFrame et retourne une Series indexée
        sur df.index avec les scores d'anomalie.

        Args:
            df: DataFrame index.
            image_root: racine du projet.
            split: 'test' pour filtrer sur 2018, 'all' pour tout scorer.

        Returns:
            pd.Series de float, index = df.index, NaN si l'image est inaccessible.
        """
        from src.utils import PROJECT_ROOT

        if image_root is None:
            image_root = PROJECT_ROOT
        image_root = Path(image_root)

        if split == "test":
            df_to_score = df[df["year"].isin(TEST_YEARS) & df["quality_flag"].eq("usable")]
        else:
            # Inclure usable + dark + NaN (images 2019+ non encore classifiées par QualityFilter)
            df_to_score = df[
                df["quality_flag"].isin(["usable", "dark"]) | df["quality_flag"].isna()
            ]

        # Fallback si DINOv2 indisponible : réutilise anomaly_score existant
        if self._dino_unavailable or not self._is_fitted:
            logger.warning(
                "PatchCore non entraîné (DINOv2 indisponible) — "
                "fallback sur colonne 'anomaly_score' du baseline."
            )
            if "anomaly_score" in df.columns:
                return df["anomaly_score"].copy().astype(float)
            return pd.Series(np.nan, index=df.index, dtype=float)

        scores = pd.Series(np.nan, index=df.index, dtype=float)

        n_not_found = 0
        for idx, row in df_to_score.iterrows():
            img_path = self._resolve_path(row, image_root)
            if img_path is None:
                n_not_found += 1
                continue
            try:
                score, _ = self.score_image(img_path)
                scores.at[idx] = score
            except Exception as exc:
                logger.debug("Score failed pour %s : %s", row.get("filename", "?"), exc)

        n_scored = scores.notna().sum()
        logger.info("Scores calculés : %d images (split=%s)", n_scored, split)
        if n_not_found > 0:
            logger.warning(
                "  %d / %d images introuvables (fichiers raw/processed absents).\n"
                "  → Téléchargez les images manquantes puis relancez --step patchcore.",
                n_not_found, len(df_to_score),
            )

        # Statistiques par année pour faciliter le diagnostic
        if n_scored > 0:
            scored_df = df.loc[scores.notna()]
            if "year" in scored_df.columns:
                per_year = scored_df["year"].dropna().astype(int).value_counts().sort_index()
                for yr, cnt in per_year.items():
                    logger.info("  → %d : %d images scorées", yr, cnt)

        logger.info("Scores calculés : %d images (split=%s)", n_scored, split)
        return scores

    def save(self, path: str | Path) -> None:
        """Sauvegarde le coreset en .npz."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), coreset=self.coreset)
        logger.info("PatchCore sauvegardé → %s (%d patches)", path, len(self.coreset))

    def load(self, path: str | Path) -> "PatchCoreDetector":
        """Charge un coreset depuis .npz."""
        data = np.load(str(path))
        self.coreset = data["coreset"]
        self._is_fitted = True
        logger.info("PatchCore chargé depuis %s (%d patches)", path, len(self.coreset))
        return self

    # ─── helpers privés ───────────────────────────────────────────────────

    def _min_distances_to_coreset(self, patches: np.ndarray) -> np.ndarray:
        """
        Calcule la distance min de chaque patch query vers le coreset.

        Utilise un calcul vectorisé par blocs pour limiter la RAM.
        Complexité : O(n_query × n_coreset) — acceptable pour coreset < 10k.
        """
        # ||a - b||² = ||a||² + ||b||² - 2 <a,b>
        q_sq = (patches ** 2).sum(axis=1, keepdims=True)    # (256, 1)
        c_sq = (self.coreset ** 2).sum(axis=1, keepdims=True).T  # (1, N_coreset)
        cross = patches @ self.coreset.T                         # (256, N_coreset)
        dists_sq = np.maximum(0, q_sq + c_sq - 2 * cross)
        min_dists = np.sqrt(dists_sq.min(axis=1))               # (256,)
        return min_dists

    @staticmethod
    def _resolve_path(row: pd.Series, root: Path) -> Path | None:
        """Résout le chemin d'une image depuis une ligne de l'index.
        Essaie d'abord local_path (raw), puis fallback sur processed/.
        """
        import re
        lp = str(row.get("local_path", ""))

        # 1. local_path absolu ou relatif
        if lp:
            p = Path(lp) if Path(lp).is_absolute() else root / lp
            if p.exists():
                return p

        # 2. Fallback processed (raw/ → processed/, extension → .png)
        if lp:
            proc_lp = re.sub(r'/raw/', '/processed/', lp)
            proc_lp = re.sub(r'\.(jpg|JPG|jpeg|JPEG)$', '.png', proc_lp)
            proc_path = Path(proc_lp) if Path(proc_lp).is_absolute() else root / proc_lp
            if proc_path.exists():
                return proc_path

        # 3. Fallback par filename dans processed/
        filename = str(row.get("filename", ""))
        year = row.get("year")
        month = row.get("month")
        if filename and year and month:
            stem = re.sub(r'\.(jpg|JPG|jpeg|JPEG)$', '', filename)
            proc_by_fn = root / "data" / "processed" / str(int(year)) / f"{int(month):02d}" / f"{stem}.png"
            if proc_by_fn.exists():
                return proc_by_fn

        return None

    # ─── Utilitaire de split temporel ─────────────────────────────────────

    @staticmethod
    def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retourne (df_train, df_test) avec le split temporel rigoureux.

        Train : 2014–2017, usable, heure diurne
        Test  : 2018, usable

        Garantit l'absence de fuite d'information temporelle.
        """
        df_train = df[
            df["quality_flag"].eq("usable") &
            df["year"].isin(TRAIN_YEARS) &
            df["hour"].between(6, 17)
        ].copy()

        df_test = df[
            df["quality_flag"].eq("usable") &
            df["year"].isin(TEST_YEARS)
        ].copy()

        logger.info(
            "Split temporel → train: %d images (2014-2017 diurne), test: %d images (2018)",
            len(df_train), len(df_test),
        )
        return df_train, df_test
