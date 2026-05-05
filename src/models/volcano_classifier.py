"""
volcano_classifier.py — Classificateur d'événements volcaniques multi-classes.

Classes :
    pyroclastique  — écoulement pyroclastique rapide (danger élevé)
    lave           — coulée de lave (activité effusive, incandescence)
    nuage          — couverture nuageuse météorologique (faux positif fréquent)
    normal         — activité de fond (cratère stable, légère fumée blanche)

Entrées :
    - Features physiques de FEATURE_NAMES (physical_features.py)
    - patchcore_score (score d'anomalie DINOv2+PatchCore, optionnel)

Modèle :
    RandomForestClassifier (principal) avec fallback SVM.
    Si moins de 4 images labellisées : règles heuristiques basées sur seuils.

Usage :
    clf = VolcanoClassifier()
    clf.fit(df_features, labels)          # entraînement
    preds = clf.predict(df_features)      # prédiction
    clf.save("outputs/models/volcano_clf.pkl")
    clf2 = VolcanoClassifier.load("outputs/models/volcano_clf.pkl")
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from loguru import logger  # type: ignore
except ImportError:
    logger = logging.getLogger("volcano_classifier")  # type: ignore

from src.features.physical_features import FEATURE_NAMES

# ── Constantes ────────────────────────────────────────────────────────────

VOLCANO_CLASSES = ["pyroclastique", "lave", "nuage", "normal"]

# Seuils heuristiques (calibrés sur images Merapi Kalor)
# Valeurs recalibrées sur 848 images 2014-2018 :
#   patchcore_score (brut) : p95=44.7  p99=47.3  max=51.3
#   texture_roughness      : p90=0.00033  p99=0.00058
#   bright_pixel_ratio     : p50=0.20  p95=0.93
#   lbp_entropy            : p50=3.75   p50_low=<3.5 → nuage
#   pixel_diff_mean        : p95=0.127
#   NB: cv2 absent → optical_flow_mag=0, edge_density=0, contour_convexity=1 toujours
_HEURISTIC_THRESHOLDS = {
    "optical_flow_mag":     2.0,   # > seuil → pyroclastique (cv2 requis, sinon 0)
    "patchcore_high":       46.5,  # > p98 → pyroclastique fort
    "patchcore_mid":        44.0,  # > p95 → pyroclastique avec texture
    "texture_roughness":    0.00033, # > p90 → texture chaotique
    "tchange_high":         0.12,  # > p95 → changement brutal
    "bright_night":         0.02,  # > 2% la nuit → incandescence lave
    "bright_day":           0.92,  # > p95 le jour → surexposition thermique
    "nuage_patchcore_max":  30.0,  # < p25 patchcore → scène ordinaire
    "nuage_entropy_max":    3.50,  # < p50 entropy → texture homogène
    "nuage_pixdiff_max":    0.03,  # très stable → fond nuageux
}

# Features utilisées pour l'entraînement
CLASSIFIER_FEATURES = FEATURE_NAMES + ["patchcore_score"]


# ── Classification heuristique (fallback sans données labellisées) ────────

def classify_heuristic(row: dict[str, float]) -> str:
    """
    Classification par règles basées sur seuils physiques.

    Calibrée sur 848 images Merapi 2014-2018.
    Compatible avec cv2 absent (optical_flow=0, edge_density=0, contour_convexity=1).

    Ordre de priorité (du plus rare au plus courant) :
      1. pyroclastique — patchcore élevé (>p98) OU (p95 + texture chaotique)
      2. lave          — images nocturnes avec pixels brillants (incandescence)
      3. nuage         — patchcore bas + entropy faible + stable
      4. normal        — cas par défaut
    """
    import math as _math

    def _sf(val, default: float = 0.0) -> float:
        """Conversion sûre float, NaN → default."""
        try:
            f = float(val) if val is not None else default
            return default if _math.isnan(f) else f
        except (ValueError, TypeError):
            return default

    flow      = _sf(row.get("optical_flow_mag", 0.0))
    roughness = _sf(row.get("texture_roughness", 0.0))
    bright    = _sf(row.get("bright_pixel_ratio", 0.0))
    patchcore = _sf(row.get("patchcore_score"), default=0.0)
    tchange   = _sf(row.get("temporal_change_score", 0.0))
    entropy   = _sf(row.get("lbp_entropy", 0.0))
    pixdiff   = _sf(row.get("pixel_diff_mean", 0.0))

    # Correction : is_night est stocké comme chaîne "True"/"False" dans l'index
    # (load_index utilise dtype=str) → bool("False") == True en Python → BUG
    _is_night_raw = row.get("is_night", False)
    is_night = str(_is_night_raw).lower().strip() in ("true", "1", "yes", "1.0")

    # Si aucune feature physique calculée, on ne peut pas classifier → normal
    _has_features = (
        flow > 0 or roughness > 0 or tchange > 0 or entropy > 0 or pixdiff > 0
        or patchcore > 0 or bright > 0
    )
    if not _has_features:
        return "normal"

    T = _HEURISTIC_THRESHOLDS

    # Règle 1 : pyroclastique
    if flow > T["optical_flow_mag"]:  # cv2 disponible
        return "pyroclastique"
    if patchcore > T["patchcore_high"]:  # top 2% anomalie
        return "pyroclastique"
    if patchcore > T["patchcore_mid"] and roughness > T["texture_roughness"]:
        return "pyroclastique"
    if tchange > T["tchange_high"] and patchcore > 40.0:
        return "pyroclastique"

    # Règle 2 : lave (incandescence nocturne ou surexposition diurne)
    if is_night and bright > T["bright_night"]:
        return "lave"
    if not is_night and bright > T["bright_day"]:
        return "lave"

    # Règle 3 : nuage (scène homogène très ordinaire)
    # N'activer la règle nuage que si on a des features de texture (évite faux positifs)
    wl = str(row.get("weather_label", "")).lower()
    if "cloud" in wl or "nuage" in wl or "fog" in wl:
        return "nuage"
    _texture_available = entropy > 0 or pixdiff > 0
    if (_texture_available
            and patchcore < T["nuage_patchcore_max"]
            and entropy < T["nuage_entropy_max"]
            and pixdiff < T["nuage_pixdiff_max"]):
        return "nuage"

    return "normal"


# ── Classe principale ─────────────────────────────────────────────────────

class VolcanoClassifier:
    """
    Classificateur d'événements volcaniques (pyroclastique / lave / nuage / normal).

    L'entraînement supervisé (RandomForest) est utilisé quand ≥ 4 exemples
    par classe sont disponibles. En dessous, le modèle bascule automatiquement
    en mode heuristique (règles basées sur seuils).
    """

    MIN_SAMPLES_PER_CLASS = 2  # minimum pour entraîner le modèle

    def __init__(self) -> None:
        self._model: Any = None
        self._is_heuristic: bool = True
        self._classes: list[str] = VOLCANO_CLASSES
        self._feature_cols: list[str] = []

    # ── Entraînement ──────────────────────────────────────────────────────

    def fit(self, df_features: pd.DataFrame, labels: pd.Series) -> "VolcanoClassifier":
        """
        Entraîne le classificateur sur les features + labels fournis.

        Args:
            df_features: DataFrame avec colonnes FEATURE_NAMES (et patchcore_score optionnel).
            labels: Series de labels (valeurs parmi VOLCANO_CLASSES).

        Returns:
            self (pour chaînage).
        """
        feature_cols = [c for c in CLASSIFIER_FEATURES if c in df_features.columns]
        if not feature_cols:
            logger.warning("Aucune feature valide — mode heuristique activé.")
            self._is_heuristic = True
            return self

        self._feature_cols = feature_cols
        X = df_features[feature_cols].fillna(0.0).values.astype(np.float32)
        y = labels.values

        # Vérifier le nombre de samples par classe
        unique, counts = np.unique(y, return_counts=True)
        min_count = int(counts.min()) if len(counts) > 0 else 0

        if min_count < self.MIN_SAMPLES_PER_CLASS or len(unique) < 2:
            logger.warning(
                "Données insuffisantes (min %d sample(s)/classe, %d classe(s)) → "
                "mode heuristique activé.",
                min_count, len(unique),
            )
            self._is_heuristic = True
            return self

        # Tentative RandomForest
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import LabelEncoder

            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(X, y)
            self._model = rf
            self._is_heuristic = False

            # Score rapide (CV si assez de données)
            if len(X) >= 10:
                cv_folds = min(5, min_count)
                if cv_folds >= 2:
                    scores = cross_val_score(rf, X, y, cv=cv_folds, scoring="f1_macro")
                    logger.info(
                        "VolcanoClassifier (RF) — F1-macro CV: {:.3f} ± {:.3f} (n={})".format(
                            scores.mean(), scores.std(), len(X)
                        )
                    )
                else:
                    logger.info(
                        "VolcanoClassifier (RF) entraîné sur %d exemples, %d classes.",
                        len(X), len(unique),
                    )
            else:
                logger.info(
                    "VolcanoClassifier (RF) entraîné sur %d exemples (pas de CV — trop peu).",
                    len(X),
                )

        except ImportError:
            logger.warning("scikit-learn non disponible — tentative SVM.")
            self._fit_svm(X, y)

        return self

    def _fit_svm(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fallback SVM si sklearn RandomForest non disponible."""
        try:
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True)),
            ])
            pipe.fit(X, y)
            self._model = pipe
            self._is_heuristic = False
            logger.info("VolcanoClassifier (SVM fallback) entraîné sur %d exemples.", len(X))
        except Exception as e:
            logger.error("SVM fallback échoué : %s — mode heuristique.", e)
            self._is_heuristic = True

    # ── Prédiction ────────────────────────────────────────────────────────

    def predict(self, df_features: pd.DataFrame) -> np.ndarray:
        """
        Prédit la classe volcanique pour chaque ligne du DataFrame.

        Args:
            df_features: DataFrame avec colonnes FEATURE_NAMES (et patchcore_score optionnel).

        Returns:
            np.ndarray de str, shape (n,), valeurs parmi VOLCANO_CLASSES.
        """
        if self._is_heuristic or self._model is None:
            return np.array([
                classify_heuristic(row.to_dict())
                for _, row in df_features.iterrows()
            ])

        X = df_features[self._feature_cols].fillna(0.0).values.astype(np.float32)
        return self._model.predict(X)

    def predict_proba(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Retourne les probabilités de classe pour chaque image.

        Returns:
            DataFrame avec colonnes = VOLCANO_CLASSES, index aligné sur df_features.
        """
        if self._is_heuristic or self._model is None:
            # Heuristique → probabilité 1.0 pour la classe prédite
            preds = self.predict(df_features)
            result = pd.DataFrame(
                np.zeros((len(df_features), len(VOLCANO_CLASSES))),
                columns=VOLCANO_CLASSES,
                index=df_features.index,
            )
            for i, cls in enumerate(preds):
                if cls in VOLCANO_CLASSES:
                    result.iloc[i][cls] = 1.0
            return result

        X = df_features[self._feature_cols].fillna(0.0).values.astype(np.float32)
        try:
            proba = self._model.predict_proba(X)
            classes = list(self._model.classes_)
            result = pd.DataFrame(proba, columns=classes, index=df_features.index)
            # Ajouter colonnes manquantes
            for cls in VOLCANO_CLASSES:
                if cls not in result.columns:
                    result[cls] = 0.0
            return result[VOLCANO_CLASSES]
        except Exception as e:
            logger.warning("predict_proba échoué : %s — retour prédictions discrètes.", e)
            preds = self._model.predict(X)
            result = pd.DataFrame(
                np.zeros((len(df_features), len(VOLCANO_CLASSES))),
                columns=VOLCANO_CLASSES,
                index=df_features.index,
            )
            for i, cls in enumerate(preds):
                if cls in VOLCANO_CLASSES:
                    result.iloc[i][cls] = 1.0
            return result

    def feature_importances(self) -> pd.Series | None:
        """Retourne les importances des features (RandomForest uniquement)."""
        if self._is_heuristic or self._model is None:
            return None
        try:
            importances = self._model.feature_importances_
            return pd.Series(importances, index=self._feature_cols).sort_values(ascending=False)
        except AttributeError:
            return None

    # ── Persistance ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Sérialise le classificateur (pickle)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self._model,
            "is_heuristic": self._is_heuristic,
            "classes": self._classes,
            "feature_cols": self._feature_cols,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("VolcanoClassifier sauvegardé → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "VolcanoClassifier":
        """Charge un classificateur sauvegardé."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {path}")
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls()
        obj._model = state["model"]
        obj._is_heuristic = state["is_heuristic"]
        obj._classes = state.get("classes", VOLCANO_CLASSES)
        obj._feature_cols = state.get("feature_cols", [])
        logger.info("VolcanoClassifier chargé depuis %s (heuristique=%s)", path, obj._is_heuristic)
        return obj

    # ── Utilitaire batch ──────────────────────────────────────────────────

    def classify_index(
        self,
        df_index: pd.DataFrame,
        features_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """
        Classifie toutes les images d'un index DataFrame.

        Si features_path est fourni, charge le CSV de features.
        Sinon tente d'utiliser les colonnes de df_index directement.

        Args:
            df_index: DataFrame avec colonne 'filename' (et features optionnelles).
            features_path: chemin vers un CSV de features (sortie de PhysicalFeatureExtractor).

        Returns:
            DataFrame avec colonnes ['filename', 'volcano_class', 'class_confidence'].
        """
        if features_path is not None:
            df_feat = pd.read_csv(Path(features_path))
        else:
            feat_cols = [c for c in CLASSIFIER_FEATURES if c in df_index.columns]
            if not feat_cols:
                logger.warning(
                    "Aucune feature dans l'index — classification heuristique sur colonnes disponibles."
                )
            df_feat = df_index[["filename"] + [c for c in CLASSIFIER_FEATURES if c in df_index.columns]].copy()

        preds = self.predict(df_feat)
        probas = self.predict_proba(df_feat)

        result = pd.DataFrame({
            "filename": df_feat["filename"].values,
            "volcano_class": preds,
            "class_confidence": probas.max(axis=1).values,
        })
        return result
