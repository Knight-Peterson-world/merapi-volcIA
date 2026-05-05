"""
weather_classifier.py — Classifieur SVM météo vs volcanique.

Objectif :
  Triage binaire pour réduire les faux positifs liés aux conditions météo.
  Classe 0 (météo)    : quality_flag == 'cloudy'
  Classe 1 (volcanique proxy) : quality_flag == 'dark' (activité nocturne)

Modèle : SVM RBF (sklearn.svm.SVC)
  Choix justifié : ~300 exemples max → SVM surpasse les réseaux sur < 1000 exemples.
  Features : vecteur de 5 features physiques (PhysicalFeatureExtractor).

Usage :
    from src.models.weather_classifier import WeatherClassifier

    clf = WeatherClassifier()
    clf.fit(df_labeled, df_features)
    predictions = clf.predict(df_test, df_features)   # pd.Series {0, 1}
    clf.save("outputs/models/weather_svm.pkl")
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("weather_classifier")

# Étiquettes de classe
LABEL_WEATHER = 0      # nuage / météo → faux positif à filtrer
LABEL_VOLCANIC = 1     # activité volcanique (proxy : dark)


class WeatherClassifier:
    """
    Classifieur SVM météo / volcanique sur features physiques.

    Le modèle est entraîné sur les images étiquetées implicitement :
      - cloudy  → classe 0 (météo)
      - dark    → classe 1 (volcanique proxy)

    La prédiction sur nouvelles images est stockée dans index.csv
    colonne 'weather_label' (0 = météo, 1 = volcanique probable).
    """

    def __init__(self) -> None:
        self._model = None
        self._scaler = None
        self._is_fitted = False
        self._model_type: str = "svm"  # 'svm' ou 'rf'

    # ─── Feature matrix (physique + temporelle) ────────────────────────────

    @staticmethod
    def _build_feature_matrix(
        df_merged: pd.DataFrame,
        feature_names: list,
    ) -> np.ndarray:
        """
        Construit la matrice X avec features physiques + features temporelles
        cycliques (hour, month) si disponibles dans df_merged.

        Encodage cyclique sin/cos : respecte la périodicité des cycles
        journalier (24h) et annuel (12 mois), sans briser l'ordre.
        """
        X_parts = [df_merged[feature_names].fillna(0).values.astype(np.float32)]

        if "hour" in df_merged.columns:
            h = pd.to_numeric(df_merged["hour"], errors="coerce").fillna(12).values.astype(float)
            X_parts.append(np.column_stack([
                np.sin(2 * np.pi * h / 24),
                np.cos(2 * np.pi * h / 24),
            ]).astype(np.float32))

        if "month" in df_merged.columns:
            m = pd.to_numeric(df_merged["month"], errors="coerce").fillna(6).values.astype(float)
            X_parts.append(np.column_stack([
                np.sin(2 * np.pi * m / 12),
                np.cos(2 * np.pi * m / 12),
            ]).astype(np.float32))

        return np.hstack(X_parts)

    # ─── Entraînement ──────────────────────────────────────────────────────

    def fit(
        self,
        df_index: pd.DataFrame,
        df_features: pd.DataFrame,
    ) -> "WeatherClassifier":
        """
        Entraîne le SVM sur les images étiquetées implicitement.

        Args:
            df_index: DataFrame index principal (colonnes: filename, quality_flag).
            df_features: DataFrame features (colonnes: filename + FEATURE_NAMES).
                         Produit par PhysicalFeatureExtractor.compute_all().

        Returns:
            self (pour chaînage).
        """
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
        from sklearn.ensemble import RandomForestClassifier
        from src.features.physical_features import FEATURE_NAMES

        # Étiquetage implicite depuis quality_flag
        labeled = df_index[df_index["quality_flag"].isin(["cloudy", "dark"])].copy()
        labeled["label"] = labeled["quality_flag"].map(
            {"cloudy": LABEL_WEATHER, "dark": LABEL_VOLCANIC}
        )

        logger.info(
            "Dataset labellisé : %d cloudy, %d dark",
            (labeled["label"] == LABEL_WEATHER).sum(),
            (labeled["label"] == LABEL_VOLCANIC).sum(),
        )

        # ── Merge avec les features ────────────────────────────────────────
        logger.debug("labeled filenames sample : %s", labeled["filename"].head(3).tolist())
        logger.debug("features filenames sample: %s", df_features["filename"].head(3).tolist())
        logger.info(
            "Avant merge — labeled: %d, features: %d",
            len(labeled), len(df_features),
        )
        inter = len(set(labeled["filename"]) & set(df_features["filename"]))
        logger.info("Intersection clé 'filename' : %d", inter)

        merged = labeled.merge(df_features, on="filename", how="inner")
        missing = len(labeled) - len(merged)
        logger.info("Après merge : %d lignes (%d labellisés sans features)", len(merged), missing)

        # Seuil abaissé à 4 (2 par classe minimum) car le dataset Merapi
        # ne contient que 14 images labellisées (8 cloudy + 6 dark)
        if len(merged) < 4:
            raise ValueError(
                f"Pas assez d'exemples labellisés pour l'entraînement ({len(merged)} < 4). "
                f"Intersection filename index/features = {inter}. "
                "Supprimez outputs/models/physical_features.csv et relancez pour forcer le recalcul."
            )

        X = self._build_feature_matrix(merged, FEATURE_NAMES)
        y = merged["label"].values

        # Normalisation — obligatoire pour SVM RBF, neutre pour RF
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # ── Sélection du meilleur modèle par grid search (CV F1-macro) ──────
        best_C = 10.0
        best_svm_f1 = 0.0
        if len(np.unique(y)) > 1 and len(y) >= 10:
            n_splits = min(5, len(y) // max(1, int(len(y) * 0.15)))
            n_splits = max(2, n_splits)
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            # 1. Grid search SVM
            gs = GridSearchCV(
                SVC(kernel="rbf", gamma="scale", class_weight="balanced", probability=True),
                {"C": [0.1, 1.0, 10.0, 50.0]},
                cv=cv, scoring="f1_macro", n_jobs=-1,
            )
            gs.fit(X_scaled, y)
            best_C = float(gs.best_params_["C"])
            best_svm_f1 = float(gs.best_score_)
            logger.info("SVM  — best C=%.1f, CV F1-macro=%.3f", best_C, best_svm_f1)

            # 2. RandomForest (pas besoin de normalisation, mais on utilise X_scaled
            #    pour comparaison juste avec le même jeu de features)
            rf_cv = cross_val_score(
                RandomForestClassifier(
                    n_estimators=300, class_weight="balanced",
                    random_state=42, n_jobs=-1,
                ),
                X_scaled, y, cv=cv, scoring="f1_macro",
            )
            best_rf_f1 = float(rf_cv.mean())
            logger.info("RF   — CV F1-macro=%.3f ± %.3f", best_rf_f1, float(rf_cv.std()))

            # Sélectionner le meilleur (RF doit gagner par >2% pour être préféré)
            if best_rf_f1 > best_svm_f1 + 0.02:
                self._model_type = "rf"
                logger.info("→ RandomForest sélectionné (F1 RF=%.3f vs SVM=%.3f)",
                            best_rf_f1, best_svm_f1)
            else:
                self._model_type = "svm"
                logger.info("→ SVM sélectionné (F1 SVM=%.3f vs RF=%.3f)",
                            best_svm_f1, best_rf_f1)

        # ── Entraînement final sur tout le jeu labellisé ─────────────────
        if self._model_type == "rf":
            self._model = RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )
        else:
            self._model = SVC(
                kernel="rbf",
                C=best_C,
                gamma="scale",
                class_weight="balanced",
                probability=True,
            )
        self._model.fit(X_scaled, y)
        self._is_fitted = True
        logger.info("WeatherClassifier entraîné sur %d exemples", len(y))
        return self

    # ─── Prédiction ────────────────────────────────────────────────────────

    def predict(
        self,
        df_index: pd.DataFrame,
        df_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prédit la classe météo/volcanique pour toutes les images avec features.

        Args:
            df_index: DataFrame index (colonnes: filename, quality_flag).
            df_features: DataFrame features.

        Returns:
            DataFrame avec colonnes ['filename', 'weather_label', 'weather_proba_volcanic']
              - weather_label : 0 (météo) ou 1 (volcanique)
              - weather_proba_volcanic : probabilité d'être volcanique [0, 1]
        """
        from src.features.physical_features import FEATURE_NAMES

        if not self._is_fitted:
            raise RuntimeError("Le classifieur n'est pas entraîné. Appelez fit() d'abord.")

        # Prédire sur TOUTES les images ayant des features (usable + cloudy + dark)
        # pour que compute_f1_triage puisse évaluer sur les images labellisées (cloudy/dark)
        flags_to_predict = ["usable", "cloudy", "dark"]
        df_all = df_index[df_index["quality_flag"].isin(flags_to_predict)].copy()
        merged = df_all.merge(df_features[["filename"] + FEATURE_NAMES], on="filename", how="inner")

        if merged.empty:
            logger.warning("Aucune image avec features disponibles pour la prédiction.")
            return pd.DataFrame(columns=["filename", "weather_label", "weather_proba_volcanic"])

        X = self._build_feature_matrix(merged, FEATURE_NAMES)
        X_scaled = self._scaler.transform(X)
        labels = self._model.predict(X_scaled)
        probas = self._model.predict_proba(X_scaled)[:, LABEL_VOLCANIC]

        result = pd.DataFrame({
            "filename": merged["filename"].values,
            "weather_label": labels.astype(int),
            "weather_proba_volcanic": probas,
        })

        n_volcanic = (labels == LABEL_VOLCANIC).sum()
        n_weather = (labels == LABEL_WEATHER).sum()
        logger.info(
            "Prédictions : %d volcanique, %d météo sur %d images",
            n_volcanic, n_weather, len(labels),
        )
        return result

    # ─── Évaluation ────────────────────────────────────────────────────────

    def evaluate(
        self,
        df_index: pd.DataFrame,
        df_features: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Évalue le classifieur sur les images labellisées (cloudy/dark).

        Returns:
            dict avec f1_macro, f1_weather, f1_volcanic, accuracy, n_samples.
        """
        from sklearn.metrics import f1_score, accuracy_score, classification_report
        from src.features.physical_features import FEATURE_NAMES

        if not self._is_fitted:
            raise RuntimeError("Appelez fit() d'abord.")

        labeled = df_index[df_index["quality_flag"].isin(["cloudy", "dark"])].copy()
        labeled["label"] = labeled["quality_flag"].map(
            {"cloudy": LABEL_WEATHER, "dark": LABEL_VOLCANIC}
        )
        merged = labeled.merge(df_features, on="filename", how="inner")

        X = self._build_feature_matrix(merged, FEATURE_NAMES)
        X_scaled = self._scaler.transform(X)
        y_true = merged["label"].values
        y_pred = self._model.predict(X_scaled)

        report = classification_report(
            y_true, y_pred,
            target_names=["météo", "volcanique"],
            output_dict=True,
        )
        logger.info(
            "Évaluation train :\n%s",
            classification_report(y_true, y_pred, target_names=["météo", "volcanique"]),
        )
        return {
            "f1_macro": float(report["macro avg"]["f1-score"]),
            "f1_weather": float(report["météo"]["f1-score"]),
            "f1_volcanic": float(report["volcanique"]["f1-score"]),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "n_samples": len(y_true),
        }

    # ─── Persistance ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "scaler": self._scaler}, f)
        logger.info("WeatherClassifier sauvegardé → %s", path)

    def load(self, path: str | Path) -> "WeatherClassifier":
        with open(Path(path), "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        self._scaler = data["scaler"]
        self._is_fitted = True
        logger.info("WeatherClassifier chargé depuis %s", path)
        return self
