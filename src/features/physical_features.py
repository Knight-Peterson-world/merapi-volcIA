"""
physical_features.py — Extraction de features physiques discriminantes.

Features implémentées :
  1. optical_flow_mag  : magnitude du flux optique Farneback entre deux frames
                         → pyroclastique >> nuage météo (vitesse de propagation)
  2. lbp_entropy       : entropie du Local Binary Pattern
                         → pyroclastique = texture turbulente (entropie élevée)
                         → nuage = texture homogène (entropie faible)
  3. bright_pixel_ratio : proportion de pixels > threshold
                          → détecte l'incandescence nocturne (lave)
  4. contour_convexity  : rapport surface_contour / surface_enveloppe_convexe
                          → nuage ≈ 1 (convexe), pyroclastique < 0.8 (irrégulier)
  5. pixel_diff_mean   : différence absolue moyenne (baseline inter-frames)

Usage :
    from src.features.physical_features import PhysicalFeatureExtractor

    extractor = PhysicalFeatureExtractor()
    features = extractor.compute_pair(img_t0, img_t1)  # dict[str, float]
    vector = extractor.to_vector(features)              # np.ndarray (5,)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    from skimage.feature import local_binary_pattern
except ImportError:
    local_binary_pattern = None  # type: ignore

logger = logging.getLogger("physical_features")


# ─── Noms des features dans l'ordre canonique ─────────────────────────────
FEATURE_NAMES = [
    "optical_flow_mag",
    "lbp_entropy",
    "bright_pixel_ratio",
    "contour_convexity",
    "pixel_diff_mean",
    # Features volcaniques supplémentaires (v2)
    "thermal_gradient",
    "bright_zone_area",
    "texture_roughness",
    "edge_density",
    "temporal_change_score",
]


# ─── Fonctions atomiques ───────────────────────────────────────────────────

def compute_optical_flow_magnitude(img_t0: np.ndarray, img_t1: np.ndarray) -> float:
    """
    Magnitude moyenne du flux optique Farneback entre deux frames grayscale uint8.

    Discriminant clé :
      - Écoulement pyroclastique : magnitude élevée (propagation rapide)
      - Nuage météo              : magnitude faible (dérive lente)

    Returns 0.0 si cv2 n'est pas disponible ou si les images sont invalides.
    """
    if cv2 is None:
        return 0.0
    if img_t0 is None or img_t1 is None:
        return 0.0
    try:
        i0 = _to_uint8(img_t0)
        i1 = _to_uint8(img_t1)
        flow = cv2.calcOpticalFlowFarneback(
            i0, i1, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0,
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(mag.mean())
    except Exception as exc:
        logger.debug("optical_flow_magnitude failed: %s", exc)
        return 0.0


def compute_lbp_entropy(img: np.ndarray, radius: int = 3, n_points: int = 24) -> float:
    """
    Entropie de Shannon du Local Binary Pattern (uniform).

    - Pyroclastique : texture très turbulente → entropie élevée
    - Nuage météo   : texture homogène, gradients doux → entropie faible
    - Terrain rocheux normal : entropie intermédiaire

    Returns 0.0 si scikit-image n'est pas disponible.
    """
    if local_binary_pattern is None:
        return 0.0
    try:
        img_u8 = _to_uint8(img)
        lbp = local_binary_pattern(img_u8, n_points, radius, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, density=True)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist)))
    except Exception as exc:
        logger.debug("lbp_entropy failed: %s", exc)
        return 0.0


def compute_bright_pixel_ratio(img: np.ndarray, threshold: int = 200) -> float:
    """
    Proportion de pixels dont la valeur dépasse `threshold` (0–255).

    Feature principale pour détecter l'incandescence nocturne (lave active).
    Threshold 200 calibré sur images nocturnes Merapi : fond ≈ 0–50, lave ≈ 200+.

    Returns valeur dans [0, 1].
    """
    try:
        img_u8 = _to_uint8(img)
        return float((img_u8 > threshold).mean())
    except Exception as exc:
        logger.debug("bright_pixel_ratio failed: %s", exc)
        return 0.0


def compute_contour_convexity(img: np.ndarray, threshold: int = 180) -> float:
    """
    Rapport (surface du plus grand contour) / (surface de son enveloppe convexe).

    - Nuage météo       : rapport proche de 1.0 (forme convexe)
    - Pyroclastique     : rapport < 0.80 (forme irrégulière, digitée)
    - Absence de région : retourne 1.0 (conservateur)

    Requiert cv2.
    """
    if cv2 is None:
        return 1.0
    try:
        img_u8 = _to_uint8(img)
        _, binary = cv2.threshold(img_u8, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 1.0
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area < 1:
            return 1.0
        return float(min(area / hull_area, 1.0))
    except Exception as exc:
        logger.debug("contour_convexity failed: %s", exc)
        return 1.0


def compute_pixel_diff_mean(img_t0: np.ndarray, img_t1: np.ndarray) -> float:
    """Différence absolue moyenne entre deux images (baseline)."""
    try:
        a = _to_float32(img_t0)
        b = _to_float32(img_t1)
        return float(np.abs(a - b).mean())
    except Exception as exc:
        logger.debug("pixel_diff_mean failed: %s", exc)
        return 0.0


# ─── Nouvelles features volcaniques (v2) ──────────────────────────────────

def compute_thermal_gradient(img: np.ndarray) -> float:
    """
    Magnitude moyenne du gradient spatial (proxy du front thermique).

    Un gradient élevé indique des transitions abruptes claires/sombres
    caractéristiques d'un front de lave incandescent ou d'un écoulement
    pyroclastique rapide.

    Implémentation PIL+numpy (sans cv2) : gradients Sobel 3×3 manuels.
    Returns valeur >= 0.0 normalisée dans [0, 1].
    """
    try:
        img_f = _to_float32(img).astype(np.float64)
        # Gradient horizontal
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        # Gradient vertical
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        from numpy.lib.stride_tricks import as_strided
        h, w = img_f.shape
        if h < 3 or w < 3:
            return 0.0
        # Convolution manuelle (slicing décalé évite scipy)
        gx = (
            -img_f[:-2, :-2] + img_f[:-2, 2:]
            - 2 * img_f[1:-1, :-2] + 2 * img_f[1:-1, 2:]
            - img_f[2:, :-2] + img_f[2:, 2:]
        )
        gy = (
            -img_f[:-2, :-2] - 2 * img_f[:-2, 1:-1] - img_f[:-2, 2:]
            + img_f[2:, :-2] + 2 * img_f[2:, 1:-1] + img_f[2:, 2:]
        )
        mag = np.sqrt(gx ** 2 + gy ** 2)
        return float(np.mean(mag) / 1448.0)  # max théorique Sobel = 4*255*sqrt(2) ≈ 1448
    except Exception as exc:
        logger.debug("thermal_gradient failed: %s", exc)
        return 0.0


def compute_bright_zone_area(img: np.ndarray, threshold: int = 180) -> float:
    """
    Fraction de pixels dépassant `threshold` (zones potentiellement laviques).

    Différence avec bright_pixel_ratio (threshold=200) :
      - threshold=180 cible les zones de chaleur diffuse (halo autour de la lave)
      - bright_pixel_ratio (200) cible uniquement l'incandescence directe

    Returns valeur dans [0, 1].
    """
    try:
        img_u8 = _to_uint8(img)
        return float((img_u8 > threshold).mean())
    except Exception as exc:
        logger.debug("bright_zone_area failed: %s", exc)
        return 0.0


def compute_texture_roughness(img: np.ndarray) -> float:
    """
    Rugosité de texture via l'écart-type local du Laplacien.

    Le Laplacien amplifie les variations locales de haute fréquence.
    Un écart-type élevé indique une texture pyroclastique irrégulière
    (bords de blocs, turbulence) vs un nuage (texture lisse, faible std).

    Returns valeur normalisée dans [0, 1].
    """
    try:
        img_f = _to_float32(img).astype(np.float64)
        if img_f.shape[0] < 3 or img_f.shape[1] < 3:
            return 0.0
        # Laplacien 3×3
        lap = (
            img_f[:-2, 1:-1] + img_f[2:, 1:-1]
            + img_f[1:-1, :-2] + img_f[1:-1, 2:]
            - 4 * img_f[1:-1, 1:-1]
        )
        return float(np.std(lap) / 255.0)
    except Exception as exc:
        logger.debug("texture_roughness failed: %s", exc)
        return 0.0


def compute_edge_density(img: np.ndarray, low: float = 0.1, high: float = 0.3) -> float:
    """
    Proportion de pixels de bord (Canny simplifié par double-seuil sur gradient).

    Les contours du cratère, des blocs rocheux et des écoulements de lave
    génèrent une densité de bords distincte des nuages (bords flous, peu denses).

    Returns valeur dans [0, 1].
    """
    try:
        img_f = _to_float32(img).astype(np.float64)
        if img_f.shape[0] < 3 or img_f.shape[1] < 3:
            return 0.0
        # Gradient magnitude normalisé
        gx = img_f[1:-1, 2:] - img_f[1:-1, :-2]
        gy = img_f[2:, 1:-1] - img_f[:-2, 1:-1]
        mag = np.sqrt(gx ** 2 + gy ** 2) / (2 * 255.0)
        # Double seuil
        edges = mag > low
        strong_edges = mag > high
        # Propagation simple : pixels faibles adjacents à un fort → bord
        from numpy.lib.stride_tricks import as_strided
        dilated = strong_edges.copy()
        dilated[1:] |= strong_edges[:-1]
        dilated[:-1] |= strong_edges[1:]
        dilated[:, 1:] |= strong_edges[:, :-1]
        dilated[:, :-1] |= strong_edges[:, 1:]
        final_edges = edges & dilated
        return float(final_edges.mean())
    except Exception as exc:
        logger.debug("edge_density failed: %s", exc)
        return 0.0


def compute_temporal_change_score(img_t0: np.ndarray, img_t1: np.ndarray) -> float:
    """
    Score de changement temporel normalisé entre deux frames.

    Amélioration de pixel_diff_mean :
      - normalisation par la variance de fond (robustesse aux variations d'éclairage)
      - retourne (diff_mean) / (1 + std(img_t0)) pour compenser les images très sombres

    Returns valeur >= 0.0 (typiquement < 1.0 pour des scènes normales).
    """
    try:
        a = _to_float32(img_t0)
        b = _to_float32(img_t1)
        diff_mean = float(np.abs(a - b).mean())
        std_bg = float(np.std(a)) + 1e-6  # évite division par zéro
        return diff_mean / (1.0 + std_bg)
    except Exception as exc:
        logger.debug("temporal_change_score failed: %s", exc)
        return 0.0


# ─── Feature vector complet (paire d'images) ──────────────────────────────

def extract_discriminative_features(
    img_t0: np.ndarray,
    img_t1: np.ndarray,
) -> dict[str, float]:
    """
    Calcule les 5 features discriminantes à partir d'une paire d'images consécutives.

    Args:
        img_t0: image au temps t0 (grayscale uint8 ou float32 normalisé)
        img_t1: image au temps t1 (même caméra, |t1-t0| < 20 min)

    Returns:
        dict avec les clés FEATURE_NAMES et leurs valeurs float.
    """
    return {
        "optical_flow_mag":       compute_optical_flow_magnitude(img_t0, img_t1),
        "lbp_entropy":            compute_lbp_entropy(img_t1),
        "bright_pixel_ratio":     compute_bright_pixel_ratio(img_t1),
        "contour_convexity":      compute_contour_convexity(img_t1),
        "pixel_diff_mean":        compute_pixel_diff_mean(img_t0, img_t1),
        # Features v2
        "thermal_gradient":       compute_thermal_gradient(img_t1),
        "bright_zone_area":       compute_bright_zone_area(img_t1),
        "texture_roughness":      compute_texture_roughness(img_t1),
        "edge_density":           compute_edge_density(img_t1),
        "temporal_change_score":  compute_temporal_change_score(img_t0, img_t1),
    }


# ─── Classe principale ─────────────────────────────────────────────────────

class PhysicalFeatureExtractor:
    """
    Extrait et stocke les features physiques pour tout le dataset.

    Usage complet :
        extractor = PhysicalFeatureExtractor()
        df_features = extractor.compute_all(df_index)
        extractor.save(df_features, "outputs/models/physical_features.csv")
    """

    def __init__(
        self,
        max_gap_minutes: int = 20,
        bright_threshold: int = 200,
    ) -> None:
        """
        Args:
            max_gap_minutes: seuil de gap temporel max entre deux frames
                             pour que la paire soit valide pour le flux optique.
            bright_threshold: seuil pixel pour compute_bright_pixel_ratio.
        """
        self.max_gap_minutes = max_gap_minutes
        self.bright_threshold = bright_threshold

    def compute_single(self, img: np.ndarray, img_prev: np.ndarray | None = None) -> dict[str, float]:
        """
        Calcule les features pour une image unique (img_prev optionnel pour flux optique).
        Si img_prev est None, optical_flow_mag et pixel_diff_mean valent 0.0.
        """
        if img_prev is not None:
            return extract_discriminative_features(img_prev, img)
        return {
            "optical_flow_mag":       0.0,
            "lbp_entropy":            compute_lbp_entropy(img),
            "bright_pixel_ratio":     compute_bright_pixel_ratio(img, self.bright_threshold),
            "contour_convexity":      compute_contour_convexity(img),
            "pixel_diff_mean":        0.0,
            # Features v2 (single image, pas de temporel)
            "thermal_gradient":       compute_thermal_gradient(img),
            "bright_zone_area":       compute_bright_zone_area(img),
            "texture_roughness":      compute_texture_roughness(img),
            "edge_density":           compute_edge_density(img),
            "temporal_change_score":  0.0,
        }

    def to_vector(self, features: dict[str, float]) -> np.ndarray:
        """Convertit un dict de features en vecteur numpy ordonné selon FEATURE_NAMES."""
        return np.array([features.get(k, 0.0) for k in FEATURE_NAMES], dtype=np.float32)

    def compute_all(self, df: pd.DataFrame, image_root: Path | str | None = None) -> pd.DataFrame:
        """
        Calcule les features physiques pour toutes les images 'usable' du dataset.

        Règle de pairing :
          - On trie par (caméra, datetime)
          - Pour chaque image, on cherche l'image précédente de la même caméra
            dans une fenêtre de max_gap_minutes minutes
          - Si pas de voisin → features inter-frames à 0.0

        Args:
            df: DataFrame index complet (colonnes: filename, year, month, day,
                hour, minute, quality_flag, local_path)
            image_root: racine du projet pour résoudre les chemins relatifs.

        Returns:
            DataFrame avec colonnes FEATURE_NAMES + ['filename'].
        """
        from src.utils import PROJECT_ROOT

        if image_root is None:
            image_root = PROJECT_ROOT

        image_root = Path(image_root)

        # Inclure usable + cloudy + dark + NaN (images non encore classifiées)
        # NaN = image téléchargée mais pas encore passée par QualityFilter → traiter comme usable
        valid_flags = ["usable", "cloudy", "dark"]
        flag_mask = df["quality_flag"].isin(valid_flags) | df["quality_flag"].isna()
        df_usable = df[flag_mask].copy()
        # Normaliser NaN → "usable" pour le logging uniquement
        qf_display = df_usable["quality_flag"].fillna("usable")
        counts = {f: int((qf_display == f).sum()) for f in valid_flags}
        logger.info(
            "compute_all : %d images (%s)",
            len(df_usable),
            ", ".join(f"{f}={counts[f]}" for f in valid_flags),
        )
        df_usable = self._add_datetime(df_usable)
        df_usable = df_usable.sort_values("datetime").reset_index(drop=True)

        # Extraire le nom de caméra depuis le filename
        df_usable["camera"] = df_usable["filename"].str.extract(
            r"^([A-Za-z0-9_]+)_Canon", expand=False
        ).str.lower().fillna("unknown")

        records = []
        for cam, group in df_usable.groupby("camera"):
            group = group.sort_values("datetime").reset_index(drop=True)
            for i, row in group.iterrows():
                img = self._load_image(row, image_root)
                if img is None:
                    continue

                img_prev = None
                if i > 0:
                    prev_row = group.iloc[i - 1]
                    gap = (row["datetime"] - prev_row["datetime"]).total_seconds() / 60
                    if gap <= self.max_gap_minutes:
                        img_prev = self._load_image(prev_row, image_root)

                feats = self.compute_single(img, img_prev)
                feats["filename"] = row["filename"]
                records.append(feats)

        if not records:
            return pd.DataFrame(columns=["filename"] + FEATURE_NAMES)

        return pd.DataFrame(records)[["filename"] + FEATURE_NAMES]

    @staticmethod
    def save(df_features: pd.DataFrame, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_csv(path, index=False)
        logger.info("Features saved → %s (%d rows)", path, len(df_features))

    @staticmethod
    def load(path: str | Path) -> pd.DataFrame:
        return pd.read_csv(Path(path))

    # ── helpers privés ────────────────────────────────────────────────────

    @staticmethod
    def _add_datetime(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n_before = len(df)

        # 1. Conversion numérique robuste
        for col in ["year", "month", "day", "hour", "minute", "second"]:
            df[col] = pd.to_numeric(df.get(col, pd.Series(0, index=df.index)), errors="coerce")

        # 2. Supprimer les lignes avec year ou month invalides (non récupérables)
        invalid_base = (
            df["year"].isna() |
            df["month"].isna() | (df["month"] <= 0) | (df["month"] > 12)
        )
        n_invalid_base = int(invalid_base.sum())
        if n_invalid_base > 0:
            logger.warning(
                "_add_datetime : %d ligne(s) supprimées (year/month invalide).",
                n_invalid_base,
            )
            df = df[~invalid_base].copy()

        # 3. Imputation des jours invalides (day=0, day=NaN, day>31) → day=1
        #    Justification : la date d'acquisition est connue au mois près ;
        #    imputer J1 est conservateur et préserve l'information mensuelle
        #    sans perdre de données (186 lignes récupérées vs supprimées).
        bad_day = df["day"].isna() | (df["day"] <= 0) | (df["day"] > 31)
        n_bad_day = int(bad_day.sum())
        if n_bad_day > 0:
            years_affected = sorted(
                df.loc[bad_day, "year"].dropna().astype(int).unique().tolist()
            )
            df.loc[bad_day, "day"] = 1
            logger.warning(
                "_add_datetime : %d jour(s) invalide(s) imputés à J1 (premier du mois). "
                "Années concernées : %s",
                n_bad_day, years_affected,
            )

        # 4. Remplir les composantes horaires manquantes par 0
        for col in ["hour", "minute", "second"]:
            df[col] = df[col].fillna(0)

        # 5. Caster en int
        for col in ["year", "month", "day", "hour", "minute", "second"]:
            df[col] = df[col].astype(int)

        # 6. Conversion datetime avec errors="coerce" → NaT sur les cas résiduels
        df["datetime"] = pd.to_datetime(
            df[["year", "month", "day", "hour", "minute", "second"]], errors="coerce"
        )

        n_nat = int(df["datetime"].isna().sum())
        if n_nat > 0:
            logger.warning("_add_datetime : %d date(s) résiduelles converties en NaT (conservées).", n_nat)

        logger.debug(
            "_add_datetime : %d lignes en entrée → %d après nettoyage "
            "(%d supprimées, %d jours imputés, %d NaT).",
            n_before, len(df), n_invalid_base, n_bad_day, n_nat,
        )
        return df

    @staticmethod
    def _load_image(row: pd.Series, root: Path) -> np.ndarray | None:
        """Charge une image (raw ou processed) et la retourne en grayscale uint8."""
        import re
        paths_to_try = []

        lp = str(row.get("local_path", ""))

        # 1. local_path absolu
        if lp and Path(lp).exists():
            paths_to_try.append(Path(lp))

        # 2. Chemin relatif depuis la racine
        if lp and not Path(lp).is_absolute():
            paths_to_try.append(root / lp)

        # 3. Fallback processed (raw/ → processed/, extension → .png)
        #    Indispensable quand data/raw/ est vide mais data/processed/ est complet
        if lp:
            proc_lp = re.sub(r'/raw/', '/processed/', lp)
            proc_lp = re.sub(r'\.(jpg|JPG|jpeg|JPEG)$', '.png', proc_lp)
            proc_path = Path(proc_lp) if Path(proc_lp).is_absolute() else root / proc_lp
            if proc_path.exists() and proc_path not in paths_to_try:
                paths_to_try.append(proc_path)

        # 4. Fallback par filename dans processed/
        filename = str(row.get("filename", ""))
        year = row.get("year")
        month = row.get("month")
        if filename and year and month:
            stem = re.sub(r'\.(jpg|JPG|jpeg|JPEG)$', '', filename)
            proc_by_fn = root / "data" / "processed" / str(int(year)) / f"{int(month):02d}" / f"{stem}.png"
            if proc_by_fn.exists() and proc_by_fn not in paths_to_try:
                paths_to_try.append(proc_by_fn)

        for p in paths_to_try:
            try:
                img = np.array(Image.open(p).convert("L"))
                return img
            except Exception:
                continue
        return None


# ─── Helpers internes ──────────────────────────────────────────────────────

def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Convertit une image float32 [0,1] ou uint8 en uint8 [0,255]."""
    if img.dtype == np.uint8:
        return img
    if img.max() <= 1.0:
        return (img * 255).astype(np.uint8)
    return img.astype(np.uint8)


def _to_float32(img: np.ndarray) -> np.ndarray:
    """Convertit une image uint8 en float32 [0,1]."""
    if img.dtype == np.float32 and img.max() <= 1.0:
        return img
    return img.astype(np.float32) / 255.0
