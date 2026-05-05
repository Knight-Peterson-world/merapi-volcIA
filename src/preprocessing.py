"""
preprocessing.py — Pipeline de prétraitement des images Merapi.

Responsabilités :
- Découverte dynamique des dossiers data/raw/{year}/{month}/ disponibles
- Redimensionnement à la résolution cible (256×256)
- Normalisation des intensités (minmax ou zscore)
- Séparation jour / nuit par seuil de luminosité
- Recadrage sur la zone d'intérêt volcanique (ROI)
- Sauvegarde des images prétraitées au format PNG (lossless) dans data/processed/
- Intégration du quality_filter avec warning si aucune image "usable"

Choix techniques justifiés :
    - Format PNG (sans perte) : préserve l'intégrité des données scientifiques
    - Redimensionnement en LANCZOS (qualité maximale, cohérence)
    - Normalisation minmax [0,1] par défaut : préserve les contrastes
      relatifs et est sans ambiguïté pour des images de surveillance
    - Pas de fichiers .npy : chargement PNG → float32 en mémoire à la volée
    - La séparation jour/nuit est un prérequis essentiel car les deux
      classes ont des distributions de pixels totalement différentes
      et doivent être modélisées séparément (autoencodeur Phase 5)

Usage :
    # Traiter toutes les données disponibles
    python -m src.preprocessing

    # Traiter seulement une année
    python -m src.preprocessing --year 2019

    # Traiter un mois précis
    python -m src.preprocessing --year 2019 --month 5

    # Depuis Python
    from src.preprocessing import MerapiPreprocessor
    from src.utils import load_config

    config = load_config()
    preprocessor = MerapiPreprocessor(config)
    results = preprocessor.process_all_available()
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging as _logging
    import sys as _sys

    class _FallbackLogger:
        def __init__(self):
            self._l = _logging.getLogger("preprocessing")
            self._l.setLevel(_logging.INFO)
            if not self._l.handlers:
                _h = _logging.StreamHandler(_sys.stderr)
                _h.setFormatter(_logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
                self._l.addHandler(_h)

        def info(self, m, *a, **k): self._l.info(m)
        def warning(self, m, *a, **k): self._l.warning(m)
        def debug(self, m, *a, **k): self._l.debug(m)
        def error(self, m, *a, **k): self._l.error(m)
        def exception(self, m, *a, **k): self._l.exception(m)

    logger = _FallbackLogger()

from PIL import Image

from src.utils import (
    PROJECT_ROOT,
    get_raw_image_dir,
    get_processed_image_dir,
    load_config,
    setup_logger,
    safe_sum,
    safe_mean,
)


# ============================================================
# Préprocesseur principal
# ============================================================

class MerapiPreprocessor:
    """
    Pipeline de prétraitement des images brutes Merapi.

    Chaque image brute → image prétraitée dans data/processed/.
    Le prétraitement est reproductible et idempotent.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.pp_cfg = config["preprocessing"]

        self.target_size = tuple(self.pp_cfg["target_size"])  # ex. (256, 256)
        self.normalization = self.pp_cfg.get("normalization", "minmax")
        self.night_thresh = self.pp_cfg.get("night_brightness_threshold", 30)
        self.cloud_var_thresh = self.pp_cfg.get("cloud_variance_threshold", 50.0)

        # ROI — désactivé par défaut jusqu'à calibration sur données réelles
        self.roi_enabled = self.pp_cfg.get("roi", {}).get("enabled", False)
        self.roi = self._parse_roi()

    # ----------------------------------------------------------
    # Découverte dynamique des données disponibles
    # ----------------------------------------------------------

    def discover_available_months(
        self,
        year: int | None = None,
        month: int | None = None,
    ) -> list[tuple[int, int]]:
        """
        Parcourt data/raw/ et retourne la liste des (year, month) qui contiennent
        au moins une image téléchargée.

        Ne hardcode aucune année : découverte entièrement dynamique.

        Args:
            year:  si fourni, restreint la découverte à cette année.
            month: si fourni (nécessite year), restreint à ce mois précis.

        Returns:
            Liste triée de (year, month).
        """
        raw_base = PROJECT_ROOT / self.config["paths"]["data_raw"]
        extensions = set(self.config["source"]["image_extensions"])
        found: list[tuple[int, int]] = []

        if not raw_base.exists():
            logger.warning(f"Dossier data/raw/ introuvable : {raw_base}")
            return []

        # Si year ET month fournis → vérifier juste ce dossier
        if year is not None and month is not None:
            candidate = raw_base / str(year) / f"{month:02d}"
            if candidate.is_dir() and any(
                p.suffix in extensions for p in candidate.iterdir()
            ):
                return [(year, month)]
            return []

        # Parcourir year_dirs (ou uniquement l'année demandée)
        if year is not None:
            year_dirs = [raw_base / str(year)]
        else:
            year_dirs = sorted(raw_base.iterdir())

        for year_dir in year_dirs:
            if not year_dir.is_dir():
                continue
            try:
                y = int(year_dir.name)
            except ValueError:
                continue  # dossier non numérique, on ignore

            month_dirs = sorted(year_dir.iterdir())
            for month_dir in month_dirs:
                if not month_dir.is_dir():
                    continue
                try:
                    m = int(month_dir.name)
                except ValueError:
                    continue
                # Vérifier qu'il y a au moins une image
                has_images = any(
                    p.suffix in extensions for p in month_dir.iterdir()
                )
                if has_images:
                    found.append((y, m))

        logger.info(f"Données disponibles : {len(found)} mois trouvés dans data/raw/")
        return found

    # ----------------------------------------------------------
    # Traitement d'une image
    # ----------------------------------------------------------

    def process_image(
        self,
        raw_path: Path,
        processed_path: Path,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """
        Prétraite une image brute et la sauvegarde dans processed_path.

        Pipeline :
            1. Chargement et vérification
            2. Recadrage ROI (si activé)
            3. Conversion en niveaux de gris
            4. Redimensionnement (LANCZOS)
            5. Normalisation
            6. Sauvegarde en PNG (lossless)

        Args:
            raw_path:       chemin de l'image brute.
            processed_path: chemin de sortie.
            overwrite:      si False et que le fichier existe → skip.

        Returns:
            dict avec les métadonnées du traitement.
        """
        result: dict[str, Any] = {
            "raw_path": str(raw_path),
            "processed_path": str(processed_path),
            "success": False,
            "mean_brightness": None,
            "std_brightness": None,
            "variance": None,
            "is_night": False,       # toujours bool (jamais None)
            "quality_flag": "unknown",  # enrichi par quality_filter si activé
            "original_size": None,
            "error": None,
        }

        # Skip si déjà traité — on charge quand même le PNG pour calculer les stats
        png_out = processed_path.with_suffix(".png")
        if not overwrite and png_out.exists():
            logger.debug(f"Skip (déjà traité) : {raw_path.name}")
            result["processed_path"] = str(png_out)
            result["success"] = True
            try:
                arr_skip = np.array(Image.open(png_out).convert("L"), dtype=np.float32)
                mean_b_s = float(arr_skip.mean())
                var_b_s  = float(arr_skip.var())
                result["mean_brightness"] = mean_b_s
                result["std_brightness"]  = float(arr_skip.std())
                result["variance"]        = var_b_s
                result["is_night"]        = bool(mean_b_s < self.night_thresh)
                result["quality_flag"]    = self._classify_stats(mean_b_s, var_b_s)
            except Exception as exc:
                logger.debug(f"Stats non calculées pour image skippée {png_out.name}: {exc}")
            return result

        if not raw_path.exists():
            result["error"] = f"Fichier source introuvable : {raw_path}"
            logger.warning(result["error"])
            return result

        try:
            # 1. Chargement — PIL gère JPG/JPEG/PNG/etc.
            img = Image.open(raw_path)
            img.verify()                  # détecte les fichiers tronqués/corrompus
            img = Image.open(raw_path)    # réouverture après verify()
            result["original_size"] = img.size

            # 2. Recadrage ROI (si configuré)
            if self.roi_enabled and self.roi:
                img = self._apply_roi(img)

            # 3. Niveaux de gris
            img_gray = img.convert("L")

            # 4. Redimensionnement
            img_resized = img_gray.resize(self.target_size, Image.LANCZOS)

            # 5. Statistiques
            arr = np.array(img_resized, dtype=np.float32)
            mean_b = float(arr.mean())
            std_b  = float(arr.std())
            var_b  = float(arr.var())

            result["mean_brightness"] = mean_b
            result["std_brightness"]  = std_b
            result["variance"]        = var_b
            result["is_night"]        = bool(mean_b < self.night_thresh)
            result["quality_flag"]    = self._classify_stats(mean_b, var_b)

            # 6. Normalisation
            arr_norm = self._normalize(arr)

            # 7. Sauvegarde PNG (lossless)
            png_out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((arr_norm * 255).astype(np.uint8)).save(str(png_out), format="PNG")

            result["processed_path"] = str(png_out)
            result["success"] = True
            logger.debug(
                f"✓ {raw_path.name} → {self.target_size} "
                f"| lum={mean_b:.1f} | nuit={result['is_night']}"
            )

        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            logger.error(f"Erreur prétraitement {raw_path} : {exc}")

        return result

    # ----------------------------------------------------------
    # Traitement par mois
    # ----------------------------------------------------------

    def process_month(
        self,
        year: int,
        month: int,
        overwrite: bool = False,
        max_images: int | None = None,
        with_quality: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Prétraite toutes les images brutes d'un mois.

        Args:
            year:         année.
            month:        mois (1–12).
            overwrite:    re-traiter même si l'image existe déjà.
            max_images:   plafond d'images à traiter (None = toutes).
            with_quality: si True, intègre le quality_filter après prétraitement.

        Returns:
            Liste de dicts résultats (un par image).
        """
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(it, **_):  # type: ignore[misc]
                return it

        raw_dir  = get_raw_image_dir(self.config, year, month)
        proc_dir = get_processed_image_dir(self.config, year, month)

        extensions = set(self.config["source"]["image_extensions"])
        raw_images = sorted(p for p in raw_dir.glob("*") if p.suffix in extensions)

        if not raw_images:
            logger.warning(f"Aucune image brute dans {raw_dir}")
            return []

        if max_images is not None and len(raw_images) > max_images:
            logger.info(f"Plafond appliqué : {max_images}/{len(raw_images)} images")
            raw_images = raw_images[:max_images]

        print(f"\nTraitement {year}/{month:02d} — {len(raw_images)} images")

        results: list[dict[str, Any]] = []
        for raw_path in tqdm(raw_images, desc=f"  {year}/{month:02d}", unit="img", leave=False):
            proc_path = proc_dir / raw_path.name
            res = self.process_image(raw_path, proc_path, overwrite=overwrite)
            results.append(res)

        n_ok   = sum(r["success"] for r in results)
        n_fail = len(results) - n_ok
        print(f"  ✔ {n_ok} OK | ❌ {n_fail} erreurs")
        if n_fail:
            for r in results:
                if not r["success"]:
                    logger.error(f"  → {r['raw_path']}: {r['error']}")

        # Intégration quality_filter (optionnel)
        if with_quality and n_ok > 0:
            results = self._run_quality_filter(results, year, month)

        return results

    # ----------------------------------------------------------
    # Traitement de toutes les données disponibles
    # ----------------------------------------------------------

    def process_all_available(
        self,
        year: int | None = None,
        month: int | None = None,
        overwrite: bool = False,
        max_per_month: int | None = None,
        with_quality: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Parcourt dynamiquement data/raw/ et prétraite tous les mois disponibles.

        N'hardcode aucune année — découverte automatique.

        Args:
            year:          si fourni, restreint à cette année.
            month:         si fourni (nécessite year), restreint à ce mois.
            overwrite:     re-traiter même si déjà présent.
            max_per_month: plafond d'images par mois (None = toutes).
            with_quality:  activer le quality_filter après chaque mois.

        Returns:
            Dict {"YYYY-MM": [résultats]}.
        """
        available = self.discover_available_months(year=year, month=month)

        if not available:
            logger.warning("Aucun mois disponible à traiter dans data/raw/")
            return {}

        total_months = len(available)
        print(f"\n{'='*55}")
        print(f"Pipeline preprocessing — {total_months} mois à traiter")
        if year:
            print(f"  Filtre : année {year}" + (f", mois {month:02d}" if month else ""))
        print(f"  Sortie : data/processed/  |  Taille cible : {self.target_size}")
        print(f"{'='*55}")

        all_results: dict[str, list[dict[str, Any]]] = {}
        total_ok = total_fail = 0

        for i, (y, m) in enumerate(available, 1):
            key = f"{y}-{m:02d}"
            print(f"\n[{i}/{total_months}] ", end="")
            results = self.process_month(
                y, m,
                overwrite=overwrite,
                max_images=max_per_month,
                with_quality=with_quality,
            )
            if results:
                all_results[key] = results
                total_ok   += sum(r["success"] for r in results)
                total_fail += sum(not r["success"] for r in results)

        print(f"\n{'='*55}")
        print(f"Pipeline terminé : {total_ok} images OK | {total_fail} erreurs")
        print(f"Périodes traitées : {len(all_results)}/{total_months}")
        print(f"{'='*55}\n")

        return all_results

    # ----------------------------------------------------------
    # Méthodes historiques conservées (compatibilité)
    # ----------------------------------------------------------

    def process_from_index(
        self,
        df,
        overwrite: bool = False,
        max_images: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Prétraite les images listées dans un DataFrame d'index.

        Utilise les colonnes year/month/filename pour localiser les fichiers.

        Args:
            df:         DataFrame d'index.
            overwrite:  re-traiter si déjà présent.
            max_images: plafond d'images (None = toutes).

        Returns:
            Liste de dicts résultats.
        """
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(it, **_):  # type: ignore[misc]
                return it

        raw_base  = PROJECT_ROOT / self.config["paths"]["data_raw"]
        proc_base = PROJECT_ROOT / self.config["paths"]["data_processed"]

        rows_to_process: list[tuple[Path, Path, str]] = []
        for _, row in df.iterrows():
            try:
                y, m = int(row["year"]), int(row["month"])
            except (ValueError, TypeError):
                continue
            raw_path = raw_base / str(y) / f"{m:02d}" / row["filename"]
            if raw_path.exists():
                proc_path = proc_base / str(y) / f"{m:02d}" / row["filename"]
                rows_to_process.append((raw_path, proc_path, row.get("url", "")))

        if not rows_to_process:
            logger.warning("Aucune image trouvée sur disque pour l'index fourni.")
            return []

        if max_images is not None and len(rows_to_process) > max_images:
            rows_to_process = rows_to_process[:max_images]

        logger.info(f"Prétraitement de {len(rows_to_process)} images depuis l'index...")

        results: list[dict[str, Any]] = []
        for raw_path, proc_path, url in tqdm(rows_to_process, desc="Prétraitement index"):
            res = self.process_image(raw_path, proc_path, overwrite=overwrite)
            res["url"] = url
            results.append(res)

        n_ok   = sum(r["success"] for r in results)
        n_fail = len(results) - n_ok
        logger.info(f"Prétraitement terminé : {n_ok} OK, {n_fail} erreurs")
        return results

    def process_all_months(
        self,
        year_start: int = 2014,
        year_end: int = 2018,
        overwrite: bool = False,
        max_per_month: int | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Prétraite tous les mois d'une plage d'années (méthode historique).
        Préférez process_all_available() qui est dynamique.
        """
        all_results: dict[str, list[dict[str, Any]]] = {}
        for year in range(year_start, year_end + 1):
            for month in range(1, 13):
                key = f"{year}-{month:02d}"
                results = self.process_month(
                    year, month, overwrite=overwrite, max_images=max_per_month
                )
                if results:
                    all_results[key] = results
        return all_results

    # ----------------------------------------------------------
    # Quality filter intégré
    # ----------------------------------------------------------

    def _run_quality_filter(
        self,
        results: list[dict[str, Any]],
        year: int,
        month: int,
    ) -> list[dict[str, Any]]:
        """
        Appelle QualityFilter sur les images traitées avec succès.
        Enrichit chaque résultat du champ quality_flag.
        Affiche un warning si aucune image n'est "usable".
        """
        try:
            from src.quality_filter import QualityFilter, QUALITY_USABLE
        except ImportError as exc:
            logger.warning(f"quality_filter non disponible : {exc}")
            return results

        processed_paths = [
            Path(r["processed_path"])
            for r in results
            if r["success"] and Path(r["processed_path"]).exists()
        ]
        if not processed_paths:
            return results

        qf = QualityFilter(self.config)
        quality_map: dict[str, str] = {}

        for pth in processed_paths:
            try:
                qr = qf.classify_image(pth)
                quality_map[pth.name] = qr.get("quality_flag", "unknown")
            except Exception as exc:
                logger.error(f"Quality filter erreur sur {pth}: {exc}")
                quality_map[pth.name] = "unknown"

        # Enrichir les résultats
        for r in results:
            pp = Path(r["processed_path"])
            r["quality_flag"] = quality_map.get(pp.name, "unknown")

        # Résumé qualité
        from collections import Counter
        counts = Counter(r.get("quality_flag", "unknown") for r in results if r["success"])
        n_usable = counts.get(QUALITY_USABLE, 0)
        logger.info(
            f"Qualité {year}/{month:02d} : "
            + " | ".join(f"{v}={c}" for v, c in sorted(counts.items()))
        )
        if n_usable == 0:
            logger.warning(
                f"⚠ Aucune image 'usable' pour {year}/{month:02d} "
                f"(seuils : lum>{self.night_thresh}, var>{self.config['preprocessing']['cloud_variance_threshold']}). "
                "Vérifiez les seuils dans config/settings.yaml."
            )
        return results

    # ----------------------------------------------------------
    # Utilitaires internes
    # ----------------------------------------------------------

    def _classify_stats(self, mean_b: float, var_b: float) -> str:
        """
        Classifie une image à partir de ses statistiques de pixels.

        Ordre de priorité :
            1. dark    → luminosité < night_thresh  (image nocturne)
            2. cloudy  → variance < cloud_var_thresh (image uniforme = nuages)
            3. usable  → reste

        Args:
            mean_b: luminosité moyenne (0–255).
            var_b:  variance des pixels.

        Returns:
            str parmi : 'dark', 'cloudy', 'usable'.
        """
        if mean_b < self.night_thresh:
            return "dark"
        if var_b < self.cloud_var_thresh:
            return "cloudy"
        return "usable"

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """
        Normalise un tableau float32.

        - minmax : redimensionne [min, max] → [0, 1]
        - zscore : centre-réduit (mean=0, std=1)
        """
        if self.normalization == "minmax":
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max - arr_min < 1e-6:
                return np.zeros_like(arr, dtype=np.float32)
            return (arr - arr_min) / (arr_max - arr_min)

        if self.normalization == "zscore":
            mean, std = arr.mean(), arr.std()
            if std < 1e-6:
                return np.zeros_like(arr, dtype=np.float32)
            return (arr - mean) / std

        raise ValueError(f"Méthode de normalisation inconnue : {self.normalization}")

    def _apply_roi(self, img: Image.Image) -> Image.Image:
        """Recadre l'image sur la zone d'intérêt volcanique (ROI)."""
        if self.roi is None:
            return img
        x_min, y_min, x_max, y_max = self.roi
        return img.crop((x_min, y_min, x_max, y_max))

    def _parse_roi(self) -> tuple[int, int, int, int] | None:
        """Parse la ROI depuis la configuration."""
        roi_cfg = self.pp_cfg.get("roi", {})
        if not roi_cfg.get("enabled", False):
            return None
        coords = [roi_cfg.get(k) for k in ["x_min", "y_min", "x_max", "y_max"]]
        if any(c is None for c in coords):
            logger.warning(
                "ROI activée dans la config mais coordonnées non définies. "
                "Définissez roi.x_min/y_min/x_max/y_max dans settings.yaml."
            )
            return None
        return tuple(int(c) for c in coords)  # type: ignore[return-value]

    # ----------------------------------------------------------
    # Utilitaires de chargement (pour les modèles)
    # ----------------------------------------------------------

    @staticmethod
    def load_processed_image(processed_path: Path) -> np.ndarray | None:
        """
        Charge une image prétraitée depuis son fichier PNG.

        Recherche d'abord .png, puis .npy (rétro-compatibilité),
        puis .jpg en dernier recours.

        Returns:
            Array float32 normalisé [0,1] ou None si introuvable.
        """
        # 1. PNG (format cible)
        png_path = processed_path.with_suffix(".png")
        if png_path.exists():
            try:
                return np.array(Image.open(png_path).convert("L"), dtype=np.float32) / 255.0
            except Exception as exc:
                logger.error(f"Erreur chargement PNG {png_path}: {exc}")

        # 2. .npy (anciens fichiers)
        npy_path = processed_path.with_suffix(".npy")
        if npy_path.exists():
            logger.debug(f"Fallback .npy (ancien format) : {npy_path}")
            return np.load(str(npy_path)).astype(np.float32)

        # 3. .jpg
        jpg_path = processed_path.with_suffix(".jpg")
        if jpg_path.exists():
            try:
                return np.array(Image.open(jpg_path).convert("L"), dtype=np.float32) / 255.0
            except Exception as exc:
                logger.error(f"Erreur chargement JPG {jpg_path}: {exc}")

        logger.debug(f"Image prétraitée introuvable : {processed_path}")
        return None

    @staticmethod
    def load_processed_batch(paths: list[Path]) -> np.ndarray:
        """
        Charge un batch d'images prétraitées en un tableau numpy.

        Returns:
            Array shape (N, H, W) float32. Les images non trouvées sont ignorées.
        """
        arrays = [
            arr for p in paths
            if (arr := MerapiPreprocessor.load_processed_image(p)) is not None
        ]
        return np.stack(arrays, axis=0) if arrays else np.array([])


# ============================================================
# Point d'entrée CLI
# ============================================================

def _build_parser():
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m src.preprocessing",
        description=(
            "Prétraitement des images Merapi.\n"
            "Sans arguments → traite TOUTES les données disponibles dans data/raw/.\n"
            "Avec --year seul → traite toute l'année.\n"
            "Avec --year ET --month → traite un seul mois."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help="Année à traiter (ex: 2019). Défaut : toutes les années disponibles.",
    )
    parser.add_argument(
        "--month", type=int, default=None,
        help="Mois à traiter (1-12). Nécessite --year. Défaut : tous les mois.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Forcer le re-traitement des images déjà présentes dans data/processed/.",
    )
    parser.add_argument(
        "--max-per-month", type=int, default=None, metavar="N",
        help="Nombre maximum d'images à traiter par mois (pour les tests).",
    )
    parser.add_argument(
        "--with-quality", action="store_true",
        help="Activer le quality_filter après chaque mois (plus lent).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Validation : --month sans --year
    if args.month is not None and args.year is None:
        parser.error("--month nécessite --year (ex: --year 2019 --month 5)")

    cfg = load_config()
    setup_logger(cfg)

    preprocessor = MerapiPreprocessor(cfg)

    all_results = preprocessor.process_all_available(
        year=args.year,
        month=args.month,
        overwrite=args.overwrite,
        max_per_month=args.max_per_month,
        with_quality=args.with_quality,
    )

    # Récapitulatif global
    all_results_flat = [r for v in all_results.values() for r in v]
    total_images = len(all_results_flat)
    total_ok     = safe_sum(r["success"] for r in all_results_flat)
    total_fail   = total_images - total_ok
    total_night  = safe_sum(
        r.get("is_night", False)
        for r in all_results_flat
        if r.get("success")
    )

    # Distribution quality_flag
    from collections import Counter
    qf_counter: Counter = Counter()
    for r in all_results_flat:
        if r.get("success"):
            flag = r.get("quality_flag") or "unknown"
            qf_counter[flag] += 1

    if total_images:
        print(f"\n{'='*55}")
        print("Pipeline terminé :")
        print(f"  ✔ {total_ok} images OK")
        print(f"  ❌ {total_fail} erreurs")
        print(f"  🌙 {total_night} images nocturnes ({total_night/max(total_ok,1)*100:.1f}%)")
        if qf_counter:
            print("  Quality :")
            for flag, count in sorted(qf_counter.items(), key=lambda x: -x[1]):
                print(f"    • {flag:<12}: {count}")
        print(f"{'='*55}\n")


# ============================================================
# Monitoring / diagnostic de qualité — fonctions utilitaires
# ============================================================

def summarize_quality(
    index_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> None:
    """
    Affiche un résumé de qualité global à partir de data/index/index.csv.

    Colonnes lues (toutes optionnelles — pas de crash si absentes) :
        quality_flag, is_night, downloaded, filename

    Args:
        index_path:  chemin vers index.csv (None = chemin par défaut du projet).
        config_path: chemin vers settings.yaml (None = chemin par défaut).

    Usage :
        from src.preprocessing import summarize_quality
        summarize_quality()
    """
    import pandas as pd

    # --- Résolution du chemin de l'index ---
    if index_path is None:
        try:
            cfg = load_config(config_path)
            index_path = PROJECT_ROOT / cfg["paths"]["index_file"]
        except Exception:
            # Fallback heuristique si la config n'est pas disponible
            index_path = PROJECT_ROOT / "data" / "index" / "index.csv"

    index_path = Path(index_path)

    if not index_path.exists():
        print(f"[summarize_quality] Index introuvable : {index_path}")
        return

    # --- Chargement ---
    try:
        df = pd.read_csv(index_path, low_memory=False)
    except Exception as exc:
        print(f"[summarize_quality] Impossible de lire l'index : {exc}")
        return

    n_total = len(df)
    sep = "=" * 48

    print(f"\n{sep}")
    print("         QUALITY SUMMARY")
    print(sep)
    print(f"Total images : {n_total}")

    # --- Distribution quality_flag ---
    if "quality_flag" in df.columns:
        flags = df["quality_flag"].fillna("unknown")
        counts = flags.value_counts(dropna=False)
        # Ordre d'affichage canonique
        order = ["usable", "dark", "cloudy", "corrupted", "unknown"]
        extra = [f for f in counts.index if f not in order]
        print("\nquality_flag :")
        for flag in order + extra:
            if flag in counts.index:
                pct = counts[flag] / n_total * 100 if n_total else 0
                print(f"  - {flag:<12}: {counts[flag]:>6}  ({pct:.1f}%)")
    else:
        print("\n[quality_flag] colonne absente de l'index")

    # --- Images nocturnes ---
    if "is_night" in df.columns:
        # is_night peut être bool, 0/1, ou NaN
        night_col = pd.to_numeric(df["is_night"], errors="coerce").fillna(0)
        n_night = int(night_col.sum())
        pct_night = n_night / n_total * 100 if n_total else 0
        print(f"\nImages nocturnes (is_night) : {n_night}  ({pct_night:.1f}%)")
    else:
        print("\n[is_night] colonne absente de l'index")

    # --- Images corrompues ---
    if "quality_flag" in df.columns:
        n_corrupted = int((df["quality_flag"].fillna("") == "corrupted").sum())
        print(f"Images corrompues            : {n_corrupted}")

    # --- Couverture du téléchargement ---
    if "downloaded" in df.columns:
        try:
            n_dl = int(pd.to_numeric(df["downloaded"], errors="coerce").fillna(0).sum())
            print(f"Images téléchargées          : {n_dl}  ({n_dl/n_total*100:.1f}%)")
        except Exception:
            pass

    print(sep + "\n")


def summarize_quality_by_month(
    index_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> None:
    """
    Affiche la distribution quality_flag par année/mois.

    Format par ligne :
        YYYY-MM  | total=XXX | usable=XX | dark=XX | cloudy=XX | corrupted=XX

    Args:
        index_path:  chemin vers index.csv (None = chemin par défaut).
        config_path: chemin vers settings.yaml (None = chemin par défaut).

    Usage :
        from src.preprocessing import summarize_quality_by_month
        summarize_quality_by_month()
    """
    import pandas as pd

    # --- Résolution du chemin de l'index ---
    if index_path is None:
        try:
            cfg = load_config(config_path)
            index_path = PROJECT_ROOT / cfg["paths"]["index_file"]
        except Exception:
            index_path = PROJECT_ROOT / "data" / "index" / "index.csv"

    index_path = Path(index_path)

    if not index_path.exists():
        print(f"[summarize_quality_by_month] Index introuvable : {index_path}")
        return

    try:
        df = pd.read_csv(index_path, low_memory=False)
    except Exception as exc:
        print(f"[summarize_quality_by_month] Impossible de lire l'index : {exc}")
        return

    required = {"year", "month", "quality_flag"}
    missing = required - set(df.columns)
    if missing:
        print(f"[summarize_quality_by_month] Colonnes manquantes : {missing}")
        return

    df["quality_flag"] = df["quality_flag"].fillna("unknown")
    df["year"]  = pd.to_numeric(df["year"],  errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df = df.dropna(subset=["year", "month"])
    df["year"]  = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    sep = "=" * 72
    print(f"\n{sep}")
    print("   QUALITY SUMMARY PAR MOIS")
    print(sep)
    print(f"  {'Période':<10} {'Total':>6}  {'usable':>7}  {'dark':>6}  {'cloudy':>7}  {'corrupted':>10}  {'unknown':>8}")
    print("-" * 72)

    flags_order = ["usable", "dark", "cloudy", "corrupted", "unknown"]

    for (y, m), group in df.groupby(["year", "month"], sort=True):
        period = f"{y}-{m:02d}"
        n = len(group)
        vc = group["quality_flag"].value_counts()
        counts = {f: vc.get(f, 0) for f in flags_order}
        print(
            f"  {period:<10} {n:>6}  "
            f"{counts['usable']:>7}  "
            f"{counts['dark']:>6}  "
            f"{counts['cloudy']:>7}  "
            f"{counts['corrupted']:>10}  "
            f"{counts['unknown']:>8}"
        )

    print(sep + "\n")


if __name__ == "__main__":
    main()

