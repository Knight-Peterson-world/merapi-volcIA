"""
processor.py — Pipeline de preprocessing sans bugs, quality_flag garanti.

Règles strictes :
  - Chaque image (nouvelle OU déjà traitée) reçoit un quality_flag réel.
  - Jamais quality_flag='unknown' pour une image accessible.
  - is_night est toujours un bool (jamais None).
  - Toutes les erreurs sont loguées et isolées — pas de crash pipeline.
  - Idempotent : re-run safe.

Flux par image :
    raw/*.jpg  →  resize (256×256)  →  normalize  →  processed/*.png
                                                       ↓
                                              QualityClassifier
                                                       ↓
                                    {quality_flag, is_night, mean_brightness, ...}

Usage :
    from src.ingestion.processor import ImageProcessor
    from src.utils import load_config

    cfg  = load_config()
    proc = ImageProcessor(cfg)
    results = proc.process_month(2019, 6)          # list[dict]
    all_r   = proc.process_all(year=2019)          # dict[str, list[dict]]
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.utils import (
    PROJECT_ROOT,
    get_raw_image_dir,
    get_processed_image_dir,
    safe_sum,
)
from src.ingestion.quality import QualityClassifier, QUALITY_CORRUPTED

try:
    from loguru import logger
except ImportError:
    import logging as _l

    logger = _l.getLogger("ingestion.processor")  # type: ignore[assignment]


class ImageProcessor:
    """
    Prétraitement des images brutes Merapi.

    Garanties :
      - Dict retourné contient toujours : success (bool), is_night (bool),
        quality_flag (str ∈ {usable, dark, cloudy, corrupted}).
      - Idempotent : si PNG existe et overwrite=False → charge le PNG
        et recalcule les stats (pas de retour sans qualité).
      - Jamais de crash même si une image est corrompue.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config    = config
        pp             = config["preprocessing"]
        self.target_size   = tuple(pp["target_size"])
        self.normalization = pp.get("normalization", "minmax")
        self.quality       = QualityClassifier.from_config(config)

    # ----------------------------------------------------------
    # Traitement d'une image
    # ----------------------------------------------------------

    def process_image(
        self,
        raw_path: Path,
        proc_dir: Path,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """
        Prétraite une image et retourne ses métadonnées complètes.

        Si le PNG de sortie existe déjà et overwrite=False : charge le PNG
        existant et recalcule les stats — ne retourne JAMAIS sans quality_flag.

        Args:
            raw_path:  chemin de l'image brute.
            proc_dir:  dossier de sortie des PNGs traités.
            overwrite: si True, re-traite même si le PNG existe.

        Returns:
            dict avec les champs :
                raw_path, processed_path, filename,
                success, is_night, quality_flag,
                mean_brightness, std_brightness, variance, error
        """
        result: dict[str, Any] = {
            "raw_path":        str(raw_path),
            "processed_path":  "",
            "filename":        raw_path.name,
            "success":         False,
            "is_night":        False,
            "quality_flag":    QUALITY_CORRUPTED,
            "mean_brightness": None,
            "std_brightness":  None,
            "variance":        None,
            "error":           None,
        }

        png_out = proc_dir / (raw_path.stem + ".png")
        result["processed_path"] = str(png_out)

        # ── Cas : image déjà traitée → recalcul stats depuis PNG ──────────
        if not overwrite and png_out.exists():
            result["success"] = True
            qr = self.quality.classify(png_out)
            result.update({
                "quality_flag":    qr["quality_flag"],
                "is_night":        bool(qr["is_night"]),
                "mean_brightness": qr["mean_brightness"],
                "std_brightness":  qr["std_brightness"],
                "variance":        qr["variance"],
            })
            if qr.get("error"):
                result["quality_flag"] = QUALITY_CORRUPTED
                result["error"]        = qr["error"]
            logger.debug(
                f"Skip {raw_path.name} → {result['quality_flag']} "
                f"| lum={result['mean_brightness']}"
            )
            return result

        # ── Cas : nouvelle image ou overwrite ─────────────────────────────
        if not raw_path.exists():
            result["error"] = f"Fichier source introuvable : {raw_path}"
            logger.warning(result["error"])
            return result

        try:
            # Chargement et vérification d'intégrité
            img = Image.open(raw_path)
            img.verify()
            img = Image.open(raw_path)  # réouverture après verify()

            # ROI optionnelle
            roi_cfg = self.config["preprocessing"].get("roi", {})
            if roi_cfg.get("enabled", False):
                img = self._apply_roi(img, roi_cfg)

            # Niveaux de gris + resize
            img_gray    = img.convert("L")
            img_resized = img_gray.resize(self.target_size, Image.LANCZOS)

            # Normalisation
            arr      = np.array(img_resized, dtype=np.float32)
            arr_norm = self._normalize(arr)

            # Sauvegarde PNG lossless
            png_out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((arr_norm * 255).astype(np.uint8)).save(
                str(png_out), format="PNG"
            )

            # Qualité calculée sur le PNG sauvegardé (source de vérité)
            qr = self.quality.classify(png_out)
            result.update({
                "success":         True,
                "quality_flag":    qr["quality_flag"],
                "is_night":        bool(qr["is_night"]),
                "mean_brightness": qr["mean_brightness"],
                "std_brightness":  qr["std_brightness"],
                "variance":        qr["variance"],
            })
            if qr.get("error"):
                result["quality_flag"] = QUALITY_CORRUPTED
                result["error"]        = qr["error"]

            logger.debug(
                f"✓ {raw_path.name} → {self.target_size} "
                f"| {result['quality_flag']} | lum={result['mean_brightness']:.1f}"
            )

        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            logger.error(f"Erreur processing {raw_path.name}: {exc}")

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
    ) -> list[dict[str, Any]]:
        """
        Prétraite toutes les images d'un mois.

        Args:
            year, month:  période.
            overwrite:    re-traiter même si PNG existe déjà.
            max_images:   plafond (None = toutes).

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
        exts     = set(self.config["source"]["image_extensions"])

        raw_images = sorted(p for p in raw_dir.glob("*") if p.suffix in exts)

        if not raw_images:
            logger.warning(f"Aucune image dans {raw_dir}")
            return []

        if max_images and len(raw_images) > max_images:
            logger.info(f"Plafond : {max_images}/{len(raw_images)} images")
            raw_images = raw_images[:max_images]

        print(f"\nPreprocessing {year}/{month:02d} — {len(raw_images)} images")

        results = []
        for raw_path in tqdm(
            raw_images, desc=f"  {year}/{month:02d}", unit="img", leave=False
        ):
            res = self.process_image(raw_path, proc_dir, overwrite=overwrite)
            results.append(res)

        n_ok   = safe_sum(r["success"] for r in results)
        n_fail = len(results) - n_ok
        print(f"  ✔ {n_ok} OK | ❌ {n_fail} erreurs")
        if n_fail:
            for r in results:
                if not r["success"]:
                    logger.error(f"  → {r['filename']}: {r['error']}")

        return results

    # ----------------------------------------------------------
    # Traitement de toutes les données disponibles
    # ----------------------------------------------------------

    def process_all(
        self,
        year: int | None = None,
        month: int | None = None,
        overwrite: bool = False,
        max_per_month: int | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Prétraite tous les mois disponibles dans data/raw/.

        Args:
            year, month:    filtres optionnels.
            overwrite:      forcer le re-traitement.
            max_per_month:  plafond d'images par mois.

        Returns:
            Dict {"YYYY-MM": [résultats]}.
        """
        # Découverte dynamique des mois disponibles (réutilise la logique existante)
        from src.preprocessing import MerapiPreprocessor

        available = MerapiPreprocessor(self.config).discover_available_months(
            year=year, month=month
        )

        if not available:
            logger.warning("Aucun mois disponible dans data/raw/")
            return {}

        print(f"\n{'='*55}")
        print(f"ImageProcessor — {len(available)} mois à traiter")
        if year:
            print(f"  Filtre : {year}" + (f"/{month:02d}" if month else ""))
        print(f"{'='*55}")

        all_results: dict[str, list[dict[str, Any]]] = {}
        for y, m in available:
            key = f"{y}-{m:02d}"
            res = self.process_month(y, m, overwrite=overwrite, max_images=max_per_month)
            if res:
                all_results[key] = res

        self._print_summary(all_results)
        return all_results

    # ----------------------------------------------------------
    # Utilitaires internes
    # ----------------------------------------------------------

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        if self.normalization == "minmax":
            lo, hi = arr.min(), arr.max()
            if hi - lo < 1e-6:
                return np.zeros_like(arr, dtype=np.float32)
            return (arr - lo) / (hi - lo)
        if self.normalization == "zscore":
            mu, sigma = arr.mean(), arr.std()
            if sigma < 1e-6:
                return np.zeros_like(arr, dtype=np.float32)
            return (arr - mu) / sigma
        raise ValueError(f"Normalisation inconnue : {self.normalization}")

    @staticmethod
    def _apply_roi(img: Image.Image, roi_cfg: dict) -> Image.Image:
        coords = [roi_cfg.get(k) for k in ["x_min", "y_min", "x_max", "y_max"]]
        if any(c is None for c in coords):
            return img
        return img.crop(tuple(int(c) for c in coords))

    @staticmethod
    def _print_summary(all_results: dict[str, list[dict[str, Any]]]) -> None:
        flat   = [r for v in all_results.values() for r in v]
        n_ok   = safe_sum(r["success"] for r in flat)
        n_fail = len(flat) - n_ok
        qf_counts = Counter(
            r.get("quality_flag", "unknown")
            for r in flat
            if r.get("success")
        )
        night = safe_sum(r.get("is_night", False) for r in flat if r.get("success"))

        print(f"\n{'='*55}")
        print("Pipeline terminé :")
        print(f"  ✔ {n_ok} images OK")
        print(f"  ❌ {n_fail} erreurs")
        print(f"  🌙 {night} images nocturnes ({night / max(n_ok, 1) * 100:.1f}%)")
        if qf_counts:
            print("  Quality :")
            for flag, count in sorted(qf_counts.items(), key=lambda x: -x[1]):
                print(f"    • {flag:<12}: {count}")
        print(f"{'='*55}\n")
