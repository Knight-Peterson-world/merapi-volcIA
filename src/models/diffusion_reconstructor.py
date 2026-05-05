"""
diffusion_reconstructor.py — Reconstruction d'images "normales" via diffusion.

Principe :
  1. Encode l'image réelle dans l'espace latent (VAE de SD 1.5 ou PIL fallback).
  2. Ajoute du bruit contrôlé (strength < 1 → préserve la structure globale).
  3. Débruite avec EulerAncestral (img2img) vers une image "volcan normal".
  4. Calcule la carte de différence : anomalie = |réel − reconstruit|.

Optimisations v2 :
  - Scheduler EulerAncestralDiscrete (30 % plus rapide que PNDM)
  - Deux modes : "fast" (384 px, 15 steps) et "precise" (512 px, 25 steps)
  - attention_slicing + vae_slicing sur MPS/CPU (réduit les swaps mémoire)
  - xformers activé automatiquement sur CUDA
  - float16 sur CUDA, float32 sur MPS/CPU (MPS float16 instable)
  - build_img2img_pipeline() séparé → compatible @st.cache_resource

Usage (Streamlit) :
    pipe, backend = build_img2img_pipeline(lora_path=..., device="auto")
    rec = DiffusionReconstructor(pipeline=pipe, backend=backend)
    result = rec.reconstruct(image_path, strength=0.35, quality_mode="fast")

Usage legacy (sans Streamlit) :
    rec = DiffusionReconstructor(lora_path=..., strength=0.35)
    result = rec.reconstruct(image_path)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger("diffusion_reconstructor")

# ─── Modes de qualité ────────────────────────────────────────────────────
# steps = étapes totales du scheduler ; étapes effectives ≈ strength × steps
_QUALITY_MODES: dict[str, dict] = {
    "fast":    {"size": (384, 384), "steps": 15, "guidance": 6.0},
    "precise": {"size": (512, 512), "steps": 25, "guidance": 7.5},
}
_DEFAULT_QUALITY = "fast"
_DEFAULT_STRENGTH = 0.35
_DEFAULT_PROMPT = (
    "Merapi volcano crater, normal volcanic activity, clear sky, daytime, "
    "no eruption, surveillance camera Kalor, gray terrain, calm"
)
_DEFAULT_NEG = (
    "pyroclastic flow, lava, eruption, explosion, dark smoke, anomaly, "
    "blurry, artifact, text, watermark"
)


# ─────────────────────────────────────────────────────────────────────────────
# Fonction autonome — wrappable avec @st.cache_resource
# ─────────────────────────────────────────────────────────────────────────────

def build_img2img_pipeline(
    lora_path: Path | str | None = None,
    device: str = "auto",
) -> tuple[Any, str]:
    """
    Charge le pipeline SD 1.5 img2img une seule fois.

    Conçu pour être appelé via @st.cache_resource afin d'éviter tout
    rechargement lors des interactions Streamlit.

    Args:
        lora_path : dossier LoRA volcanique (optionnel).
        device    : "auto" | "cuda" | "mps" | "cpu".

    Returns:
        (pipeline, backend) où backend = "diffusers" | "fallback".
    """
    # Détection du device
    if device == "auto":
        device = _detect_device()

    try:
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler

        # fp16 sur CUDA seulement — MPS/CPU : NaN avec fp16
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info("Chargement SD 1.5 img2img (device=%s, dtype=%s)…", device, dtype)
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

        # ── Scheduler rapide : EulerAncestral converge mieux que PNDM ──────
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )

        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)

        # ── Optimisations mémoire ──────────────────────────────────────────
        pipe.enable_attention_slicing()          # réduit le pic VRAM / RAM
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()            # decode VAE par bandes

        # xformers uniquement sur CUDA (non dispo sur MPS)
        if device == "cuda":
            try:
                pipe.enable_xformers_memory_efficient_attention()
                logger.info("xformers activé")
            except Exception:
                pass  # non bloquant

        # ── LoRA volcanique ────────────────────────────────────────────────
        if lora_path is not None:
            lora_path = Path(lora_path)
        if lora_path and lora_path.exists():
            try:
                pipe.load_lora_weights(str(lora_path))
                logger.info("LoRA volcanique chargé : %s", lora_path.name)
            except Exception as e:
                logger.warning("LoRA non chargé (non bloquant) : %s", e)

        logger.info("Pipeline SD 1.5 img2img prêt sur %s", device)
        return pipe, "diffusers"

    except Exception as e:
        logger.warning("diffusers non disponible (%s) — fallback léger activé.", e)
        return None, "fallback"


def _detect_device() -> str:
    """Détecte le meilleur device disponible."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Classe principale
# ─────────────────────────────────────────────────────────────────────────────

class DiffusionReconstructor:
    """
    Reconstruit une image de volcan vers son état "normal" via img2img diffusion.

    Usage recommandé (Streamlit) :
        pipe, backend = build_img2img_pipeline(lora_path=..., device="auto")
        rec = DiffusionReconstructor(pipeline=pipe, backend=backend)
        result = rec.reconstruct(img_path, strength=0.35, quality_mode="fast")

    Usage legacy (lazy-load interne, pour scripts) :
        rec = DiffusionReconstructor(lora_path=..., strength=0.35)
        result = rec.reconstruct(img_path)

    Args:
        pipeline     : pipeline pré-chargé (issu de build_img2img_pipeline).
        backend      : "diffusers" | "fallback" (déduit si pipeline fourni).
        lora_path    : dossier LoRA (legacy — ignoré si pipeline est fourni).
        strength     : niveau img2img par défaut (0 = copie, 1 = pur génératif).
        quality_mode : "fast" (384 px, 15 steps) | "precise" (512 px, 25 steps).
        device       : "auto" | "cuda" | "mps" | "cpu" (legacy).
    """

    def __init__(
        self,
        pipeline: Any = None,
        backend: str | None = None,
        lora_path: Path | str | None = None,
        strength: float = _DEFAULT_STRENGTH,
        quality_mode: str = _DEFAULT_QUALITY,
        device: str = "auto",
    ) -> None:
        self.strength = float(np.clip(strength, 0.05, 0.95))
        self.quality_mode = quality_mode if quality_mode in _QUALITY_MODES else _DEFAULT_QUALITY

        if pipeline is not None:
            # Mode recommandé : pipeline injecté de l'extérieur
            self._pipe = pipeline
            self._backend = backend if backend is not None else "diffusers"
            self.device = "external"
        elif backend == "fallback":
            # Fallback explicite — pas de lazy-load
            self._pipe = None
            self._backend = "fallback"
            self.lora_path = None
            self.device = "cpu"
        else:
            # Mode legacy : lazy-load interne
            self._pipe = None
            self._backend = "none"
            self.lora_path = Path(lora_path) if lora_path else None
            self.device = device if device != "auto" else _detect_device()

    # ─── Chargement lazy (mode legacy uniquement) ─────────────────────────

    def _load_pipeline(self) -> bool:
        """Charge le pipeline si pas encore fait (mode legacy)."""
        if self._pipe is not None:
            return self._backend == "diffusers"
        if self._backend == "fallback":
            return False

        lora = getattr(self, "lora_path", None)
        self._pipe, self._backend = build_img2img_pipeline(
            lora_path=lora, device=getattr(self, "device", "auto")
        )
        return self._backend == "diffusers"

    # ─── API principale ───────────────────────────────────────────────────

    def reconstruct(
        self,
        image_input: Path | str | np.ndarray | Image.Image,
        prompt: str = _DEFAULT_PROMPT,
        negative_prompt: str = _DEFAULT_NEG,
        seed: int | None = 42,
        strength: float | None = None,
        quality_mode: str | None = None,
    ) -> dict[str, Any]:
        """
        Reconstruit une image vers son état "volcan normal".

        Args:
            image_input    : chemin, numpy array (H×W×3) ou PIL Image.
            prompt         : description de l'état normal cible.
            negative_prompt: éléments à éviter dans la reconstruction.
            seed           : graine pour la reproductibilité (None = aléatoire).
            strength       : écrase self.strength si fourni.
            quality_mode   : écrase self.quality_mode si fourni.

        Returns:
            dict :
              "original"      → np.ndarray H×W×3 uint8
              "reconstructed" → np.ndarray H×W×3 uint8
              "diff_map"      → np.ndarray H×W float32 [0, 1]
              "anomaly_score" → float (MAE normalisée × 100)
              "backend"       → str "diffusers" | "fallback"
              "quality_mode"  → str "fast" | "precise"
        """
        _strength = float(np.clip(strength if strength is not None else self.strength, 0.05, 0.95))
        _qmode = quality_mode if quality_mode in _QUALITY_MODES else self.quality_mode
        _params = _QUALITY_MODES[_qmode]
        _work_size: tuple[int, int] = _params["size"]

        orig_pil = self._load_image(image_input)
        orig_arr = np.array(orig_pil.resize(_work_size))

        # ── Assure que le pipeline est chargé (mode legacy) ────────────────
        ok = self._load_pipeline() if self._pipe is None else (self._backend == "diffusers")

        if ok and self._backend == "diffusers":
            recon_arr = self._reconstruct_diffusion(
                orig_pil, prompt, negative_prompt, seed,
                _strength, _params["steps"], _params["guidance"], _work_size,
            )
        else:
            recon_arr = self._reconstruct_fallback(orig_arr)

        orig_f  = orig_arr.astype(np.float32) / 255.0
        recon_f = recon_arr.astype(np.float32) / 255.0
        diff_map = np.abs(orig_f - recon_f).mean(axis=2)  # H×W [0,1]
        anomaly_score = float(diff_map.mean() * 100)

        return {
            "original":      orig_arr,
            "reconstructed": recon_arr,
            "diff_map":      diff_map,
            "anomaly_score": anomaly_score,
            "backend":       self._backend,
            "quality_mode":  _qmode,
        }

    # ─── Reconstruction via diffusers ─────────────────────────────────────

    def _reconstruct_diffusion(
        self,
        orig_pil: Image.Image,
        prompt: str,
        negative_prompt: str,
        seed: int | None,
        strength: float,
        steps: int,
        guidance: float,
        work_size: tuple[int, int],
    ) -> np.ndarray:
        """Reconstruction img2img avec Stable Diffusion."""
        try:
            import torch
            img_sd = orig_pil.resize(work_size).convert("RGB")
            # Génère le générateur sur le bon device
            _gen_device = "cpu" if self.device in ("external", "mps") else self.device
            generator = (
                torch.Generator(device=_gen_device).manual_seed(seed)
                if seed is not None else None
            )
            result = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=img_sd,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )
            return np.array(result.images[0].resize(work_size))
        except Exception as e:
            logger.warning("Erreur img2img (%s) — fallback.", e)
            return self._reconstruct_fallback(np.array(orig_pil.resize(work_size)))

    # ─── Fallback léger (sans GPU) ────────────────────────────────────────

    @staticmethod
    def _reconstruct_fallback(img: np.ndarray) -> np.ndarray:
        """Approximation légère sans GPU : flou gaussien + mélange."""
        pil = Image.fromarray(img.astype(np.uint8))
        blurred = pil.filter(ImageFilter.GaussianBlur(radius=8))
        blended = Image.blend(pil, blurred, alpha=0.7)
        return np.array(blended)

    # ─── Colorisation de la carte de différence ───────────────────────────

    @staticmethod
    def colorize_diff(diff_map: np.ndarray, percentile_clip: float = 99.0) -> np.ndarray:
        """
        Convertit une carte de différence [0,1] en image RGB colorisée (jet colormap).

        Args:
            diff_map       : H×W float32 [0,1]
            percentile_clip: percentile pour le clipping (évite les outliers)

        Returns:
            np.ndarray H×W×3 uint8 colorisé
        """
        vmax = float(np.percentile(diff_map, percentile_clip))
        normalized = np.clip(diff_map / (vmax + 1e-8), 0, 1)
        r = np.clip(1.5 - abs(4 * normalized - 3), 0, 1)
        g = np.clip(1.5 - abs(4 * normalized - 2), 0, 1)
        b = np.clip(1.5 - abs(4 * normalized - 1), 0, 1)
        return (np.stack([r, g, b], axis=2) * 255).astype(np.uint8)

    # ─── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_image(source: Path | str | np.ndarray | Image.Image) -> Image.Image:
        """Charge une image depuis n'importe quelle source."""
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, np.ndarray):
            return Image.fromarray(source.astype(np.uint8)).convert("RGB")
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image introuvable : {path}")
        return Image.open(path).convert("RGB")

    @property
    def is_diffusion_ready(self) -> bool:
        """True si le pipeline diffusers est chargé et opérationnel."""
        return self._backend == "diffusers" and self._pipe is not None
