"""
generator.py — Génération d'images volcaniques par IA.

Backends supportés :
  1. Hugging Face Diffusers (Stable Diffusion 2.1 / 1.5) — GPU local
     ↳ optimisé Apple Silicon M1 (MPS) : float16, attention slicing,
       fallback CPU automatique, nettoyage mémoire après génération.
  2. OpenAI DALL·E 3 — API cloud

Contraintes M1 / MPS :
  - Mémoire unifiée ~9 Go : SDXL OOM systématique.
  - SD 2.1 en float16 ≈ 3,5 Go → OK.
  - torch.Generator DOIT être sur "cpu" (pas "mps").
  - Nettoyage torch.mps.empty_cache() après chaque génération.
  - Fallback automatique CPU si MPS échoue.

Versions requises (compatibilité vérifiée) :
  torch >= 2.4          (support MPS stable)
  diffusers >= 0.28, < 1.0
  transformers >= 4.44, < 5.0   (⚠️ PAS transformers 5.x = breaking changes)
  accelerate >= 0.33

Usage :
    from src.generator import ImageGenerator
    gen = ImageGenerator()
    img = gen.generate("Coulée de lave sur le Merapi, nuit", backend="auto")
"""

from __future__ import annotations

import gc
import io
import os
# Désactiver TensorFlow dans transformers/diffusers (conflit protobuf dans conda base).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import base64
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Constantes ──────────────────────────────────────────────
SUPPORTED_BACKENDS = ("diffusers", "openai")
DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, drawing, anime, low quality, blurry, watermark, text, "
    "unrealistic colors, oversaturated"
)

_DIFFUSERS_MODEL_MAP = {
    # SD 1.5 — repo public (accessible sans token)
    "Stable Diffusion 1.5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    # SD 2.1 — ⚠ modèle gated (Stability AI a rendu privé en 2025).
    # Nécessite un compte HuggingFace + acceptation des conditions :
    # https://huggingface.co/stabilityai/stable-diffusion-2-1
    # puis : huggingface-cli login
    "Stable Diffusion 2.1 (gated)": "stabilityai/stable-diffusion-2-1",
    # SDXL — trop lourd pour MPS, désactivé sur M1
    "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0",
}

# Modèles trop lourds pour MPS (mémoire unifiée ≤ 16 Go)
_MPS_BLACKLIST = {
    "stabilityai/stable-diffusion-xl-base-1.0",
    # SD 2.1 en float32 = 7-8 Go → pression mémoire critique sur M1 8 Go
    # (le laisser disponible pour les M1 16 Go)
}

# Modèles nécessitant une authentification HuggingFace
_GATED_MODELS = {
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-xl-base-1.0",
}

# Résolution max par device (largeur ou hauteur)
_MAX_RESOLUTION = {"mps": 512, "cuda": 1024, "cpu": 512}


def _parse_resolution(res_str: str) -> tuple[int, int]:
    """'512×512' ou '512x512' → (512, 512)."""
    parts = res_str.replace("×", "x").split("x")
    return int(parts[0]), int(parts[1])


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


def _cleanup_memory(device: str) -> None:
    """Libère la mémoire GPU/MPS après une génération."""
    gc.collect()
    try:
        import torch
        if device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass


def check_versions() -> dict[str, str]:
    """Vérifie les versions des dépendances critiques et retourne un rapport."""
    report: dict[str, str] = {}
    for name in ("torch", "diffusers", "transformers", "accelerate", "safetensors"):
        try:
            mod = __import__(name)
            report[name] = getattr(mod, "__version__", "?")
        except ImportError:
            report[name] = "NOT INSTALLED"
    return report


# ── Backend : Hugging Face Diffusers ────────────────────────

class DiffusersBackend:
    """Génération locale via Stable Diffusion (torch + diffusers).

    Optimisations Apple Silicon :
    - float16 sur MPS (réduit de moitié la consommation mémoire)
    - attention_slicing("max") (réduit le pic mémoire UNet)
    - VAE slicing (décode l'image latente par tranches)
    - Generator toujours sur CPU (bug MPS connu pytorch/pytorch#97292)
    - Nettoyage mémoire après chaque inférence
    - Fallback automatique CPU si MPS OOM
    """

    def __init__(self) -> None:
        self._pipe = None
        self._current_model: str | None = None
        self._device: str = "cpu"
        self._lora_loaded: str | None = None  # chemin du LoRA actuellement chargé

    @staticmethod
    def is_available() -> bool:
        try:
            import torch  # noqa: F401
            import diffusers  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def models_for_device(device: str | None = None) -> list[str]:
        """Liste les modèles compatibles avec le device courant."""
        if device is None:
            device = _detect_device()
        models = []
        for name, repo in _DIFFUSERS_MODEL_MAP.items():
            if device == "mps" and repo in _MPS_BLACKLIST:
                continue
            models.append(name)
        return models

    def _load_pipeline(self, model_name: str, force_device: str | None = None) -> None:
        """Charge ou change le pipeline Stable Diffusion.

        Args:
            model_name: Nom du modèle (clé de _DIFFUSERS_MODEL_MAP).
            force_device: Si spécifié, force le device (ex: "cpu" pour fallback).
        """
        repo_id = _DIFFUSERS_MODEL_MAP.get(model_name, model_name)
        target_device = force_device or _detect_device()

        # Si déjà chargé sur le bon device, réutiliser
        if (
            self._pipe is not None
            and self._current_model == repo_id
            and self._device == target_device
        ):
            return

        import torch
        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

        self._device = target_device

        # ── Dtype MPS ───────────────────────────────────────────────────────────
        # float16 MPS = ~4 Go → OK pour M1 8 Go.
        # float32 MPS = ~7-8 Go → dépasse la mémoire M1 8 Go → swap → très lent.
        # → On utilise float16 sur MPS.
        #
        # Bug images NOIRES float16 sur MPS :
        # Les scores d'attention Q@K.T / sqrt(d) débordent float16 (max=65504).
        # Fix : upcast_attention=True (softmax en float32, poids restent float16).
        # Appliqué après chargement via patch sur chaque couche Attention du UNet.
        if self._device == "mps":
            dtype = torch.float16
        elif self._device == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Bloquer SDXL sur MPS
        if self._device == "mps" and repo_id in _MPS_BLACKLIST:
            raise RuntimeError(
                f"Le modèle {model_name} est trop volumineux pour Apple Silicon.\n"
                f"Utilisez 'Stable Diffusion 1.5' sur ce Mac (< 16 Go RAM)."
            )

        # Modèles gated : prévenir avant de tenter le téléchargement
        if repo_id in _GATED_MODELS:
            logger.warning(
                "%s est un modèle gated Stability AI. "
                "Téléchargement impossible sans : huggingface-cli login",
                repo_id,
            )

        # Libérer l'ancien pipeline
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            _cleanup_memory(self._device)

        logger.info("Chargement %s sur %s (dtype=%s)…", repo_id, self._device, dtype)

        PipeClass = StableDiffusionXLPipeline if "xl" in repo_id.lower() else StableDiffusionPipeline

        # ── Stratégie de chargement (3 tentatives) ──────────────────────────────
        # 1. variant="fp16" + safetensors  (fichiers *.fp16.safetensors en cache)
        # 2. variant=None  + safetensors   (fichiers *.safetensors sans suffix)
        # 3. variant=None  sans safetensors (fichiers *.bin, fallback universel)
        load_kwargs_list = [
            dict(torch_dtype=dtype, use_safetensors=True,  variant="fp16"),
            dict(torch_dtype=dtype, use_safetensors=True,  variant=None),
            dict(torch_dtype=dtype, use_safetensors=False, variant=None),
        ]

        pipe = None
        last_error: Exception | None = None
        for kwargs in load_kwargs_list:
            try:
                pipe = PipeClass.from_pretrained(repo_id, **kwargs)
                logger.info(
                    "Pipeline chargé avec variant=%s safetensors=%s",
                    kwargs.get("variant"), kwargs.get("use_safetensors"),
                )
                break
            except (OSError, EnvironmentError) as e:
                last_error = e
                logger.debug("Tentative échouée (%s) : %s", kwargs, e)

        if pipe is None:
            hint = (
                "Modèle non trouvé en cache et/ou connexion HuggingFace impossible.\n"
                "Téléchargez manuellement :\n"
                f"  huggingface-cli download {repo_id}\n"
                "Ou lancez une fois avec accès internet."
            )
            raise OSError(f"Impossible de charger {repo_id}.\n{hint}") from last_error

        self._pipe = pipe.to(self._device)

        # ── Fix images noires sur MPS (float16 NaN dans l'attention) ────────────
        # Les scores d'attention Q@K.T débordent float16 (max=65504).
        # upcast_attention=True force le softmax en float32 uniquement.
        if self._device == "mps":
            try:
                from diffusers.models.attention_processor import Attention as _Attn
                _n = sum(
                    1 for _m in self._pipe.unet.modules()
                    if isinstance(_m, _Attn)
                    and not setattr(_m, "upcast_attention", True)
                )
                logger.debug("upcast_attention=True appliqué sur %d couches (fix NaN MPS fp16)", _n)
            except (ImportError, AttributeError) as _e:
                logger.warning("upcast_attention patch impossible : %s", _e)

        # ── Optimisations mémoire ────────────────────────────────────────────────
        self._pipe.enable_attention_slicing(1)  # slice=1 pour MPS fp16
        # VAE slicing : API changée en diffusers 0.30+
        if hasattr(self._pipe, "vae") and hasattr(self._pipe.vae, "enable_slicing"):
            self._pipe.vae.enable_slicing()
        elif hasattr(self._pipe, "enable_vae_slicing"):
            self._pipe.enable_vae_slicing()

        # Désactiver le safety checker (inutile pour volcanologie, réduit la mémoire)
        if hasattr(self._pipe, "safety_checker"):
            self._pipe.safety_checker = None
        if hasattr(self._pipe, "requires_safety_checker"):
            self._pipe.requires_safety_checker = False

        self._current_model = repo_id
        self._lora_loaded = None  # Reset LoRA quand le modèle change
        logger.info("Modèle prêt : %s sur %s", repo_id, self._device)

    def load_lora(self, lora_path: str | Path) -> None:
        """Charge un adaptateur LoRA (PEFT) sur le pipeline actif.

        Args:
            lora_path: Chemin vers le dossier contenant adapter_model.safetensors
                       et adapter_config.json.
        """
        lora_path = Path(lora_path)
        lora_str = str(lora_path)

        if self._lora_loaded == lora_str:
            logger.info("LoRA déjà chargé : %s", lora_str)
            return

        if self._pipe is None:
            raise RuntimeError("Pipeline non chargé. Appelez generate() d'abord.")

        if not lora_path.exists():
            logger.warning("LoRA introuvable : %s", lora_path)
            return

        import torch
        from peft import PeftModel

        logger.info("Chargement LoRA : %s", lora_path)

        # Sauvegarder le UNet original si c'est le premier LoRA
        if not hasattr(self, "_unet_original_state"):
            self._unet_original_state = None

        unet = self._pipe.unet
        unet = PeftModel.from_pretrained(unet, lora_str)
        # Unwrap PEFT pour le pipeline
        self._pipe.unet = unet.base_model.model
        self._lora_loaded = lora_str
        logger.info("LoRA chargé avec succès")

    def unload_lora(self) -> None:
        """Décharge le LoRA actif (recharge le pipeline sans LoRA)."""
        if self._lora_loaded is not None:
            logger.info("Déchargement LoRA — rechargement du pipeline")
            model = self._current_model
            device = self._device
            del self._pipe
            self._pipe = None
            self._lora_loaded = None
            _cleanup_memory(device)
            # Recharger pour restaurer les noms de modèle
            for name, repo in _DIFFUSERS_MODEL_MAP.items():
                if repo == model:
                    self._load_pipeline(name, force_device=device)
                    break

    def generate(
        self,
        prompt: str,
        model_name: str = "Stable Diffusion 1.5",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        seed: int | None = None,
    ) -> Image.Image:
        """Génère une image. Fallback automatique MPS → CPU si OOM."""
        return self._generate_with_fallback(
            prompt=prompt,
            model_name=model_name,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            seed=seed,
        )

    def _generate_with_fallback(
        self,
        prompt: str,
        model_name: str,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        negative_prompt: str,
        seed: int | None,
        _retry_cpu: bool = False,
    ) -> Image.Image:
        """Tente la génération ; si MPS OOM, re-essaye automatiquement sur CPU."""
        device_to_use = "cpu" if _retry_cpu else None
        self._load_pipeline(model_name, force_device=device_to_use)

        import torch

        # Generator TOUJOURS sur CPU (bug MPS connu)
        generator = torch.Generator("cpu")
        if seed is not None:
            generator.manual_seed(seed)
        else:
            import random
            generator.manual_seed(random.randint(0, 2**32 - 1))

        try:
            result = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            return result.images[0]

        except RuntimeError as e:
            err_msg = str(e).lower()
            is_oom = any(k in err_msg for k in ("mps", "memory", "out of memory", "allocat"))

            if is_oom and self._device == "mps" and not _retry_cpu:
                logger.warning("MPS OOM — fallback automatique sur CPU…")
                _cleanup_memory("mps")
                # Libérer le pipeline MPS
                del self._pipe
                self._pipe = None
                _cleanup_memory("mps")
                return self._generate_with_fallback(
                    prompt=prompt,
                    model_name=model_name,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    _retry_cpu=True,
                )
            raise  # Autre erreur ou déjà sur CPU → propagate

        finally:
            _cleanup_memory(self._device)


# ── Backend : OpenAI DALL·E ─────────────────────────────────

class OpenAIBackend:
    """Génération via l'API OpenAI (DALL·E 3)."""

    def __init__(self) -> None:
        self._client = None

    @staticmethod
    def is_available() -> bool:
        try:
            import openai  # noqa: F401
            return bool(os.environ.get("OPENAI_API_KEY"))
        except ImportError:
            return False

    def _ensure_client(self) -> None:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()  # utilise OPENAI_API_KEY

    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        **_kwargs: Any,
    ) -> Image.Image:
        self._ensure_client()

        # DALL-E 3 supporte 1024x1024, 1024x1792, 1792x1024
        if width <= 512 and height <= 512:
            size = "1024x1024"
        elif width > height:
            size = "1792x1024"
        elif height > width:
            size = "1024x1792"
        else:
            size = "1024x1024"

        response = self._client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality="standard",
            n=1,
            response_format="b64_json",
        )

        img_data = base64.b64decode(response.data[0].b64_json)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")

        # Redimensionner à la taille demandée si différente
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)

        return img


# ── Façade principale ───────────────────────────────────────

class ImageGenerator:
    """
    Point d'entrée unique pour la génération d'images.

    Exemples :
        gen = ImageGenerator()
        gen.available_backends()        # ['diffusers', 'openai']
        img = gen.generate("Lave incandescente", backend="auto")
    """

    def __init__(self) -> None:
        self._diffusers = DiffusersBackend()
        self._openai = OpenAIBackend()
        self.device = _detect_device()

    def available_backends(self) -> list[str]:
        """Liste les backends prêts à l'emploi."""
        available = []
        if self._diffusers.is_available():
            available.append("diffusers")
        if self._openai.is_available():
            available.append("openai")
        return available

    def diffusers_model_options(self) -> list[str]:
        """Modèles Diffusers compatibles avec le device courant."""
        return DiffusersBackend.models_for_device(self.device)

    @property
    def is_mps(self) -> bool:
        return self.device == "mps"

    def _resolve_backend(
        self, backend: str, model_name: str
    ) -> Literal["diffusers", "openai"]:
        if backend != "auto":
            return backend  # type: ignore[return-value]
        if "DALL" in model_name.upper():
            return "openai"
        return "diffusers"

    def generate(
        self,
        prompt: str,
        *,
        backend: str = "auto",
        model_name: str = "Stable Diffusion 2.1",
        resolution: str = "512×512",
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        seed: int | None = None,
    ) -> Image.Image:
        """
        Génère une image à partir d'un prompt textuel.

        Args:
            prompt: Description de la scène volcanique.
            backend: 'diffusers', 'openai', ou 'auto'.
            model_name: Nom du modèle affiché dans l'UI.
            resolution: '512×512', '768×768', '1024×1024'.
            num_inference_steps: Nombre d'étapes de diffusion.
            guidance_scale: Guidance scale (classifier-free guidance).
            negative_prompt: Prompt négatif (diffusers uniquement).
            seed: Graine de reproductibilité (diffusers uniquement).

        Returns:
            PIL.Image.Image

        Raises:
            RuntimeError: si aucun backend n'est disponible.
        """
        width, height = _parse_resolution(resolution)
        resolved = self._resolve_backend(backend, model_name)

        if resolved == "openai":
            if not self._openai.is_available():
                raise RuntimeError(
                    "Backend OpenAI non disponible.\n"
                    "Installez : pip install openai\n"
                    "Puis définissez : export OPENAI_API_KEY='sk-...'"
                )
            logger.info("Génération via OpenAI DALL·E 3")
            return self._openai.generate(
                prompt=prompt, width=width, height=height,
            )

        if resolved == "diffusers":
            if not self._diffusers.is_available():
                raise RuntimeError(
                    "Backend Diffusers non disponible.\n"
                    "Installez : pip install torch diffusers transformers accelerate"
                )
            logger.info("Génération via Diffusers — %s", model_name)
            return self._diffusers.generate(
                prompt=prompt,
                model_name=model_name,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                seed=seed,
            )

        raise RuntimeError(f"Backend inconnu : {resolved}")

    def load_lora(self, lora_path: str | Path) -> None:
        """Charge un LoRA sur le backend Diffusers."""
        self._diffusers.load_lora(str(lora_path))

    def unload_lora(self) -> None:
        """Décharge le LoRA actif."""
        self._diffusers.unload_lora()

    @property
    def lora_loaded(self) -> str | None:
        """Chemin du LoRA actuellement chargé, ou None."""
        return self._diffusers._lora_loaded

    def generate_physics(
        self,
        camera: str,
        time_of_day: str,
        brightness: str,
        lava_intensity: str,
        slope: str,
        weather: str,
        *,
        viscosity: str = "medium",
        temperature: str = "moderate",
        eruption_type: str = "none",
        plume: str = "none",
        model_name: str = "Stable Diffusion 1.5",
        resolution: str = "512×512",
        num_inference_steps: int = 30,
        guidance_scale: float = 10.0,
        seed: int | None = None,
        lora_path: str | Path | None = None,
    ) -> Image.Image:
        """Génère une image avec des paramètres physiques.

        Construit un prompt en LANGAGE NATUREL riche (compréhensible par
        CLIP) au lieu de paires key=value que l'encodeur texte ignore.

        Paramètres physiques (v2) :
            camera, time_of_day, brightness, lava_intensity, slope, weather,
            viscosity, temperature, eruption_type, plume.

        Returns:
            PIL.Image.Image
        """
        from src.physics_prompts import build_rich_prompt, PHYSICS_NEGATIVE_PROMPT

        # Prompt NL riche (template aléatoire pour la diversité)
        prompt = build_rich_prompt(
            camera=camera,
            time_of_day=time_of_day,
            brightness=brightness,
            lava_intensity=lava_intensity,
            slope=slope,
            weather=weather,
            viscosity=viscosity,
            temperature=temperature,
            eruption_type=eruption_type,
            plume=plume,
            template_index=None,  # aléatoire pour l'inférence
        )

        # Charger le LoRA si demandé
        if lora_path is not None:
            self._diffusers._load_pipeline(model_name)
            self._diffusers.load_lora(lora_path)

        return self.generate(
            prompt=prompt,
            backend="diffusers",
            model_name=model_name,
            resolution=resolution,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=PHYSICS_NEGATIVE_PROMPT,
            seed=seed,
        )
