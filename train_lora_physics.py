#!/usr/bin/env python3
"""
train_lora_physics.py — Fine-tuning LoRA de Stable Diffusion 1.5
                         avec conditionnement par paramètres physiques.

VERSION 2 — Corrections majeures :
  1. Prompts en LANGAGE NATUREL riche (au lieu de key=value que CLIP ignore)
  2. Augmentation couleur physiquement motivée (au lieu de gris → gris)
  3. LoRA rank 16 par défaut (au lieu de 4 → capacité insuffisante)
  4. Nouveaux paramètres : viscosité, température, type éruption, panache
  5. guidance_scale=10 pour l'inférence de test

Fonctionnalités :
  - Prompts NL riches via src/physics_prompts.py (shared module)
  - Coloration condition-aware des images webcam grises
  - Classifier-Free Guidance (CFG) training avec drop conditionnel 10%
  - LoRA rank configurable (défaut 16)
  - EMA, gradient accumulation, early stopping, cosine annealing
  - Génération de test avant/après avec 6 scénarios contrastés
  - Compatible MPS (fp32) et CUDA (fp16)

Usage :
    # Local (MPS) — test rapide
    python train_lora_physics.py --epochs 2 --max-images 20 --skip-gen

    # Colab / GPU (CUDA) — entraînement complet (50 epochs recommandés)
    python train_lora_physics.py --epochs 50 --resolution 256 --fp16

Prérequis :
    pip install torch diffusers transformers accelerate peft pillow pandas
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

# Empêcher l'import de TensorFlow (conflit protobuf)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Import du module prompt partagé ──
from src.physics_prompts import (
    PHYSICS_NEGATIVE_PROMPT,
    build_rich_prompt,
    build_prompt_from_metadata,
    apply_physics_color,
    camera_from_filename,
)


# ============================================================
# 1. Dataset Merapi avec paramètres physiques + couleur
# ============================================================

class MerapiPhysicsDataset(Dataset):
    """
    Dataset PyTorch qui charge les images Merapi, applique une
    coloration physiquement motivée, et construit des prompts
    en langage naturel riche.

    Chaque échantillon :
      - pixel_values : tensor [3, H, W] dans [-1, 1] (RGB coloré)
      - prompt       : description NL riche (CLIP-friendly)
      - metadata     : paramètres physiques extraits
    """

    def __init__(
        self,
        index_path: Path,
        processed_base: Path,
        resolution: int = 256,
        max_images: int | None = None,
        quality_flags: list[str] | None = None,
        min_anomaly: float = 0.0,
        exclude_dark_night: bool = True,
    ) -> None:
        self.resolution = resolution

        if quality_flags is None:
            quality_flags = ["usable"]

        df = pd.read_csv(index_path, dtype=str, na_values=["", "None", "nan"])
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["month"] = pd.to_numeric(df["month"], errors="coerce")
        df["anomaly_score"] = pd.to_numeric(
            df["anomaly_score"], errors="coerce"
        ).fillna(0.0)

        # Filtrer par qualité
        df = df[df["quality_flag"].isin(quality_flags)].copy()

        # Filtrer les nuits sans incandescence
        if exclude_dark_night:
            is_night = ~df["hour"].between(6, 17)
            has_incandescence = df["anomaly_score"] > 0.05
            df = df[~(is_night & ~has_incandescence)].copy()

        # Filtrer par score d'anomalie minimum
        if min_anomaly > 0:
            df = df[df["anomaly_score"] >= min_anomaly].copy()

        self.samples: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            y, m, fn = row.get("year"), row.get("month"), row.get("filename", "")
            if pd.isna(y) or pd.isna(m) or not fn:
                continue

            png_stem = Path(fn).stem + ".png"
            png_path = processed_base / str(int(y)) / f"{int(m):02d}" / png_stem
            if not png_path.exists():
                continue

            # Prompt NL riche (template fixe=0 pour pouvoir pré-cacher)
            prompt = build_prompt_from_metadata(row.to_dict(), template_index=0)

            self.samples.append({
                "path": png_path,
                "prompt": prompt,
                "hour": int(row.get("hour", 12)),
                "anomaly_score": float(row.get("anomaly_score", 0.0)),
                "camera": camera_from_filename(fn),
                "filename": fn,
            })

        # Limiter
        if max_images is not None and len(self.samples) > max_images:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(self.samples), max_images, replace=False)
            self.samples = [self.samples[i] for i in sorted(indices)]

        # Stats
        n_day = sum(1 for s in self.samples if 6 <= s["hour"] < 18)
        n_night = len(self.samples) - n_day
        n_active = sum(1 for s in self.samples if s["anomaly_score"] > 0.15)
        cameras = {}
        for s in self.samples:
            cameras[s["camera"]] = cameras.get(s["camera"], 0) + 1

        print(f"[Dataset] {len(self.samples)} images chargées")
        print(f"  Jour/Nuit: {n_day}/{n_night}")
        print(f"  Activité (anomaly>0.15): {n_active}")
        print(f"  Caméras: {cameras}")
        if self.samples:
            print(f"  [Exemple prompt] {self.samples[0]['prompt'][:120]}...")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        img = Image.open(sample["path"]).convert("L")

        if img.size != (self.resolution, self.resolution):
            img = img.resize((self.resolution, self.resolution), Image.LANCZOS)

        # ── CORRECTION CLÉ : coloration physiquement motivée ──
        # Au lieu de gris → gris (le modèle n'apprend aucune variation),
        # on applique une coloration basée sur les conditions réelles.
        img_rgb = apply_physics_color(
            img, sample["hour"], sample["anomaly_score"]
        )

        arr = np.array(img_rgb, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0

        return {
            "pixel_values": tensor,
            "prompt": sample["prompt"],
            "hour": sample["hour"],
            "anomaly_score": sample["anomaly_score"],
            "camera": sample["camera"],
        }


# ============================================================
# 2. Utilitaires modèle
# ============================================================

def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_lora_unet(unet, lora_rank: int = 16, lora_alpha: int = 16):
    """Configure LoRA sur le U-Net.

    Rank 16 (au lieu de 4) : capacité suffisante pour apprendre
    le conditionnement par paramètres physiques.
    """
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            # Cross-attention feed-forward — critiques pour le conditionnement
            "ff.net.0.proj", "ff.net.2",
        ],
    )
    unet = get_peft_model(unet, config)

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    print(f"[LoRA] r={lora_rank}, alpha={lora_alpha}")
    print(f"[LoRA] {trainable:,} / {total:,} params entraînables "
          f"({100 * trainable / total:.2f}%)")

    return unet


class EMAModel:
    """Exponential Moving Average des paramètres entraînables."""

    def __init__(self, parameters, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {i: p.clone().detach() for i, p in enumerate(parameters)}

    @torch.no_grad()
    def update(self, parameters):
        for i, p in enumerate(parameters):
            self.shadow[i].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_shadow(self, parameters):
        self.backup = {i: p.clone() for i, p in enumerate(parameters)}
        for i, p in enumerate(parameters):
            p.data.copy_(self.shadow[i])

    def restore(self, parameters):
        for i, p in enumerate(parameters):
            p.data.copy_(self.backup[i])


def _unwrap_unet(unet):
    """Extrait le UNet2DConditionModel depuis un wrapper PEFT."""
    if hasattr(unet, "base_model") and hasattr(unet.base_model, "model"):
        return unet.base_model.model
    return unet


def encode_prompt(prompt: str, tokenizer, text_encoder, device) -> torch.Tensor:
    """Encode un prompt texte en embedding CLIP."""
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        return text_encoder(tokens.input_ids.to(device)).last_hidden_state


# ============================================================
# 3. Scénarios de test (contrastés pour vérifier le conditionn.)
# ============================================================

# 6 scénarios TRÈS différents pour vérifier visuellement
# que le modèle répond aux conditions
TEST_SCENARIOS = [
    {
        "name": "jour_calme_clair",
        "params": {
            "camera": "Suki",
            "time_of_day": "midday",
            "brightness": "daylight",
            "lava_intensity": "none",
            "slope": "30deg_south",
            "weather": "clear",
        },
    },
    {
        "name": "nuit_eruption_intense",
        "params": {
            "camera": "Kalor",
            "time_of_day": "night",
            "brightness": "incandescent_glow",
            "lava_intensity": "very_high",
            "slope": "25deg_west",
            "weather": "clear_night",
            "viscosity": "low",
            "temperature": "extreme",
            "eruption_type": "effusive",
            "plume": "medium",
        },
    },
    {
        "name": "crepuscule_moderate",
        "params": {
            "camera": "Suki",
            "time_of_day": "dusk",
            "brightness": "dim_glow",
            "lava_intensity": "moderate",
            "slope": "30deg_south",
            "weather": "clear",
            "viscosity": "medium",
            "temperature": "moderate",
        },
    },
    {
        "name": "nuit_faible_couvert",
        "params": {
            "camera": "Kalor",
            "time_of_day": "night",
            "brightness": "dim_glow",
            "lava_intensity": "low",
            "slope": "25deg_west",
            "weather": "overcast",
            "viscosity": "high",
            "temperature": "low",
        },
    },
    {
        "name": "matin_brume_calme",
        "params": {
            "camera": "Kali",
            "time_of_day": "early_morning",
            "brightness": "daylight",
            "lava_intensity": "none",
            "slope": "35deg_east",
            "weather": "hazy",
        },
    },
    {
        "name": "nuit_explosive_panache",
        "params": {
            "camera": "Suki",
            "time_of_day": "night",
            "brightness": "incandescent_glow",
            "lava_intensity": "high",
            "slope": "30deg_south",
            "weather": "clear_night",
            "viscosity": "low",
            "temperature": "high",
            "eruption_type": "explosive",
            "plume": "high",
        },
    },
]


def generate_physics_samples(
    pipe,
    scenarios: list[dict],
    output_dir: Path,
    prefix: str = "test",
    num_inference_steps: int = 30,
    guidance_scale: float = 10.0,
    seed: int = 42,
) -> list[Path]:
    """Génère des images de test pour chaque scénario physique."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for scenario in scenarios:
        name = scenario["name"]
        params = scenario["params"]

        # ── CORRECTION CLÉ : prompt NL riche (pas key=value) ──
        prompt = build_rich_prompt(**params, template_index=0)

        print(f"  [{prefix}] {name}")
        print(f"    prompt: {prompt[:100]}...")

        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            result = pipe(
                prompt,
                negative_prompt=PHYSICS_NEGATIVE_PROMPT,
                num_inference_steps=num_inference_steps,
                generator=generator,
                guidance_scale=guidance_scale,
            )
        img = result.images[0]
        path = output_dir / f"{prefix}_{name}.png"
        img.save(str(path), format="PNG")
        paths.append(path)
        print(f"    → {path.name}")

    return paths


# ============================================================
# 4. Boucle d'entraînement
# ============================================================

def train(args: argparse.Namespace) -> None:
    from diffusers import (
        AutoencoderKL,
        DDPMScheduler,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextModel, CLIPTokenizer

    device = detect_device()
    dtype = torch.float32
    # fp16 auto-activé sur CUDA (sauf si --no-fp16 forcé)
    use_fp16 = (args.fp16 is True) or (args.fp16 is None and device.type == "cuda")
    if args.no_fp16:
        use_fp16 = False
    if use_fp16 and device.type == "cuda":
        dtype = torch.float16

    print(f"\n{'=' * 60}")
    print("ENTRAÎNEMENT LoRA PHYSICS v2 — Prompts NL + Couleur")
    print(f"{'=' * 60}")
    print(f"[Config] Device={device}, dtype={dtype}")
    print(f"[Config] Résolution={args.resolution}, LR={args.lr}, "
          f"Epochs={args.epochs}, Batch={args.batch_size}")
    print(f"[Config] LoRA rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"[Config] Gradient accumulation={args.grad_accum}")
    print(f"[Config] CFG dropout={args.cfg_dropout}")

    # Dossiers de sortie
    output_dir = PROJECT_ROOT / "outputs" / "lora_merapi_physics"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================
    # Charger SD 1.5
    # ==========================================================
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    print(f"\n[Modèle] Chargement de {model_id}...")

    _kw: dict[str, Any] = {}
    try:
        CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer", local_files_only=True
        )
        _kw["local_files_only"] = True
        print("  (cache local)")
    except Exception:
        print("  (téléchargement HuggingFace)")

    tokenizer = CLIPTokenizer.from_pretrained(
        model_id, subfolder="tokenizer", **_kw
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", dtype=dtype, **_kw
    )
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=dtype, **_kw
    )

    try:
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=dtype, **_kw
        )
    except (OSError, EnvironmentError):
        print("  [UNet] fp32 absent → fp16 + conversion")
        unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            torch_dtype=torch.float16,
            variant="fp16",
            **_kw,
        ).to(dtype=dtype)

    noise_scheduler = DDPMScheduler.from_pretrained(
        model_id, subfolder="scheduler", **_kw
    )

    # Geler VAE et text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # LoRA sur le U-Net (rank 16 par défaut)
    unet = setup_lora_unet(unet, args.lora_rank, args.lora_alpha)

    # Déplacer
    vae.to(device, dtype=dtype)
    text_encoder.to(device, dtype=dtype)
    unet.to(device, dtype=dtype)

    # ==========================================================
    # Dataset
    # ==========================================================
    print("\n[Dataset] Chargement...")
    index_path = PROJECT_ROOT / "data" / "index" / "index.csv"
    processed_base = PROJECT_ROOT / "data" / "processed"

    dataset = MerapiPhysicsDataset(
        index_path=index_path,
        processed_base=processed_base,
        resolution=args.resolution,
        max_images=args.max_images,
        quality_flags=["usable"],
        exclude_dark_night=True,
    )

    if len(dataset) == 0:
        print("[ERREUR] Aucune image.")
        sys.exit(1)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    # ==========================================================
    # Pré-encoder les prompts (cache)
    # ==========================================================
    print("[Prompts] Pré-encodage...")
    _prompt_cache: dict[str, torch.Tensor] = {}

    # Encoder le prompt vide (pour CFG training)
    uncond_emb = encode_prompt("", tokenizer, text_encoder, device)

    unique_prompts = set(s["prompt"] for s in dataset.samples)
    for p in unique_prompts:
        _prompt_cache[p] = encode_prompt(p, tokenizer, text_encoder, device)
    print(f"  {len(unique_prompts)} prompts uniques encodés")

    # ==========================================================
    # Optimiseur
    # ==========================================================
    params_to_optimize = list(
        filter(lambda p: p.requires_grad, unet.parameters())
    )

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    total_steps = args.epochs * math.ceil(len(dataloader) / args.grad_accum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=args.lr * 0.1
    )

    ema = (
        EMAModel(
            [p for p in unet.parameters() if p.requires_grad],
            decay=args.ema_decay,
        )
        if args.use_ema
        else None
    )

    # ==========================================================
    # Génération AVANT fine-tuning
    # ==========================================================
    if not args.skip_gen:
        print("\n[Pré-FT] Génération avant fine-tuning...")
        unet_raw = _unwrap_unet(unet)
        pre_pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet_raw,
            scheduler=noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).to(device)
        pre_pipe.set_progress_bar_config(disable=True)
        if device.type == "mps":
            pre_pipe.enable_attention_slicing()

        generate_physics_samples(
            pre_pipe,
            TEST_SCENARIOS,
            samples_dir,
            prefix="before",
            seed=42,
        )
        del pre_pipe
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print("\n[Pré-FT] Génération ignorée (--skip-gen)")

    # ==========================================================
    # Boucle d'entraînement avec CFG training
    # ==========================================================
    print(f"\n{'=' * 60}")
    print(f"ENTRAÎNEMENT — {args.epochs} epochs, {len(dataset)} images")
    print(f"{'=' * 60}")

    unet.train()
    global_step = 0
    best_loss = float("inf")
    patience_counter = 0
    loss_history = []
    rng_cfg = np.random.default_rng(42)
    num_batches_per_epoch = len(dataloader)

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        epoch_t0 = time.time()

        for batch_idx, batch in enumerate(dataloader):
            batch_t0 = time.time()
            pixel_values = batch["pixel_values"].to(device, dtype=dtype)
            prompts = batch["prompt"]
            bsz = pixel_values.shape[0]

            # 1. VAE encode
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # 2. Noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=device,
            ).long()

            # 3. Noisy latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 4. Encoder les prompts — avec CFG dropout
            encoder_hidden_states = []
            for p in prompts:
                if rng_cfg.random() < args.cfg_dropout:
                    encoder_hidden_states.append(uncond_emb)
                else:
                    if p in _prompt_cache:
                        encoder_hidden_states.append(_prompt_cache[p])
                    else:
                        emb = encode_prompt(
                            p, tokenizer, text_encoder, device
                        )
                        _prompt_cache[p] = emb
                        encoder_hidden_states.append(emb)
            encoder_hidden_states = torch.cat(encoder_hidden_states, dim=0)

            # 5. UNet forward
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            # 6. Loss MSE (epsilon prediction)
            loss = F.mse_loss(
                noise_pred.float(), noise.float(), reduction="mean"
            )
            loss = loss / args.grad_accum

            # 7. Backward
            loss.backward()

            # Sync MPS
            if device.type == "mps":
                torch.mps.synchronize()

            # 8. Gradient accumulation step
            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if ema is not None:
                    ema.update(
                        p for p in unet.parameters() if p.requires_grad
                    )

                global_step += 1

            epoch_loss += loss.item() * args.grad_accum
            num_batches += 1

            batch_dt = time.time() - batch_t0
            print(
                f"\r  Epoch {epoch}/{args.epochs} "
                f"[{batch_idx + 1}/{num_batches_per_epoch}] "
                f"loss={loss.item() * args.grad_accum:.4f} "
                f"({batch_dt:.1f}s/batch)",
                end="",
                flush=True,
            )

        # Fin d'epoch
        avg_loss = epoch_loss / max(num_batches, 1)
        loss_history.append(avg_loss)
        current_lr = scheduler.get_last_lr()[0]
        epoch_dt = time.time() - epoch_t0

        print(
            f"\r  Epoch {epoch:03d}/{args.epochs} | "
            f"Loss={avg_loss:.6f} | LR={current_lr:.2e} | "
            f"Step={global_step} | {epoch_dt:.0f}s"
        )

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            save_path = ckpt_dir / "best_lora"
            unet.save_pretrained(str(save_path))
            print(f"    → Best model saved (loss={best_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(
                    f"\n[Early Stopping] {args.patience} epochs "
                    f"sans amélioration."
                )
                break

        # Checkpoint périodique
        if epoch % args.save_every == 0:
            unet.save_pretrained(
                str(ckpt_dir / f"lora_epoch_{epoch:03d}")
            )

        # Génération périodique (2 premiers scénarios contrastés)
        if epoch % args.sample_every == 0 and not args.skip_gen:
            unet.eval()
            if ema is not None:
                ema.apply_shadow(
                    p for p in unet.parameters() if p.requires_grad
                )

            unet_raw = _unwrap_unet(unet)
            gen_pipe = StableDiffusionPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet_raw,
                scheduler=noise_scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            ).to(device)
            gen_pipe.set_progress_bar_config(disable=True)
            if device.type == "mps":
                gen_pipe.enable_attention_slicing()

            generate_physics_samples(
                gen_pipe,
                TEST_SCENARIOS[:2],  # jour_calme vs nuit_eruption
                samples_dir,
                prefix=f"e{epoch:03d}",
                seed=42,
            )
            del gen_pipe

            if ema is not None:
                ema.restore(
                    p for p in unet.parameters() if p.requires_grad
                )
            unet.train()

    # ==========================================================
    # Sauvegarde finale
    # ==========================================================
    print(f"\n{'=' * 60}")
    print("SAUVEGARDE FINALE")
    print(f"{'=' * 60}")

    if ema is not None:
        ema.apply_shadow(
            p for p in unet.parameters() if p.requires_grad
        )

    final_path = output_dir / "lora_merapi_physics_final"
    unet.save_pretrained(str(final_path))
    print(f"[Sauvegarde] LoRA → {final_path}")

    # Loss CSV
    loss_path = output_dir / "training_loss.csv"
    pd.DataFrame(
        {
            "epoch": list(range(1, len(loss_history) + 1)),
            "loss": loss_history,
        }
    ).to_csv(loss_path, index=False)
    print(f"[Sauvegarde] Loss → {loss_path}")

    # Config
    config_path = output_dir / "training_config.txt"
    with open(config_path, "w") as f:
        f.write(f"model_id: {model_id}\n")
        f.write(f"version: 2 (NL prompts + color augment)\n")
        f.write(f"lora_rank: {args.lora_rank}\n")
        f.write(f"lora_alpha: {args.lora_alpha}\n")
        f.write(f"resolution: {args.resolution}\n")
        f.write(f"epochs_run: {min(epoch, args.epochs)}\n")
        f.write(f"best_loss: {best_loss:.6f}\n")
        f.write(f"lr: {args.lr}\n")
        f.write(f"cfg_dropout: {args.cfg_dropout}\n")
        f.write(f"dataset_size: {len(dataset)}\n")
        f.write(f"prompt_type: rich_natural_language\n")
        f.write(f"color_augmentation: physics_aware\n")

    # ==========================================================
    # Génération APRÈS fine-tuning (tous les scénarios)
    # ==========================================================
    if not args.skip_gen:
        print("\n[Post-FT] Génération finale tous scénarios...")
        unet.eval()
        unet_raw = _unwrap_unet(unet)
        post_pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet_raw,
            scheduler=noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).to(device)
        post_pipe.set_progress_bar_config(disable=True)
        if device.type == "mps":
            post_pipe.enable_attention_slicing()

        generate_physics_samples(
            post_pipe,
            TEST_SCENARIOS,
            samples_dir,
            prefix="final",
            seed=42,
        )
        del post_pipe
    else:
        print("\n[Post-FT] Génération ignorée (--skip-gen)")

    if ema is not None:
        ema.restore(p for p in unet.parameters() if p.requires_grad)

    print(f"\n{'=' * 60}")
    print("ENTRAÎNEMENT TERMINÉ")
    print(f"{'=' * 60}")
    print(f"  Epochs effectués : {min(epoch, args.epochs)}")
    print(f"  Meilleure loss   : {best_loss:.6f}")
    print(f"  Steps totaux     : {global_step}")
    print(f"  Sorties dans     : {output_dir}")
    print(f"\nPour évaluer : python evaluate_lora_physics.py --ssim --fid")
    print(f"Pour Streamlit : USE_TF=0 streamlit run app/streamlit_app.py")


# ============================================================
# 5. CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA SD 1.5 physics v2 — NL prompts + color augment"
    )

    # Dataset
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max-images", type=int, default=500)
    p.add_argument(
        "--min-anomaly",
        type=float,
        default=0.0,
        help="Score d'anomalie minimum (0=toutes)",
    )

    # Entraînement
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5,
                   help="Learning rate (plus élevé pour rank 16)")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument(
        "--cfg-dropout",
        type=float,
        default=0.1,
        help="Probabilité de prompt vide (CFG training)",
    )

    # LoRA (rank 16 par défaut au lieu de 4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)

    # EMA
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.9999)

    # Sauvegarde
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--sample-every", type=int, default=5)

    # Misc — fp16 activé par défaut sur CUDA (plus rapide, moins de VRAM)
    p.add_argument("--fp16", action="store_true", default=None,
                   help="Mixed precision fp16 (auto-activé sur CUDA si non spécifié)")
    p.add_argument("--no-fp16", action="store_true",
                   help="Forcer fp32 même sur CUDA")
    p.add_argument("--skip-gen", action="store_true")

    args = p.parse_args()
    if args.no_ema:
        args.use_ema = False
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
