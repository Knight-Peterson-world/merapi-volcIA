#!/usr/bin/env python3
"""
train_lora_merapi.py — Fine-tuning LoRA de Stable Diffusion 1.5
                        sur le dataset volcanologique Merapi.

Ce script entraîne un adaptateur LoRA (Low-Rank Adaptation) sur le
U-Net de Stable Diffusion 1.5 en utilisant les images prétraitées
(PNG 256×256, niveaux de gris) du réseau VELI/TéléVolc.

Fonctionnalités :
  - Chargement automatique du dataset depuis data/processed/ + index
  - Séparation explicite jour/nuit via conditionnement textuel
  - LoRA avec rank configurable
  - Textual Inversion optionnelle (token <merapi>)
  - EMA pour stabiliser les poids
  - Gradient accumulation pour compenser les petits batch
  - Sauvegarde du LoRA + logs d'entraînement
  - Génération d'images avant/après fine-tuning
  - Compatible Apple Silicon (MPS) et CUDA

Usage :
    cd merapi_anomaly/
    python train_lora_merapi.py
    python train_lora_merapi.py --epochs 100 --lr 5e-6 --resolution 256
    python train_lora_merapi.py --textual-inversion --token "<merapi>"

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

# Empêcher l'import de TensorFlow (conflit protobuf avec transformers)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ---------------------------------------------------------------------------
# Projet root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 1. Dataset Merapi
# ============================================================

class MerapiDataset(Dataset):
    """
    Dataset PyTorch pour les images volcaniques du Merapi.

    Charge les images prétraitées (PNG 256×256, mode L) depuis
    data/processed/, filtre par qualité, et génère des paires
    (image, prompt) pour le fine-tuning texte → image.

    Le conditionnement jour/nuit est explicite dans le prompt :
      - Jour  : "volcanic landscape of Mount Merapi, daytime, ..."
      - Nuit  : "volcanic landscape of Mount Merapi, nighttime, incandescence, ..."
    """

    # Prompts par défaut pour le conditionnement jour/nuit
    PROMPT_DAY = (
        "volcanic landscape of Mount Merapi, daytime surveillance camera, "
        "gray terrain, lava flow, volcanic texture, smoke plume, "
        "Canon EOS 1100D, scientific observation"
    )
    PROMPT_NIGHT = (
        "volcanic landscape of Mount Merapi, nighttime surveillance camera, "
        "incandescence, glowing lava, dark terrain, volcanic eruption, "
        "Canon EOS 1100D, scientific observation"
    )

    def __init__(
        self,
        index_path: Path,
        processed_base: Path,
        resolution: int = 256,
        max_images: int | None = None,
        use_token: str | None = None,
        quality_flags: list[str] | None = None,
        day_hours: tuple[int, int] = (6, 18),
    ) -> None:
        """
        Args:
            index_path: chemin vers index.csv
            processed_base: chemin vers data/processed/
            resolution: résolution cible (256 ou 512)
            max_images: nombre max d'images (None = toutes)
            use_token: token Textual Inversion à injecter (ex. "<merapi>")
            quality_flags: flags qualité à inclure (défaut: ["usable"])
            day_hours: (heure_début_jour, heure_fin_jour) pour séparer jour/nuit
        """
        self.resolution = resolution
        self.use_token = use_token
        self.day_hours = day_hours

        if quality_flags is None:
            quality_flags = ["usable"]

        # Charger l'index
        df = pd.read_csv(index_path, dtype=str, na_values=["", "None", "nan"])
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["month"] = pd.to_numeric(df["month"], errors="coerce")

        # Filtrer par qualité
        df = df[df["quality_flag"].isin(quality_flags)].copy()

        # Résoudre les chemins processed (PNG)
        self.samples: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            y = row.get("year")
            m = row.get("month")
            fn = row.get("filename", "")
            if pd.isna(y) or pd.isna(m) or not fn:
                continue

            # Chemin PNG prétraité
            png_stem = Path(fn).stem + ".png"
            png_path = processed_base / str(int(y)) / f"{int(m):02d}" / png_stem
            if not png_path.exists():
                continue

            # Déterminer jour/nuit
            hour = row.get("hour")
            is_day = True
            if pd.notna(hour):
                h = int(hour)
                is_day = self.day_hours[0] <= h < self.day_hours[1]

            self.samples.append({
                "path": png_path,
                "is_day": is_day,
                "filename": fn,
            })

        # Limiter le nombre d'images
        if max_images is not None and len(self.samples) > max_images:
            # Échantillonner de manière déterministe
            rng = np.random.default_rng(42)
            indices = rng.choice(len(self.samples), max_images, replace=False)
            self.samples = [self.samples[i] for i in sorted(indices)]

        print(f"[Dataset] {len(self.samples)} images chargées "
              f"({sum(s['is_day'] for s in self.samples)} jour, "
              f"{sum(not s['is_day'] for s in self.samples)} nuit)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]

        # Charger l'image PNG (mode L → RGB pour SD)
        img = Image.open(sample["path"]).convert("L")

        # Resize si nécessaire
        if img.size != (self.resolution, self.resolution):
            img = img.resize((self.resolution, self.resolution), Image.LANCZOS)

        # L → RGB (SD attend 3 canaux)
        img_rgb = Image.merge("RGB", [img, img, img])

        # Convertir en tensor [0, 1] → [-1, 1] (format SD)
        arr = np.array(img_rgb, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
        tensor = tensor * 2.0 - 1.0  # normalisation SD [-1, 1]

        # Construire le prompt
        if sample["is_day"]:
            prompt = self.PROMPT_DAY
        else:
            prompt = self.PROMPT_NIGHT

        if self.use_token:
            prompt = f"{self.use_token} {prompt}"

        return {
            "pixel_values": tensor,
            "prompt": prompt,
            "filename": sample["filename"],
        }


# ============================================================
# 2. EMA (Exponential Moving Average)
# ============================================================

class EMAModel:
    """Exponential Moving Average des poids du modèle."""

    def __init__(self, parameters, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {i: p.clone().detach() for i, p in enumerate(parameters)}

    @torch.no_grad()
    def update(self, parameters):
        for i, p in enumerate(parameters):
            self.shadow[i].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_shadow(self, parameters):
        """Applique les poids EMA (pour l'inférence/évaluation)."""
        self.backup = {i: p.clone() for i, p in enumerate(parameters)}
        for i, p in enumerate(parameters):
            p.data.copy_(self.shadow[i])

    def restore(self, parameters):
        """Restaure les poids originaux après évaluation."""
        for i, p in enumerate(parameters):
            p.data.copy_(self.backup[i])


# ============================================================
# 3. Pipeline d'entraînement LoRA
# ============================================================

def detect_device() -> torch.device:
    """Détecte le meilleur device disponible."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_lora_unet(unet, lora_rank: int = 4, lora_alpha: int = 4):
    """
    Configure LoRA sur le U-Net de Stable Diffusion.

    Utilise PEFT (Parameter-Efficient Fine-Tuning) pour injecter
    des adaptateurs LoRA sur les couches d'attention du U-Net.
    """
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",  # cross/self-attention
            "proj_in", "proj_out",                  # projections
        ],
    )

    unet = get_peft_model(unet, lora_config)

    # Comptage des paramètres
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    print(f"[LoRA] Paramètres entraînables : {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    return unet


def setup_textual_inversion(tokenizer, text_encoder, token: str = "<merapi>",
                             init_token: str = "volcano"):
    """
    Configure Textual Inversion : ajoute un nouveau token au vocabulaire
    et l'initialise avec l'embedding d'un mot existant.
    """
    # Ajouter le token
    num_added = tokenizer.add_tokens([token])
    if num_added == 0:
        print(f"[TI] Token '{token}' déjà présent dans le vocabulaire.")
    else:
        print(f"[TI] Token '{token}' ajouté au vocabulaire.")

    # Resize embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialiser avec l'embedding de init_token
    token_id = tokenizer.convert_tokens_to_ids(token)
    init_id = tokenizer.convert_tokens_to_ids(init_token)

    with torch.no_grad():
        text_encoder.get_input_embeddings().weight[token_id] = (
            text_encoder.get_input_embeddings().weight[init_id].clone()
        )

    # Ne rendre entraînable que le nouvel embedding
    text_encoder.get_input_embeddings().weight.requires_grad_(False)
    # Seul le token <merapi> est entraînable
    token_embeds = text_encoder.get_input_embeddings().weight
    token_embeds[token_id].requires_grad_(True)

    print(f"[TI] Token '{token}' initialisé depuis '{init_token}' "
          f"(id={token_id})")

    return token_id


def encode_prompt(prompt: str, tokenizer, text_encoder, device) -> torch.Tensor:
    """Encode un prompt texte en embedding pour le U-Net."""
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        encoder_output = text_encoder(tokens.input_ids.to(device))
    return encoder_output.last_hidden_state


def _unwrap_unet(unet):
    """Extrait le UNet2DConditionModel depuis un wrapper PEFT."""
    if hasattr(unet, "base_model") and hasattr(unet.base_model, "model"):
        return unet.base_model.model
    return unet


def generate_samples(
    pipe,
    prompts: list[str],
    output_dir: Path,
    prefix: str = "sample",
    num_inference_steps: int = 15,
    seed: int = 42,
) -> list[Path]:
    """Génère des images de test et les sauvegarde en PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    generator = torch.Generator(device="cpu").manual_seed(seed)

    for i, prompt in enumerate(prompts):
        print(f"  [Gen] {prefix}_{i:03d} ({num_inference_steps} steps)...", end=" ", flush=True)
        with torch.no_grad():
            result = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                generator=generator,
                guidance_scale=7.5,
            )
        img = result.images[0]
        path = output_dir / f"{prefix}_{i:03d}.png"
        img.save(str(path), format="PNG")
        paths.append(path)
        print("OK")

    return paths


# ============================================================
# 4. Boucle d'entraînement principale
# ============================================================

def train(args: argparse.Namespace) -> None:
    """Boucle d'entraînement LoRA complète."""

    from diffusers import (
        AutoencoderKL,
        DDPMScheduler,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextModel, CLIPTokenizer

    device = detect_device()
    dtype = torch.float32  # MPS ne supporte pas fp16 nativement
    if device.type == "cuda" and args.fp16:
        dtype = torch.float16

    print(f"[Config] Device={device}, dtype={dtype}")
    print(f"[Config] Résolution={args.resolution}, LR={args.lr}, "
          f"Epochs={args.epochs}, Batch={args.batch_size}")
    print(f"[Config] LoRA rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"[Config] Gradient accumulation={args.grad_accum}")

    # ----- Dossiers de sortie -----
    output_dir = PROJECT_ROOT / "outputs" / "lora_merapi"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================
    # Charger le modèle SD 1.5
    # ==========================================================
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    print(f"\n[Modèle] Chargement de {model_id}...")

    # Stratégie de chargement : essayer cache local, puis en ligne.
    # Le cache peut contenir uniquement les poids fp16 (variant="fp16")
    # qu'on convertit ensuite en fp32 pour MPS.
    _load_kwargs: dict[str, Any] = {}
    try:
        CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer",
                                       local_files_only=True)
        _load_kwargs["local_files_only"] = True
        print("  (modèle trouvé en cache local)")
    except Exception:
        print("  (téléchargement depuis HuggingFace Hub)")

    tokenizer = CLIPTokenizer.from_pretrained(
        model_id, subfolder="tokenizer", **_load_kwargs,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", dtype=dtype, **_load_kwargs,
    )
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=dtype, **_load_kwargs,
    )

    # Le UNet fp32 (~3.4 Go) peut ne pas être en cache si seul le
    # fp16 a été téléchargé. On charge fp16 puis convertit en fp32.
    try:
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=dtype, **_load_kwargs,
        )
    except (OSError, EnvironmentError):
        print("  [UNet] Poids fp32 absents — chargement depuis fp16 + conversion")
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=torch.float16,
            variant="fp16", **_load_kwargs,
        )
        unet = unet.to(dtype=dtype)

    noise_scheduler = DDPMScheduler.from_pretrained(
        model_id, subfolder="scheduler", **_load_kwargs,
    )

    # Geler VAE et text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # ----- Textual Inversion optionnelle -----
    ti_token_id = None
    token_str = None
    if args.textual_inversion:
        token_str = args.token
        ti_token_id = setup_textual_inversion(
            tokenizer, text_encoder, token=token_str
        )
        # text_encoder sera partiellement entraînable (juste le token)

    # ----- Configurer LoRA sur le U-Net -----
    unet = setup_lora_unet(unet, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)

    # Déplacer sur le device
    vae.to(device, dtype=dtype)
    text_encoder.to(device, dtype=dtype)
    unet.to(device, dtype=dtype)

    # ==========================================================
    # Dataset et DataLoader
    # ==========================================================
    print("\n[Dataset] Chargement...")
    index_path = PROJECT_ROOT / "data" / "index" / "index.csv"
    processed_base = PROJECT_ROOT / "data" / "processed"

    dataset = MerapiDataset(
        index_path=index_path,
        processed_base=processed_base,
        resolution=args.resolution,
        max_images=args.max_images,
        use_token=token_str,
    )

    if len(dataset) == 0:
        print("[ERREUR] Aucune image dans le dataset. Vérifiez l'index et data/processed/.")
        sys.exit(1)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # MPS ne gère pas bien le multiprocess
        pin_memory=False,
        drop_last=True,
    )

    # ==========================================================
    # Optimiseur et Scheduler
    # ==========================================================
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))

    # Ajouter l'embedding TI si activé
    if ti_token_id is not None:
        ti_params = [text_encoder.get_input_embeddings().weight]
        params_to_optimize = params_to_optimize + ti_params

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    total_steps = args.epochs * math.ceil(len(dataloader) / args.grad_accum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.1,
    )

    # EMA
    ema = EMAModel(
        [p for p in unet.parameters() if p.requires_grad],
        decay=args.ema_decay,
    ) if args.use_ema else None

    # ==========================================================
    # Génération AVANT fine-tuning (optionnelle)
    # ==========================================================
    test_prompts = [
        MerapiDataset.PROMPT_DAY,
        MerapiDataset.PROMPT_NIGHT,
    ]
    if token_str:
        test_prompts = [f"{token_str} {p}" for p in test_prompts]

    if not args.skip_gen:
        print("\n[Pré-FT] Génération d'images AVANT fine-tuning (2 images, 15 steps)...")
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

        generate_samples(pre_pipe, test_prompts, samples_dir / "before_ft",
                         prefix="before", seed=42, num_inference_steps=15)
        del pre_pipe
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print("\n[Pré-FT] Génération ignorée (--skip-gen)")

    # ==========================================================
    # Boucle d'entraînement
    # ==========================================================
    print(f"\n{'='*60}")
    print(f"ENTRAÎNEMENT LoRA — {args.epochs} epochs")
    print(f"{'='*60}")

    unet.train()
    global_step = 0
    best_loss = float("inf")
    patience_counter = 0
    loss_history = []

    # Pré-encoder les prompts (ils sont fixes) pour éviter
    # de recalculer chaque batch.
    _prompt_cache: dict[str, torch.Tensor] = {}

    def _get_prompt_emb(prompt: str) -> torch.Tensor:
        if prompt not in _prompt_cache:
            _prompt_cache[prompt] = encode_prompt(
                prompt, tokenizer, text_encoder, device
            )
        return _prompt_cache[prompt]

    num_batches_per_epoch = len(dataloader)

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        epoch_t0 = time.time()

        for batch_idx, batch in enumerate(dataloader):
            batch_t0 = time.time()
            pixel_values = batch["pixel_values"].to(device, dtype=dtype)
            prompts = batch["prompt"]

            # 1. Encoder les images en latent via le VAE
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # 2. Échantillonner du bruit
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # 3. Échantillonner un timestep aléatoire
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bsz,), device=device,
            ).long()

            # 4. Ajouter le bruit aux latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 5. Encoder le prompt (avec cache)
            encoder_hidden_states = torch.cat(
                [_get_prompt_emb(p) for p in prompts], dim=0,
            )

            # 6. Prédire le bruit
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            # 7. Loss MSE
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            loss = loss / args.grad_accum

            # 8. Backward
            loss.backward()

            # Synchroniser MPS pour un timing fiable
            if device.type == "mps":
                torch.mps.synchronize()

            # 9. Gradient accumulation
            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # EMA update
                if ema is not None:
                    ema.update(p for p in unet.parameters() if p.requires_grad)

                global_step += 1

            epoch_loss += loss.item() * args.grad_accum
            num_batches += 1

            batch_dt = time.time() - batch_t0
            print(f"\r  Epoch {epoch}/{args.epochs} "
                  f"[{batch_idx+1}/{num_batches_per_epoch}] "
                  f"loss={loss.item() * args.grad_accum:.4f} "
                  f"({batch_dt:.1f}s/batch)", end="", flush=True)

        avg_loss = epoch_loss / max(num_batches, 1)
        loss_history.append(avg_loss)
        current_lr = scheduler.get_last_lr()[0]
        epoch_dt = time.time() - epoch_t0

        print(f"\r  Epoch {epoch:03d}/{args.epochs} | "
              f"Loss={avg_loss:.6f} | LR={current_lr:.2e} | "
              f"Step={global_step} | {epoch_dt:.0f}s")

        # ----- Early stopping -----
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0

            # Sauvegarder le meilleur checkpoint
            save_path = ckpt_dir / "best_lora"
            unet.save_pretrained(str(save_path))
            print(f"  → Meilleur modèle sauvegardé ({save_path.name})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[Early Stopping] Pas d'amélioration depuis "
                      f"{args.patience} epochs. Arrêt.")
                break

        # ----- Checkpoint périodique -----
        if epoch % args.save_every == 0:
            save_path = ckpt_dir / f"lora_epoch_{epoch:03d}"
            unet.save_pretrained(str(save_path))

        # ----- Génération périodique -----
        if epoch % args.sample_every == 0 and not args.skip_gen:
            unet.eval()
            if ema is not None:
                ema.apply_shadow(p for p in unet.parameters() if p.requires_grad)

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

            generate_samples(
                gen_pipe, test_prompts[:2],
                samples_dir / f"epoch_{epoch:03d}",
                prefix=f"e{epoch:03d}", seed=42,
                num_inference_steps=15,
            )
            del gen_pipe

            if ema is not None:
                ema.restore(p for p in unet.parameters() if p.requires_grad)
            unet.train()

    # ==========================================================
    # Sauvegarde finale
    # ==========================================================
    print(f"\n{'='*60}")
    print("SAUVEGARDE FINALE")
    print(f"{'='*60}")

    # Appliquer EMA pour la sauvegarde finale
    if ema is not None:
        ema.apply_shadow(p for p in unet.parameters() if p.requires_grad)

    final_path = output_dir / "lora_merapi_final"
    unet.save_pretrained(str(final_path))
    print(f"[Sauvegarde] LoRA final → {final_path}")

    # Sauvegarder le token TI si utilisé
    if ti_token_id is not None:
        ti_path = output_dir / "textual_inversion_merapi"
        ti_path.mkdir(parents=True, exist_ok=True)
        learned_embed = text_encoder.get_input_embeddings().weight[ti_token_id]
        torch.save(
            {token_str: learned_embed.detach().cpu()},
            str(ti_path / "learned_embeds.bin"),
        )
        tokenizer.save_pretrained(str(ti_path))
        print(f"[Sauvegarde] Textual Inversion → {ti_path}")

    # Sauvegarder l'historique des pertes
    loss_path = output_dir / "training_loss.csv"
    pd.DataFrame({
        "epoch": list(range(1, len(loss_history) + 1)),
        "loss": loss_history,
    }).to_csv(loss_path, index=False)
    print(f"[Sauvegarde] Historique loss → {loss_path}")

    # ==========================================================
    # Génération APRÈS fine-tuning
    # ==========================================================
    if not args.skip_gen:
        print("\n[Post-FT] Génération d'images APRÈS fine-tuning...")
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

        generate_samples(post_pipe, test_prompts, samples_dir / "after_ft",
                         prefix="after", seed=42, num_inference_steps=15)
        del post_pipe

        if ema is not None:
            ema.restore(p for p in unet.parameters() if p.requires_grad)
    else:
        print("\n[Post-FT] Génération ignorée (--skip-gen)")

    print(f"\n{'='*60}")
    print("ENTRAÎNEMENT TERMINÉ")
    print(f"{'='*60}")
    print(f"  Epochs effectués : {min(epoch, args.epochs)}")
    print(f"  Meilleure loss   : {best_loss:.6f}")
    print(f"  Steps totaux     : {global_step}")
    print(f"  Sorties dans     : {output_dir}")
    print(f"\nPour évaluer : python evaluate_lora_merapi.py")


# ============================================================
# 5. CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tuning LoRA de Stable Diffusion 1.5 sur le dataset Merapi"
    )

    # Dataset
    parser.add_argument("--resolution", type=int, default=256,
                        help="Résolution des images (256 ou 512)")
    parser.add_argument("--max-images", type=int, default=500,
                        help="Nombre max d'images (défaut: 500)")

    # Entraînement
    parser.add_argument("--epochs", type=int, default=100,
                        help="Nombre d'epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Taille du batch")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay AdamW")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (epochs)")

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=4,
                        help="LoRA rank (r)")
    parser.add_argument("--lora-alpha", type=int, default=4,
                        help="LoRA alpha")

    # Textual Inversion
    parser.add_argument("--textual-inversion", action="store_true",
                        help="Activer Textual Inversion")
    parser.add_argument("--token", type=str, default="<merapi>",
                        help="Token pour Textual Inversion")

    # EMA
    parser.add_argument("--use-ema", action="store_true", default=True,
                        help="Utiliser EMA (activé par défaut)")
    parser.add_argument("--no-ema", action="store_true",
                        help="Désactiver EMA")
    parser.add_argument("--ema-decay", type=float, default=0.9999,
                        help="EMA decay factor")

    # Sauvegarde / Sampling
    parser.add_argument("--save-every", type=int, default=25,
                        help="Sauvegarder un checkpoint tous les N epochs")
    parser.add_argument("--sample-every", type=int, default=10,
                        help="Générer des échantillons tous les N epochs")

    # Device
    parser.add_argument("--fp16", action="store_true",
                        help="Utiliser fp16 (CUDA uniquement)")

    # Génération
    parser.add_argument("--skip-gen", action="store_true",
                        help="Sauter la génération avant/après FT (accélère le test)")

    args = parser.parse_args()

    if args.no_ema:
        args.use_ema = False

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
