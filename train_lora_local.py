#!/usr/bin/env python3
"""
train_lora_local.py — Entraînement LoRA physics v2, optimisé Mac M1 / CPU.

Stratégie mémoire (8 GB RAM unifiée) :
  Phase 1 — Pré-encodage : charge VAE + text_encoder, encode TOUT
            (latents + prompts), sauvegarde en tensors, puis LIBÈRE
            VAE et text_encoder de la mémoire.
  Phase 2 — Entraînement : seul le UNet+LoRA reste en mémoire (~3.5 GB).
            Gradient checkpointing activé. Batch size 1 + accumulation.
  Phase 3 — Génération : recharge le pipeline complet pour 2 scénarios.

Résultat identique au script Colab, mais faisable en local.

Usage :
    # Mac M1 — entraînement complet (quelques heures)
    USE_TF=0 python train_lora_local.py

    # Rapide (test 5 epochs)
    USE_TF=0 python train_lora_local.py --epochs 5 --max-images 50

    # CPU pur (plus lent, si MPS pose problème)
    USE_TF=0 python train_lora_local.py --device cpu

Prérequis :
    pip install torch diffusers transformers accelerate peft pillow pandas
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

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

from src.physics_prompts import (
    PHYSICS_NEGATIVE_PROMPT,
    build_rich_prompt,
    build_prompt_from_metadata,
    apply_physics_color,
    camera_from_filename,
)


# ============================================================
# 1. Dataset pré-encodé (latents + prompts sur disque/RAM)
# ============================================================

class PreEncodedDataset(Dataset):
    """Dataset qui sert des latents et embeddings pré-encodés.

    Après la phase 1, VAE et text_encoder sont supprimés.
    Ce dataset ne charge que des tensors légers.
    """

    def __init__(self, latents: list[torch.Tensor], embeddings: list[torch.Tensor]):
        assert len(latents) == len(embeddings)
        self.latents = latents
        self.embeddings = embeddings

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "latents": self.latents[idx],
            "embeddings": self.embeddings[idx],
        }


# ============================================================
# 2. Phase 1 : Pré-encodage
# ============================================================

def collect_image_data(
    index_path: Path,
    processed_base: Path,
    resolution: int,
    max_images: int | None,
) -> list[dict[str, Any]]:
    """Charge les métadonnées et chemins (sans charger en mémoire GPU)."""
    df = pd.read_csv(index_path, dtype=str, na_values=["", "None", "nan"])
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df["anomaly_score"] = pd.to_numeric(
        df["anomaly_score"], errors="coerce"
    ).fillna(0.0)

    df = df[df["quality_flag"] == "usable"].copy()

    # Exclure nuits sans incandescence
    is_night = ~df["hour"].between(6, 17)
    has_glow = df["anomaly_score"] > 0.05
    df = df[~(is_night & ~has_glow)].copy()

    samples: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        y, m, fn = row.get("year"), row.get("month"), row.get("filename", "")
        if pd.isna(y) or pd.isna(m) or not fn:
            continue
        png_path = processed_base / str(int(y)) / f"{int(m):02d}" / (Path(fn).stem + ".png")
        if not png_path.exists():
            continue

        prompt = build_prompt_from_metadata(row.to_dict(), template_index=0)
        hour_val = pd.to_numeric(row.get("hour", 12), errors="coerce")
        score_val = pd.to_numeric(row.get("anomaly_score", 0.0), errors="coerce")
        samples.append({
            "path": png_path,
            "prompt": prompt,
            "hour": int(hour_val) if pd.notna(hour_val) else 12,
            "anomaly_score": float(score_val) if pd.notna(score_val) else 0.0,
            "camera": camera_from_filename(fn),
        })

    if max_images is not None and len(samples) > max_images:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(samples), max_images, replace=False)
        samples = [samples[i] for i in sorted(idx)]

    n_day = sum(1 for s in samples if 6 <= s["hour"] < 18)
    n_active = sum(1 for s in samples if s["anomaly_score"] > 0.15)
    cams = {}
    for s in samples:
        cams[s["camera"]] = cams.get(s["camera"], 0) + 1
    print(f"[Dataset] {len(samples)} images")
    print(f"  Jour/Nuit: {n_day}/{len(samples) - n_day}")
    print(f"  Actives: {n_active} | Caméras: {cams}")

    return samples


def pre_encode_all(
    samples: list[dict[str, Any]],
    resolution: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Phase 1 : encode toutes les images (VAE) et prompts (CLIP).

    Charge VAE et text_encoder temporairement, encode tout,
    puis les supprime pour libérer la mémoire.

    Returns:
        (latents_list, embeddings_list, uncond_embedding)
    """
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer

    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    dtype = torch.float32

    _kw: dict[str, Any] = {}
    try:
        CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", local_files_only=True)
        _kw["local_files_only"] = True
        print("  (cache local)")
    except Exception:
        print("  (téléchargement HF)")

    # ── Charger text_encoder, encoder les prompts, libérer ──
    print("[Phase 1a] Encodage des prompts CLIP...")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", **_kw)
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype, **_kw
    ).to(device).eval()

    def _encode_prompt(prompt: str) -> torch.Tensor:
        tokens = tokenizer(
            prompt, padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            emb = text_encoder(tokens.input_ids.to(device)).last_hidden_state
        return emb.squeeze(0).cpu()  # [77, 768] sur CPU

    # Cache de prompts uniques
    unique_prompts = list(set(s["prompt"] for s in samples))
    prompt_emb_map: dict[str, torch.Tensor] = {}
    for i, p in enumerate(unique_prompts):
        prompt_emb_map[p] = _encode_prompt(p)
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(unique_prompts)} prompts...")
    print(f"  {len(unique_prompts)} prompts uniques encodés")

    uncond_emb = _encode_prompt("").cpu()

    embeddings_list = [prompt_emb_map[s["prompt"]] for s in samples]

    # Libérer text_encoder + tokenizer
    del text_encoder, tokenizer, prompt_emb_map
    _flush_memory(device)
    print("  text_encoder libéré ✓")

    # ── Charger VAE, encoder les images, libérer ──
    print("[Phase 1b] Encodage VAE des images...")
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=dtype, **_kw
    ).to(device).eval()
    scaling_factor = vae.config.scaling_factor

    latents_list: list[torch.Tensor] = []
    for i, sample in enumerate(samples):
        img = Image.open(sample["path"]).convert("L")
        if img.size != (resolution, resolution):
            img = img.resize((resolution, resolution), Image.LANCZOS)

        img_rgb = apply_physics_color(img, sample["hour"], sample["anomaly_score"])
        arr = np.array(img_rgb, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
        tensor = tensor.to(device, dtype=dtype)

        with torch.no_grad():
            latent = vae.encode(tensor).latent_dist.sample() * scaling_factor

        latents_list.append(latent.squeeze(0).cpu())  # [4, H/8, W/8] sur CPU

        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(samples)} images...")

    print(f"  {len(latents_list)} latents encodés ({latents_list[0].shape})")

    # Libérer VAE
    del vae
    _flush_memory(device)
    print("  VAE libéré ✓")

    return latents_list, embeddings_list, uncond_emb


def _flush_memory(device: torch.device) -> None:
    """Libère agressivement la mémoire."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


# ============================================================
# 3. Phase 2 : Entraînement (UNet + LoRA seuls en mémoire)
# ============================================================

def setup_lora_unet(unet, lora_rank: int = 8, lora_alpha: int = 8):
    """LoRA sur le UNet. Rank 8 par défaut en local (compromis mémoire/capacité)."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
        ],
    )
    unet = get_peft_model(unet, config)

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    print(f"[LoRA] r={lora_rank}, alpha={lora_alpha}")
    print(f"[LoRA] {trainable:,} / {total:,} params ({100 * trainable / total:.2f}%)")
    return unet


class EMAModel:
    """EMA des paramètres entraînables."""

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
    if hasattr(unet, "base_model") and hasattr(unet.base_model, "model"):
        return unet.base_model.model
    return unet


# Scénarios de test (2 contrastés pour la génération locale)
TEST_SCENARIOS_LOCAL = [
    {
        "name": "jour_calme_clair",
        "params": {
            "camera": "Suki", "time_of_day": "midday",
            "brightness": "daylight", "lava_intensity": "none",
            "slope": "30deg_south", "weather": "clear",
        },
    },
    {
        "name": "nuit_eruption_intense",
        "params": {
            "camera": "Kalor", "time_of_day": "night",
            "brightness": "incandescent_glow", "lava_intensity": "very_high",
            "slope": "25deg_west", "weather": "clear_night",
            "viscosity": "low", "temperature": "extreme",
            "eruption_type": "effusive", "plume": "medium",
        },
    },
]


def train(args: argparse.Namespace) -> None:
    from diffusers import DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel

    # ── Device ──
    if args.device:
        device = torch.device(args.device)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dtype = torch.float32  # Toujours fp32 sur Mac/CPU

    print(f"\n{'=' * 60}")
    print("ENTRAÎNEMENT LoRA LOCAL — Optimisé Mac M1 / CPU")
    print(f"{'=' * 60}")
    print(f"[Config] Device={device}, dtype=fp32")
    print(f"[Config] Résolution={args.resolution}, Epochs={args.epochs}")
    print(f"[Config] Batch=1 × grad_accum={args.grad_accum} = effectif {args.grad_accum}")
    print(f"[Config] LoRA rank={args.lora_rank}, LR={args.lr}")
    print(f"[Config] Max images={args.max_images}")
    print(f"[Config] EMA={'oui' if args.use_ema else 'non'}")

    output_dir = PROJECT_ROOT / "outputs" / "lora_merapi_physics"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    index_path = PROJECT_ROOT / "data" / "index" / "index.csv"
    processed_base = PROJECT_ROOT / "data" / "processed"

    # ==================================================================
    # PHASE 1 : Pré-encodage complet (VAE + CLIP) → puis libération
    # ==================================================================
    print(f"\n{'─' * 60}")
    print("PHASE 1 — Pré-encodage (VAE + CLIP)")
    print(f"{'─' * 60}")

    samples = collect_image_data(
        index_path, processed_base, args.resolution, args.max_images,
    )
    if not samples:
        print("[ERREUR] Aucune image trouvée.")
        sys.exit(1)

    latents_list, embeddings_list, uncond_emb = pre_encode_all(
        samples, args.resolution, device,
    )

    # Déplacer uncond_emb sur device
    uncond_emb = uncond_emb.to(device)

    dataset = PreEncodedDataset(latents_list, embeddings_list)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True,
    )
    print(f"[DataLoader] {len(dataset)} samples, {len(dataloader)} batches/epoch")

    # ==================================================================
    # PHASE 2 : Charger UNet seul + LoRA + entraîner
    # ==================================================================
    print(f"\n{'─' * 60}")
    print("PHASE 2 — Entraînement (UNet + LoRA seuls en mémoire)")
    print(f"{'─' * 60}")

    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    _kw: dict[str, Any] = {}
    try:
        from transformers import CLIPTokenizer
        CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", local_files_only=True)
        _kw["local_files_only"] = True
    except Exception:
        pass

    print("[Modèle] Chargement UNet...")
    try:
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=dtype, **_kw,
        )
    except (OSError, EnvironmentError):
        print("  fp32 absent → fp16 + conversion")
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=torch.float16,
            variant="fp16", **_kw,
        ).to(dtype=dtype)

    noise_scheduler = DDPMScheduler.from_pretrained(
        model_id, subfolder="scheduler", **_kw,
    )

    # LoRA
    unet = setup_lora_unet(unet, args.lora_rank, args.lora_alpha)

    # Gradient checkpointing — CRITIQUE pour la mémoire
    unet_raw = _unwrap_unet(unet)
    if hasattr(unet_raw, "enable_gradient_checkpointing"):
        unet_raw.enable_gradient_checkpointing()
        print("[Memory] Gradient checkpointing activé ✓")

    unet.to(device, dtype=dtype)
    _flush_memory(device)

    # Mémoire estimée
    unet_mb = sum(p.nelement() * p.element_size() for p in unet.parameters()) / 1e6
    print(f"[Memory] UNet en mémoire: ~{unet_mb:.0f} MB")

    # Optimiseur
    params_to_optimize = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params_to_optimize, lr=args.lr,
        weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8,
    )

    total_steps = args.epochs * math.ceil(len(dataloader) / args.grad_accum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=args.lr * 0.1,
    )

    ema = (
        EMAModel(
            [p for p in unet.parameters() if p.requires_grad],
            decay=args.ema_decay,
        )
        if args.use_ema
        else None
    )

    # ── Boucle d'entraînement ──
    print(f"\n{'=' * 60}")
    print(f"ENTRAÎNEMENT — {args.epochs} epochs, {len(dataset)} images")
    print(f"{'=' * 60}")
    print(f"Estimation: ~{len(dataloader) * 0.8:.0f}s/epoch sur MPS M1\n")

    unet.train()
    global_step = 0
    best_loss = float("inf")
    patience_counter = 0
    loss_history = []
    rng_cfg = np.random.default_rng(42)
    num_batches = len(dataloader)
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        n_batch = 0
        epoch_t0 = time.time()

        for batch_idx, batch in enumerate(dataloader):
            batch_t0 = time.time()

            latents = batch["latents"].to(device, dtype=dtype)
            emb = batch["embeddings"].to(device, dtype=dtype)
            bsz = latents.shape[0]

            # CFG dropout : remplacer par uncond 10% du temps
            if rng_cfg.random() < args.cfg_dropout:
                emb = uncond_emb.unsqueeze(0).expand(bsz, -1, -1)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bsz,), device=device,
            ).long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            noise_pred = unet(
                noisy_latents, timesteps,
                encoder_hidden_states=emb,
            ).sample

            loss = F.mse_loss(noise_pred.float(), noise.float()) / args.grad_accum
            loss.backward()

            # Sync MPS
            if device.type == "mps":
                torch.mps.synchronize()

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(p for p in unet.parameters() if p.requires_grad)
                global_step += 1

            epoch_loss += loss.item() * args.grad_accum
            n_batch += 1

            batch_dt = time.time() - batch_t0
            print(
                f"\r  Epoch {epoch}/{args.epochs} "
                f"[{batch_idx + 1}/{num_batches}] "
                f"loss={loss.item() * args.grad_accum:.4f} "
                f"({batch_dt:.1f}s)",
                end="", flush=True,
            )

        # Fin d'epoch
        avg_loss = epoch_loss / max(n_batch, 1)
        loss_history.append(avg_loss)
        lr = scheduler.get_last_lr()[0]
        epoch_dt = time.time() - epoch_t0
        elapsed_total = (time.time() - train_start) / 60

        print(
            f"\r  Epoch {epoch:03d}/{args.epochs} | "
            f"Loss={avg_loss:.6f} | LR={lr:.2e} | "
            f"Step={global_step} | {epoch_dt:.0f}s | "
            f"Total={elapsed_total:.1f}min"
        )

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            unet.save_pretrained(str(ckpt_dir / "best_lora"))
            print(f"    ★ Best (loss={best_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[Early Stopping] {args.patience} epochs sans amélioration.")
                break

        if epoch % args.save_every == 0:
            unet.save_pretrained(str(ckpt_dir / f"lora_epoch_{epoch:03d}"))

    # ==================================================================
    # Sauvegarde finale
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("SAUVEGARDE FINALE")
    print(f"{'=' * 60}")

    if ema is not None:
        ema.apply_shadow(p for p in unet.parameters() if p.requires_grad)

    final_path = output_dir / "lora_merapi_physics_final"
    unet.save_pretrained(str(final_path))
    print(f"[Sauvegarde] LoRA → {final_path}")

    pd.DataFrame({
        "epoch": list(range(1, len(loss_history) + 1)),
        "loss": loss_history,
    }).to_csv(output_dir / "training_loss.csv", index=False)

    config_path = output_dir / "training_config.txt"
    with open(config_path, "w") as f:
        f.write(f"model_id: {model_id}\n")
        f.write(f"version: 2-local (NL prompts + color + pre-encoded)\n")
        f.write(f"device: {device}\n")
        f.write(f"lora_rank: {args.lora_rank}\n")
        f.write(f"lora_alpha: {args.lora_alpha}\n")
        f.write(f"resolution: {args.resolution}\n")
        f.write(f"epochs_run: {min(epoch, args.epochs)}\n")
        f.write(f"best_loss: {best_loss:.6f}\n")
        f.write(f"lr: {args.lr}\n")
        f.write(f"cfg_dropout: {args.cfg_dropout}\n")
        f.write(f"dataset_size: {len(dataset)}\n")
        f.write(f"gradient_checkpointing: True\n")
        f.write(f"pre_encoded: True\n")
        f.write(f"training_time_min: {(time.time() - train_start) / 60:.1f}\n")

    total_min = (time.time() - train_start) / 60
    print(f"\n  Epochs effectués : {min(epoch, args.epochs)}")
    print(f"  Meilleure loss   : {best_loss:.6f}")
    print(f"  Temps total      : {total_min:.1f} min")

    # ==================================================================
    # PHASE 3 : Génération finale (recharge pipeline complet)
    # ==================================================================
    if not args.skip_gen:
        print(f"\n{'─' * 60}")
        print("PHASE 3 — Génération de test (2 scénarios)")
        print(f"{'─' * 60}")

        # Libérer les gradients et optimizer
        del optimizer, scheduler, dataset, dataloader, latents_list, embeddings_list
        if ema is not None:
            ema.restore(p for p in unet.parameters() if p.requires_grad)
            del ema
        optimizer = None
        _flush_memory(device)

        unet.eval()
        unet_raw = _unwrap_unet(unet)

        from diffusers import AutoencoderKL
        from transformers import CLIPTextModel, CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", **_kw)
        text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=dtype, **_kw
        )
        vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=dtype, **_kw
        )

        pipe = StableDiffusionPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
            unet=unet_raw, scheduler=noise_scheduler,
            safety_checker=None, feature_extractor=None,
            requires_safety_checker=False,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)
        if device.type == "mps":
            pipe.enable_attention_slicing()

        for sc in TEST_SCENARIOS_LOCAL:
            name = sc["name"]
            prompt = build_rich_prompt(**sc["params"], template_index=0)
            print(f"  [{name}] {prompt[:80]}...")
            gen = torch.Generator(device="cpu").manual_seed(42)
            with torch.no_grad():
                img = pipe(
                    prompt, negative_prompt=PHYSICS_NEGATIVE_PROMPT,
                    num_inference_steps=25, guidance_scale=10.0,
                    generator=gen,
                ).images[0]
            path = samples_dir / f"final_{name}.png"
            img.save(str(path), format="PNG")
            print(f"    → {path.name}")

        del pipe
        _flush_memory(device)
    else:
        if ema is not None:
            ema.restore(p for p in unet.parameters() if p.requires_grad)

    print(f"\n{'=' * 60}")
    print("TERMINÉ ✓")
    print(f"{'=' * 60}")
    print(f"  Sorties      : {output_dir}")
    print(f"  LoRA final   : {final_path}")
    print(f"  Pour évaluer : USE_TF=0 python evaluate_lora_physics.py --ssim")
    print(f"  Pour Streamlit: USE_TF=0 streamlit run app/streamlit_app.py")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA SD 1.5 physics v2 — Optimisé Mac M1 / CPU"
    )
    p.add_argument("--resolution", type=int, default=128,
                   help="Résolution images (128=rapide, 192=meilleur, 256=lent)")
    p.add_argument("--max-images", type=int, default=300,
                   help="Nombre max d'images (300 OK pour 8GB)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--grad-accum", type=int, default=8,
                   help="Accumulation gradient (compense batch_size=1)")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--cfg-dropout", type=float, default=0.1)
    p.add_argument("--lora-rank", type=int, default=8,
                   help="Rank LoRA (8=bon compromis mémoire/capacité)")
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--skip-gen", action="store_true",
                   help="Sauter la génération finale (économise mémoire)")
    p.add_argument("--device", type=str, default=None,
                   help="Forcer le device (mps, cpu). Auto-détecté sinon.")
    args = p.parse_args()
    if args.no_ema:
        args.use_ema = False
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
