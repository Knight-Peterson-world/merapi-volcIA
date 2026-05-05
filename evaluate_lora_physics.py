#!/usr/bin/env python3
"""
evaluate_lora_physics.py — Évaluation du LoRA physics v2 (NL prompts).

Génère des images pour 10+ scénarios physiques contrastés et calcule
des métriques quantitatives (SSIM, FID) par catégorie.

Usage :
    python evaluate_lora_physics.py --ssim --fid
    python evaluate_lora_physics.py --ssim --num-images 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.physics_prompts import (
    PHYSICS_NEGATIVE_PROMPT,
    build_rich_prompt,
)


# ── Scénarios d'évaluation (contrastés) ──────────────────────

EVAL_SCENARIOS = [
    # ── Jour calme ──
    {
        "name": "jour_calme_suki_clair",
        "category": "day_calm",
        "params": {
            "camera": "Suki", "time_of_day": "midday",
            "brightness": "daylight", "lava_intensity": "none",
            "slope": "30deg_south", "weather": "clear",
        },
    },
    {
        "name": "matin_brume_kali",
        "category": "day_calm",
        "params": {
            "camera": "Kali", "time_of_day": "early_morning",
            "brightness": "daylight", "lava_intensity": "none",
            "slope": "35deg_east", "weather": "hazy",
        },
    },
    {
        "name": "aprem_couvert_kalor",
        "category": "day_calm",
        "params": {
            "camera": "Kalor", "time_of_day": "afternoon",
            "brightness": "daylight", "lava_intensity": "none",
            "slope": "25deg_west", "weather": "overcast",
        },
    },
    # ── Jour avec activité ──
    {
        "name": "jour_actif_moderate",
        "category": "day_active",
        "params": {
            "camera": "Suki", "time_of_day": "midday",
            "brightness": "bright_with_incandescence",
            "lava_intensity": "moderate",
            "slope": "30deg_south", "weather": "clear",
            "viscosity": "medium", "temperature": "moderate",
            "eruption_type": "effusive", "plume": "low",
        },
    },
    # ── Crépuscule ──
    {
        "name": "crepuscule_moderate_suki",
        "category": "dusk",
        "params": {
            "camera": "Suki", "time_of_day": "dusk",
            "brightness": "dim_glow", "lava_intensity": "moderate",
            "slope": "30deg_south", "weather": "clear",
            "viscosity": "medium", "temperature": "moderate",
        },
    },
    {
        "name": "crepuscule_calme_kalor",
        "category": "dusk",
        "params": {
            "camera": "Kalor", "time_of_day": "dusk",
            "brightness": "daylight", "lava_intensity": "none",
            "slope": "25deg_west", "weather": "overcast",
        },
    },
    # ── Nuit active ──
    {
        "name": "nuit_eruption_intense_kalor",
        "category": "night_active",
        "params": {
            "camera": "Kalor", "time_of_day": "night",
            "brightness": "incandescent_glow",
            "lava_intensity": "very_high",
            "slope": "25deg_west", "weather": "clear_night",
            "viscosity": "low", "temperature": "extreme",
            "eruption_type": "effusive", "plume": "medium",
        },
    },
    {
        "name": "nuit_high_suki",
        "category": "night_active",
        "params": {
            "camera": "Suki", "time_of_day": "night",
            "brightness": "incandescent_glow",
            "lava_intensity": "high",
            "slope": "30deg_south", "weather": "clear_night",
            "viscosity": "low", "temperature": "high",
            "eruption_type": "explosive", "plume": "high",
        },
    },
    # ── Nuit faible ──
    {
        "name": "nuit_faible_couvert",
        "category": "night_calm",
        "params": {
            "camera": "Kalor", "time_of_day": "night",
            "brightness": "dim_glow", "lava_intensity": "low",
            "slope": "25deg_west", "weather": "overcast",
            "viscosity": "high", "temperature": "low",
        },
    },
    {
        "name": "nuit_aucune_activite",
        "category": "night_calm",
        "params": {
            "camera": "Suki", "time_of_day": "night",
            "brightness": "dark", "lava_intensity": "none",
            "slope": "30deg_south", "weather": "clear_night",
        },
    },
    # ── Extrapolation (phréatique — pas dans le dataset) ──
    {
        "name": "phreatic_jour",
        "category": "extrapolation",
        "params": {
            "camera": "Suki", "time_of_day": "midday",
            "brightness": "daylight", "lava_intensity": "none",
            "slope": "30deg_south", "weather": "hazy",
            "eruption_type": "phreatic", "plume": "high",
        },
    },
    {
        "name": "nuit_extreme_viscous",
        "category": "extrapolation",
        "params": {
            "camera": "Kalor", "time_of_day": "night",
            "brightness": "incandescent_glow",
            "lava_intensity": "very_high",
            "slope": "25deg_west", "weather": "clear_night",
            "viscosity": "high", "temperature": "extreme",
            "eruption_type": "effusive", "plume": "high",
        },
    },
]


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_lora_path() -> Path | None:
    """Auto-détecte le LoRA physics."""
    candidates = [
        PROJECT_ROOT / "outputs" / "lora_merapi_physics" / "lora_merapi_physics_final",
        PROJECT_ROOT / "outputs" / "lora_merapi_physics_results" / "lora_merapi_physics_final",
        PROJECT_ROOT / "outputs" / "lora_merapi_results" / "lora_merapi_final",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_pipeline(lora_path: Path | None = None):
    """Charge le pipeline SD 1.5 avec LoRA optionnel."""
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers import AutoencoderKL, DDPMScheduler

    device = detect_device()
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"[Eval] Device={device}, dtype={dtype}")
    print(f"[Eval] Chargement {model_id}...")

    _kw = {}
    try:
        CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer", local_files_only=True
        )
        _kw["local_files_only"] = True
    except Exception:
        pass

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", **_kw)
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype, **_kw
    )
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=dtype, **_kw
    )

    try:
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=dtype, **_kw
        )
    except (OSError, EnvironmentError):
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=torch.float16,
            variant="fp16", **_kw
        ).to(dtype=dtype)

    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler", **_kw)

    # Charger LoRA si disponible
    if lora_path and lora_path.exists():
        from peft import PeftModel
        print(f"[Eval] Chargement LoRA : {lora_path}")
        unet = PeftModel.from_pretrained(unet, str(lora_path))
        unet = unet.base_model.model
        print("[Eval] LoRA chargé ✓")
    else:
        print("[Eval] Pas de LoRA — SD 1.5 vanilla")

    pipe = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        unet=unet, scheduler=scheduler,
        safety_checker=None, feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    if device.type == "mps":
        pipe.enable_attention_slicing()

    return pipe, device


def generate_scenario_images(
    pipe,
    scenarios: list[dict],
    output_dir: Path,
    num_per_scenario: int = 3,
    guidance_scale: float = 10.0,
    num_inference_steps: int = 30,
) -> dict[str, list[Image.Image]]:
    """Génère N images par scénario."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[Image.Image]] = {}

    for scenario in scenarios:
        name = scenario["name"]
        params = scenario["params"]
        prompt = build_rich_prompt(**params, template_index=0)

        print(f"\n  [{name}] {prompt[:80]}...")
        images = []

        for i in range(num_per_scenario):
            gen = torch.Generator(device="cpu").manual_seed(42 + i * 1000)
            with torch.no_grad():
                result = pipe(
                    prompt,
                    negative_prompt=PHYSICS_NEGATIVE_PROMPT,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=gen,
                )
            img = result.images[0]
            img_path = output_dir / f"{name}_{i:02d}.png"
            img.save(str(img_path), format="PNG")
            images.append(img)
            print(f"    → {img_path.name}")

        results[name] = images

    return results


def compute_ssim_between_scenarios(
    results: dict[str, list[Image.Image]],
) -> dict[str, float]:
    """Calcule le SSIM moyen entre toutes les paires de scénarios.

    Un SSIM élevé entre scénarios différents = mauvais (images identiques).
    Un SSIM bas = bon (images variées selon les conditions).
    """
    from skimage.metrics import structural_similarity as ssim

    scenario_names = list(results.keys())
    ssim_scores: dict[str, float] = {}

    for i, name_a in enumerate(scenario_names):
        for j, name_b in enumerate(scenario_names):
            if i >= j:
                continue
            imgs_a = results[name_a]
            imgs_b = results[name_b]

            scores = []
            for img_a in imgs_a:
                for img_b in imgs_b:
                    arr_a = np.array(img_a.resize((256, 256)).convert("RGB"), dtype=np.float32)
                    arr_b = np.array(img_b.resize((256, 256)).convert("RGB"), dtype=np.float32)
                    s = ssim(arr_a, arr_b, channel_axis=2, data_range=255.0)
                    scores.append(s)
            avg = np.mean(scores)
            ssim_scores[f"{name_a} ↔ {name_b}"] = avg

    return ssim_scores


def compute_intra_ssim(
    results: dict[str, list[Image.Image]],
) -> dict[str, float]:
    """SSIM intra-scénario (cohérence — devrait être élevé)."""
    from skimage.metrics import structural_similarity as ssim

    intra: dict[str, float] = {}
    for name, imgs in results.items():
        if len(imgs) < 2:
            continue
        scores = []
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                arr_a = np.array(imgs[i].resize((256, 256)).convert("RGB"), dtype=np.float32)
                arr_b = np.array(imgs[j].resize((256, 256)).convert("RGB"), dtype=np.float32)
                scores.append(ssim(arr_a, arr_b, channel_axis=2, data_range=255.0))
        intra[name] = np.mean(scores) if scores else 0.0
    return intra


def create_comparison_grid(
    results: dict[str, list[Image.Image]],
    output_path: Path,
    title: str = "Physics Conditioning Evaluation",
) -> None:
    """Crée une grille de comparaison visuelle."""
    scenarios = list(results.keys())
    n_scenarios = len(scenarios)
    n_per = min(3, max(len(imgs) for imgs in results.values()))

    cell_size = 256
    padding = 4
    label_h = 20

    grid_w = n_per * (cell_size + padding) + padding
    grid_h = n_scenarios * (cell_size + padding + label_h) + padding + 30

    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(grid)
    except ImportError:
        draw = None

    y = 30
    for scenario_name in scenarios:
        imgs = results[scenario_name]
        if draw:
            draw.text((padding, y - 18), scenario_name, fill=(0, 0, 0))
        x = padding
        for i in range(min(n_per, len(imgs))):
            thumb = imgs[i].resize((cell_size, cell_size), Image.LANCZOS)
            grid.paste(thumb, (x, y))
            x += cell_size + padding
        y += cell_size + padding + label_h

    grid.save(str(output_path), format="PNG")
    print(f"\n[Grid] Sauvegardée : {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Évaluation LoRA physics v2"
    )
    parser.add_argument("--num-images", type=int, default=3,
                        help="Images par scénario")
    parser.add_argument("--guidance-scale", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--ssim", action="store_true",
                        help="Calculer SSIM inter/intra scénarios")
    parser.add_argument("--fid", action="store_true",
                        help="Calculer FID (nécessite pytorch-fid)")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Chemin LoRA (auto-détection si omis)")
    args = parser.parse_args()

    # Trouver LoRA
    lora_path = Path(args.lora_path) if args.lora_path else find_lora_path()
    if lora_path:
        print(f"[Eval] LoRA : {lora_path}")
    else:
        print("[Eval] Pas de LoRA trouvé — évaluation vanilla SD 1.5")

    # Charger pipeline
    pipe, device = load_pipeline(lora_path)

    # Output
    eval_dir = PROJECT_ROOT / "outputs" / "lora_merapi_physics" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Générer
    print(f"\n[Eval] Génération de {len(EVAL_SCENARIOS)} scénarios "
          f"× {args.num_images} images...")
    results = generate_scenario_images(
        pipe, EVAL_SCENARIOS, eval_dir,
        num_per_scenario=args.num_images,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
    )

    # Grille visuelle
    create_comparison_grid(
        results,
        eval_dir / "comparison_grid.png",
    )

    # SSIM
    if args.ssim:
        print("\n[SSIM] Calcul inter-scénarios...")
        try:
            inter_ssim = compute_ssim_between_scenarios(results)
            intra_ssim = compute_intra_ssim(results)

            print("\n  SSIM inter-scénarios (bas = bonne différenciation) :")
            for pair, score in sorted(inter_ssim.items(), key=lambda x: -x[1]):
                print(f"    {score:.4f}  {pair}")

            avg_inter = np.mean(list(inter_ssim.values()))
            print(f"\n  Moyenne inter-scénarios : {avg_inter:.4f}")
            print(f"  (< 0.5 = excellente différenciation, "
                  f"> 0.7 = images trop similaires)")

            print("\n  SSIM intra-scénario (élevé = cohérence) :")
            for name, score in sorted(intra_ssim.items()):
                print(f"    {score:.4f}  {name}")

            avg_intra = np.mean(list(intra_ssim.values()))
            print(f"\n  Moyenne intra-scénario : {avg_intra:.4f}")

            # Sauvegarder CSV
            import pandas as pd
            rows = []
            for pair, score in inter_ssim.items():
                rows.append({"type": "inter", "pair": pair, "ssim": score})
            for name, score in intra_ssim.items():
                rows.append({"type": "intra", "pair": name, "ssim": score})
            pd.DataFrame(rows).to_csv(eval_dir / "ssim_report.csv", index=False)
            print(f"\n  [CSV] {eval_dir / 'ssim_report.csv'}")

        except ImportError:
            print("  ⚠️ scikit-image non installé (pip install scikit-image)")

    # Nettoyage
    del pipe
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    print(f"\n{'=' * 60}")
    print("ÉVALUATION TERMINÉE")
    print(f"{'=' * 60}")
    print(f"  Résultats dans : {eval_dir}")


if __name__ == "__main__":
    main()
