#!/usr/bin/env python3
"""
evaluate_lora_physics.py — Évaluation du LoRA physics-conditionné du Merapi.

Évalue la capacité du modèle à respecter les paramètres physiques
en générant des images pour chaque scénario et en comparant les
métriques SSIM/FID par catégorie de paramètres.

Métriques :
  - FID par catégorie (jour vs nuit, calme vs éruption, par caméra)
  - SSIM par catégorie
  - Grille de comparaison : baseline vs LoRA physics pour chaque scénario
  - Cohérence inter-paramètres (le modèle génère-t-il des images
    plausibles pour des paramètres extrêmes ?)

Usage :
    python evaluate_lora_physics.py
    python evaluate_lora_physics.py --fid --num-gen 50
    python evaluate_lora_physics.py --lora-path outputs/lora_merapi_physics/lora_merapi_physics_final

Prérequis :
    pip install torch diffusers peft pillow scikit-image torchvision pandas scipy
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
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from train_lora_physics import (
    NEGATIVE_PROMPT,
    PROMPT_TEMPLATE,
    TEST_SCENARIOS,
    build_structured_prompt,
    detect_device,
)


# ============================================================
# 1. Scénarios d'évaluation étendus
# ============================================================

EVAL_SCENARIOS = TEST_SCENARIOS + [
    # Activité basse de jour
    {
        "name": "matin_calme_suki",
        "params": {
            "camera": "Suki", "time_of_day": "early_morning",
            "brightness": "daylight", "lava_intensity": "low",
            "slope": "30deg_south", "weather": "clear",
        },
        "category": {"time": "day", "activity": "low"},
    },
    # Éruption intense la nuit vue depuis Kalor
    {
        "name": "nuit_intense_kalor",
        "params": {
            "camera": "Kalor", "time_of_day": "night",
            "brightness": "incandescent_glow", "lava_intensity": "very_high",
            "slope": "25deg_west", "weather": "clear_night",
        },
        "category": {"time": "night", "activity": "high"},
    },
    # Ciel couvert, activité modérée
    {
        "name": "jour_couvert_moderate",
        "params": {
            "camera": "Suki", "time_of_day": "afternoon",
            "brightness": "daylight", "lava_intensity": "moderate",
            "slope": "30deg_south", "weather": "overcast",
        },
        "category": {"time": "day", "activity": "moderate"},
    },
    # Crépuscule, activité forte
    {
        "name": "crepuscule_actif_suki",
        "params": {
            "camera": "Suki", "time_of_day": "dusk",
            "brightness": "bright_with_incandescence", "lava_intensity": "high",
            "slope": "30deg_south", "weather": "clear",
        },
        "category": {"time": "dusk", "activity": "high"},
    },
    # Nuit sans activité (cas limite — le modèle doit générer une image sombre)
    {
        "name": "nuit_calme_suki",
        "params": {
            "camera": "Suki", "time_of_day": "night",
            "brightness": "dark", "lava_intensity": "none",
            "slope": "30deg_south", "weather": "clear_night",
        },
        "category": {"time": "night", "activity": "none"},
    },
    # Midi, très actif
    {
        "name": "midi_eruption_kalor",
        "params": {
            "camera": "Kalor", "time_of_day": "midday",
            "brightness": "bright_with_incandescence", "lava_intensity": "very_high",
            "slope": "25deg_west", "weather": "clear",
        },
        "category": {"time": "day", "activity": "high"},
    },
]

# Ajouter les catégories par défaut aux scénarios de test originaux
for s in EVAL_SCENARIOS[:4]:
    if "category" not in s:
        name = s["name"]
        if "nuit" in name:
            s["category"] = {"time": "night", "activity": "high" if "eruption" in name else "moderate"}
        elif "dusk" in name:
            s["category"] = {"time": "dusk", "activity": "moderate"}
        else:
            s["category"] = {"time": "day", "activity": "none"}


# ============================================================
# 2. SSIM
# ============================================================

def compute_ssim_batch(real_dir: Path, gen_dir: Path, max_images: int = 100) -> dict:
    """SSIM moyen entre images réelles et générées."""
    from skimage.metrics import structural_similarity as ssim

    real_paths = sorted(real_dir.glob("*.png"))[:max_images]
    gen_paths = sorted(gen_dir.glob("*.png"))[:max_images]
    n = min(len(real_paths), len(gen_paths))
    if n == 0:
        return {"ssim_mean": 0.0, "ssim_std": 0.0, "n": 0}

    scores = []
    for rp, gp in zip(real_paths[:n], gen_paths[:n]):
        real_img = np.array(Image.open(rp).convert("L"), dtype=np.float64)
        gen_img = np.array(Image.open(gp).convert("L"), dtype=np.float64)
        h = min(real_img.shape[0], gen_img.shape[0])
        w = min(real_img.shape[1], gen_img.shape[1])
        s = ssim(real_img[:h, :w], gen_img[:h, :w], data_range=255.0)
        scores.append(s)

    return {"ssim_mean": float(np.mean(scores)), "ssim_std": float(np.std(scores)), "n": n}


# ============================================================
# 3. FID
# ============================================================

def compute_fid(real_dir: Path, gen_dir: Path, device=None, max_images=200) -> float:
    """FID entre deux dossiers d'images via InceptionV3."""
    from torchvision import transforms
    from torchvision.models import inception_v3
    from scipy.linalg import sqrtm

    if device is None:
        device = detect_device()

    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    def extract(d: Path) -> np.ndarray:
        paths = sorted(d.glob("*.png"))[:max_images]
        feats = []
        for i in range(0, len(paths), 8):
            batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[i:i+8]]).to(device)
            with torch.no_grad():
                feats.append(model(batch).cpu().numpy())
        return np.concatenate(feats, 0) if feats else np.zeros((0, 2048))

    f_real, f_gen = extract(real_dir), extract(gen_dir)
    if f_real.shape[0] < 2 or f_gen.shape[0] < 2:
        return float("inf")

    mu_r, mu_g = np.mean(f_real, 0), np.mean(f_gen, 0)
    s_r, s_g = np.cov(f_real, rowvar=False), np.cov(f_gen, rowvar=False)
    diff = mu_r - mu_g
    covmean = sqrtm(s_r @ s_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(s_r + s_g - 2.0 * covmean))


# ============================================================
# 4. Chargement du pipeline avec LoRA physics
# ============================================================

def load_pipeline(lora_path: Path | None, device=None):
    """Charge le pipeline SD 1.5 avec optionnellement le LoRA physics."""
    from diffusers import (
        AutoencoderKL, DDPMScheduler, StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextModel, CLIPTokenizer

    if device is None:
        device = detect_device()
    dtype = torch.float32
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    _kw: dict[str, Any] = {}
    try:
        CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", local_files_only=True)
        _kw["local_files_only"] = True
    except Exception:
        pass

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", **_kw)
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", dtype=dtype, **_kw).to(device)
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=dtype, **_kw).to(device)
    try:
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=dtype, **_kw)
    except (OSError, EnvironmentError):
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=torch.float16,
            variant="fp16", **_kw).to(dtype=dtype)
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler", **_kw)

    if lora_path and lora_path.exists():
        from peft import PeftModel
        print(f"[Pipeline] Chargement LoRA depuis {lora_path}")
        unet = PeftModel.from_pretrained(unet, str(lora_path))
        unet = unet.base_model.model

    unet.to(device)
    pipe = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        unet=unet, scheduler=scheduler,
        safety_checker=None, feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)
    if device.type == "mps":
        pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ============================================================
# 5. Génération par scénario physique
# ============================================================

def generate_scenario_images(
    pipe,
    scenarios: list[dict],
    output_dir: Path,
    num_per_scenario: int = 5,
    seed: int = 42,
    steps: int = 25,
) -> dict[str, Path]:
    """Génère num_per_scenario images pour chaque scénario physique."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths_map = {}

    for scenario in scenarios:
        name = scenario["name"]
        params = scenario["params"]
        prompt = PROMPT_TEMPLATE.format(**params)

        sc_dir = output_dir / name
        sc_dir.mkdir(exist_ok=True)
        paths_map[name] = sc_dir

        print(f"  {name} ({num_per_scenario} images)...")
        for i in range(num_per_scenario):
            gen = torch.Generator(device="cpu").manual_seed(seed + i)
            with torch.no_grad():
                result = pipe(
                    prompt, negative_prompt=NEGATIVE_PROMPT,
                    num_inference_steps=steps, guidance_scale=7.5,
                    generator=gen,
                )
            result.images[0].save(str(sc_dir / f"{name}_{i:03d}.png"), format="PNG")

    return paths_map


# ============================================================
# 6. Collecte d'images réelles filtrées par catégorie
# ============================================================

def collect_real_by_category(
    index_path: Path,
    processed_base: Path,
    output_dir: Path,
    max_per_category: int = 50,
) -> dict[str, Path]:
    """Collecte les images réelles par catégorie (jour/nuit, activité)."""
    import shutil

    df = pd.read_csv(index_path, dtype=str, na_values=["", "None", "nan"])
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df["anomaly_score"] = pd.to_numeric(df["anomaly_score"], errors="coerce").fillna(0.0)
    df = df[df["quality_flag"] == "usable"].copy()

    categories = {
        "day_calm": df[(df["hour"].between(6, 17)) & (df["anomaly_score"] < 0.1)],
        "day_active": df[(df["hour"].between(6, 17)) & (df["anomaly_score"] >= 0.15)],
        "night_active": df[(~df["hour"].between(6, 17)) & (df["anomaly_score"] >= 0.15)],
        "night_calm": df[(~df["hour"].between(6, 17)) & (df["anomaly_score"] < 0.05)],
        "dusk": df[df["hour"].between(18, 20)],
    }

    cat_dirs = {}
    rng = np.random.default_rng(42)

    for cat_name, cat_df in categories.items():
        cat_dir = output_dir / "real" / cat_name
        cat_dir.mkdir(parents=True, exist_ok=True)
        cat_dirs[cat_name] = cat_dir

        n = min(max_per_category, len(cat_df))
        if n == 0:
            continue

        sample = cat_df.sample(n=n, random_state=42)
        count = 0
        for _, row in sample.iterrows():
            y, m, fn = row.get("year"), row.get("month"), row.get("filename", "")
            if pd.isna(y) or pd.isna(m) or not fn:
                continue
            png = processed_base / str(int(float(y))) / f"{int(float(m)):02d}" / (Path(fn).stem + ".png")
            if png.exists():
                shutil.copy2(png, cat_dir / f"real_{count:04d}.png")
                count += 1

        print(f"  [{cat_name}] {count} images réelles")

    return cat_dirs


# ============================================================
# 7. Grille de comparaison multi-scénarios
# ============================================================

def create_physics_grid(
    scenario_dirs: dict[str, Path],
    output_path: Path,
    max_cols: int = 4,
) -> None:
    """Crée une grille visuelle montrant un exemple par scénario."""
    scenarios = list(scenario_dirs.items())
    n = len(scenarios)
    if n == 0:
        return

    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols

    # Prendre la première image de chaque scénario
    imgs = []
    labels = []
    for name, d in scenarios:
        pngs = sorted(d.glob("*.png"))
        if pngs:
            imgs.append(Image.open(pngs[0]).convert("RGB"))
            labels.append(name)

    if not imgs:
        return

    w, h = imgs[0].size
    pad = 4
    label_h = 20
    grid_w = cols * (w + pad) + pad
    grid_h = rows * (h + pad + label_h) + pad

    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    for i, img in enumerate(imgs):
        r, c = divmod(i, cols)
        x = pad + c * (w + pad)
        y = pad + r * (h + pad + label_h) + label_h
        grid.paste(img.resize((w, h)), (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(output_path), format="PNG")
    print(f"[Grille] {output_path}")


# ============================================================
# 8. Rapport complet
# ============================================================

def generate_report(results: dict, output_path: Path) -> None:
    """Génère un rapport CSV détaillé des métriques."""
    rows = []
    for key, val in results.items():
        if isinstance(val, dict):
            for sub_k, sub_v in val.items():
                rows.append({"metric": f"{key}_{sub_k}", "value": sub_v})
        else:
            rows.append({"metric": key, "value": val})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n[Rapport] {output_path}")
    print(df.to_string(index=False))


# ============================================================
# 9. Main
# ============================================================

def main(args: argparse.Namespace) -> None:
    device = detect_device()
    print(f"[Config] Device={device}")

    # Localiser le LoRA
    if args.lora_path:
        lora_path = Path(args.lora_path)
    else:
        physics_path = PROJECT_ROOT / "outputs" / "lora_merapi_physics" / "lora_merapi_physics_final"
        colab_path = PROJECT_ROOT / "outputs" / "lora_merapi_physics_results" / "lora_merapi_physics_final"
        if physics_path.exists():
            lora_path = physics_path
        elif colab_path.exists():
            lora_path = colab_path
        else:
            # Fallback sur le LoRA de base
            lora_path = PROJECT_ROOT / "outputs" / "lora_merapi_results" / "lora_merapi_final"

    print(f"[LoRA] {lora_path} (exists={lora_path.exists()})")

    eval_dir = PROJECT_ROOT / "outputs" / "lora_merapi_physics" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    # ----- 1. Générer images par scénario -----
    if args.generate:
        print("\n=== GÉNÉRATION BASELINE (sans LoRA) ===")
        pipe_base = load_pipeline(lora_path=None, device=device)
        base_dirs = generate_scenario_images(
            pipe_base, EVAL_SCENARIOS, eval_dir / "baseline",
            num_per_scenario=args.num_per_scenario, seed=42,
        )
        del pipe_base
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print("\n=== GÉNÉRATION AVEC LoRA PHYSICS ===")
        pipe_lora = load_pipeline(lora_path=lora_path, device=device)
        lora_dirs = generate_scenario_images(
            pipe_lora, EVAL_SCENARIOS, eval_dir / "with_lora",
            num_per_scenario=args.num_per_scenario, seed=42,
        )
        del pipe_lora
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        base_dirs = {s["name"]: eval_dir / "baseline" / s["name"] for s in EVAL_SCENARIOS}
        lora_dirs = {s["name"]: eval_dir / "with_lora" / s["name"] for s in EVAL_SCENARIOS}

    # ----- 2. Collecter images réelles par catégorie -----
    print("\n=== COLLECTE IMAGES RÉELLES PAR CATÉGORIE ===")
    index_path = PROJECT_ROOT / "data" / "index" / "index.csv"
    processed_base = PROJECT_ROOT / "data" / "processed"

    cat_dirs = collect_real_by_category(
        index_path, processed_base, eval_dir,
        max_per_category=args.num_per_scenario * 5,
    )

    # ----- 3. SSIM par scénario -----
    if args.ssim:
        print("\n=== SSIM PAR SCÉNARIO ===")
        ssim_results = {}

        # Mapper scénarios → catégories réelles
        scenario_to_real = {
            "jour_calme_suki": "day_calm",
            "matin_calme_suki": "day_calm",
            "jour_couvert_moderate": "day_active",
            "midi_eruption_kalor": "day_active",
            "nuit_eruption_kalor": "night_active",
            "nuit_intense_kalor": "night_active",
            "crepuscule_actif_suki": "dusk",
            "dusk_moderate_suki": "dusk",
            "nuit_faible_couvert": "night_calm",
            "nuit_calme_suki": "night_calm",
        }

        for scenario in EVAL_SCENARIOS:
            name = scenario["name"]
            real_cat = scenario_to_real.get(name, "day_calm")
            real_dir = cat_dirs.get(real_cat)

            if real_dir and real_dir.exists() and (eval_dir / "with_lora" / name).exists():
                s = compute_ssim_batch(real_dir, eval_dir / "with_lora" / name)
                ssim_results[name] = s["ssim_mean"]
                print(f"  {name}: SSIM={s['ssim_mean']:.4f} ± {s['ssim_std']:.4f}")

        results["ssim_per_scenario"] = ssim_results

        # SSIM global
        all_gen = eval_dir / "with_lora"
        all_real = eval_dir / "real" / "day_calm"
        if all_gen.exists() and all_real and all_real.exists():
            s_glob = compute_ssim_batch(all_real, all_gen)
            results["ssim_global"] = s_glob["ssim_mean"]

    # ----- 4. FID -----
    if args.fid:
        print("\n=== FID PAR CATÉGORIE ===")
        fid_results = {}

        for cat_name, real_dir in cat_dirs.items():
            # Trouver les scénarios générés correspondants
            match_dirs = []
            scenario_to_real_inv = {}
            for sc in EVAL_SCENARIOS:
                n = sc["name"]
                cat = sc.get("category", {})
                if cat.get("activity") in ("high", "moderate") and cat_name == "day_active":
                    match_dirs.append(eval_dir / "with_lora" / n)
                elif cat.get("time") == "night" and cat.get("activity") in ("high",) and cat_name == "night_active":
                    match_dirs.append(eval_dir / "with_lora" / n)

            # FID baseline vs real pour la catégorie
            if real_dir.exists() and len(list(real_dir.glob("*.png"))) >= 2:
                # Combiner les images générées
                import shutil
                combined = eval_dir / "combined_gen" / cat_name
                combined.mkdir(parents=True, exist_ok=True)
                idx = 0
                for d in match_dirs:
                    if d.exists():
                        for p in d.glob("*.png"):
                            shutil.copy2(p, combined / f"gen_{idx:04d}.png")
                            idx += 1

                if idx >= 2:
                    fid_val = compute_fid(real_dir, combined, device=device)
                    fid_results[cat_name] = fid_val
                    print(f"  FID({cat_name}): {fid_val:.2f}")

        results["fid_per_category"] = fid_results

    # ----- 5. Grilles visuelles -----
    print("\n=== GRILLES VISUELLES ===")
    if any((eval_dir / "with_lora" / s["name"]).exists() for s in EVAL_SCENARIOS):
        existing = {s["name"]: eval_dir / "with_lora" / s["name"]
                    for s in EVAL_SCENARIOS
                    if (eval_dir / "with_lora" / s["name"]).exists()}
        create_physics_grid(existing, eval_dir / "physics_grid_lora.png")

    if any((eval_dir / "baseline" / s["name"]).exists() for s in EVAL_SCENARIOS):
        existing_base = {s["name"]: eval_dir / "baseline" / s["name"]
                         for s in EVAL_SCENARIOS
                         if (eval_dir / "baseline" / s["name"]).exists()}
        create_physics_grid(existing_base, eval_dir / "physics_grid_baseline.png")

    # ----- 6. Rapport -----
    generate_report(results, eval_dir / "physics_evaluation_report.csv")

    print("\n=== ÉVALUATION PHYSIQUE TERMINÉE ===")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Évaluation du LoRA physics-conditionné Merapi"
    )
    p.add_argument("--lora-path", type=str, default=None)
    p.add_argument("--generate", action="store_true", default=True)
    p.add_argument("--no-generate", action="store_true")
    p.add_argument("--num-per-scenario", type=int, default=5)
    p.add_argument("--ssim", action="store_true", default=True)
    p.add_argument("--fid", action="store_true")
    args = p.parse_args()
    if args.no_generate:
        args.generate = False
    return args


if __name__ == "__main__":
    main(parse_args())
