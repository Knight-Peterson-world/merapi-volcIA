#!/usr/bin/env python3
"""
evaluate_lora_merapi.py — Évaluation du LoRA fine-tuné sur Merapi.

Métriques :
  - FID  (Fréchet Inception Distance)  : diversité et qualité vs images réelles
  - SSIM (Structural Similarity Index) : similarité structurelle pixel-level
  - Comparaisons visuelles avant/après fine-tuning

Usage :
    python evaluate_lora_merapi.py
    python evaluate_lora_merapi.py --num-gen 50 --fid
    python evaluate_lora_merapi.py --lora-path outputs/lora_merapi/lora_merapi_final

Prérequis :
    pip install torch diffusers peft pillow scikit-image torchvision pandas
    pip install pytorch-fid   # pour FID (optionnel)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Empêcher l'import de TensorFlow (conflit protobuf avec transformers)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 1. SSIM
# ============================================================

def compute_ssim_batch(
    real_dir: Path,
    gen_dir: Path,
    max_images: int = 100,
) -> dict:
    """
    Calcule le SSIM moyen entre images réelles et générées.
    Compare les images dans l'ordre alphabétique, recadrées
    à la même taille.
    """
    from skimage.metrics import structural_similarity as ssim

    real_paths = sorted(real_dir.glob("*.png"))[:max_images]
    gen_paths = sorted(gen_dir.glob("*.png"))[:max_images]

    n = min(len(real_paths), len(gen_paths))
    if n == 0:
        print("[SSIM] Aucune image trouvée.")
        return {"ssim_mean": 0.0, "ssim_std": 0.0, "n": 0}

    scores = []
    for rp, gp in zip(real_paths[:n], gen_paths[:n]):
        real_img = np.array(Image.open(rp).convert("L"), dtype=np.float64)
        gen_img = np.array(Image.open(gp).convert("L"), dtype=np.float64)

        # Harmoniser les tailles
        h = min(real_img.shape[0], gen_img.shape[0])
        w = min(real_img.shape[1], gen_img.shape[1])
        real_img = real_img[:h, :w]
        gen_img = gen_img[:h, :w]

        s = ssim(real_img, gen_img, data_range=255.0)
        scores.append(s)

    return {
        "ssim_mean": float(np.mean(scores)),
        "ssim_std": float(np.std(scores)),
        "n": n,
    }


# ============================================================
# 2. FID (via torchvision InceptionV3)
# ============================================================

def compute_fid(
    real_dir: Path,
    gen_dir: Path,
    device: torch.device | None = None,
    batch_size: int = 8,
    max_images: int = 200,
) -> float:
    """
    Calcule le FID entre deux dossiers d'images via InceptionV3.

    Implémentation « lightweight » sans dépendance pytorch-fid,
    compatible MPS/CPU/CUDA.
    """
    from torchvision import transforms
    from torchvision.models import inception_v3

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # Charger InceptionV3
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()  # Extraire les features, pas les logits
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    def extract_features(img_dir: Path) -> np.ndarray:
        paths = sorted(img_dir.glob("*.png"))[:max_images]
        all_feats = []

        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            imgs = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                imgs.append(transform(img))

            batch = torch.stack(imgs).to(device)
            with torch.no_grad():
                feats = model(batch)
            all_feats.append(feats.cpu().numpy())

        if not all_feats:
            return np.zeros((0, 2048))
        return np.concatenate(all_feats, axis=0)

    print(f"[FID] Extraction des features réelles ({real_dir})...")
    feats_real = extract_features(real_dir)
    print(f"[FID] Extraction des features générées ({gen_dir})...")
    feats_gen = extract_features(gen_dir)

    if feats_real.shape[0] < 2 or feats_gen.shape[0] < 2:
        print("[FID] Pas assez d'images pour calculer le FID.")
        return float("inf")

    # Calculer les statistiques
    mu_real = np.mean(feats_real, axis=0)
    mu_gen = np.mean(feats_gen, axis=0)
    sigma_real = np.cov(feats_real, rowvar=False)
    sigma_gen = np.cov(feats_gen, rowvar=False)

    # FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1 @ sigma2))
    from scipy.linalg import sqrtm

    diff = mu_real - mu_gen
    covmean = sqrtm(sigma_real @ sigma_gen)

    # Correction numérique
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_real + sigma_gen - 2.0 * covmean))
    return fid


# ============================================================
# 3. Génération d'images pour l'évaluation
# ============================================================

def generate_eval_images(
    lora_path: Path | None,
    output_dir: Path,
    num_images: int = 50,
    resolution: int = 256,
    ti_path: Path | None = None,
    seed: int = 42,
) -> Path:
    """Génère des images avec le modèle (avec ou sans LoRA) pour l'évaluation."""
    from diffusers import (
        AutoencoderKL, DDPMScheduler, StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextModel, CLIPTokenizer
    from peft import PeftModel

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    dtype = torch.float32
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    print(f"[Gen] Chargement du pipeline SD 1.5 (composant par composant)...")
    _kw: dict = {}
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

    # Charger le LoRA si fourni
    if lora_path and lora_path.exists():
        print(f"[Gen] Chargement du LoRA depuis {lora_path}...")
        unet = PeftModel.from_pretrained(unet, str(lora_path))
        unet = unet.base_model.model  # unwrap PEFT pour le pipeline
        print("[Gen] LoRA chargé.")

    unet.to(device)
    pipe = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        unet=unet, scheduler=scheduler,
        safety_checker=None, feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)

    # Charger TI si fourni
    if ti_path and (ti_path / "learned_embeds.bin").exists():
        embeds = torch.load(ti_path / "learned_embeds.bin", weights_only=True)
        for token, embed in embeds.items():
            pipe.tokenizer.add_tokens([token])
            pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
            token_id = pipe.tokenizer.convert_tokens_to_ids(token)
            with torch.no_grad():
                pipe.text_encoder.get_input_embeddings().weight[token_id] = embed.to(device)
            print(f"[Gen] Token TI '{token}' chargé.")

    if device.type == "mps":
        pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)

    # Prompts variés
    day_prompt = (
        "volcanic landscape of Mount Merapi, daytime surveillance camera, "
        "gray terrain, lava flow, volcanic texture, Canon EOS 1100D"
    )
    night_prompt = (
        "volcanic landscape of Mount Merapi, nighttime surveillance camera, "
        "incandescence, glowing lava, Canon EOS 1100D"
    )

    gen_dir = output_dir / ("with_lora" if lora_path else "baseline")
    gen_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cpu").manual_seed(seed)

    for i in range(num_images):
        prompt = day_prompt if i % 2 == 0 else night_prompt
        with torch.no_grad():
            result = pipe(
                prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator,
                height=resolution,
                width=resolution,
            )
        img = result.images[0]
        img.save(str(gen_dir / f"gen_{i:04d}.png"), format="PNG")

        if (i + 1) % 10 == 0:
            print(f"  [Gen] {i + 1}/{num_images}")

    del pipe
    return gen_dir


# ============================================================
# 4. Collecte d'images réelles pour comparaison
# ============================================================

def collect_real_images(
    processed_base: Path,
    output_dir: Path,
    max_images: int = 200,
) -> Path:
    """
    Copie un échantillon d'images réelles dans un dossier dédié
    pour le calcul du FID.
    """
    import shutil

    real_dir = output_dir / "real"
    real_dir.mkdir(parents=True, exist_ok=True)

    # Collecter toutes les images processed
    all_pngs = sorted(processed_base.rglob("*.png"))
    if len(all_pngs) == 0:
        print("[Réel] Aucune image PNG trouvée dans processed/")
        return real_dir

    # Échantillonner
    rng = np.random.default_rng(42)
    indices = rng.choice(len(all_pngs), min(max_images, len(all_pngs)), replace=False)

    for idx in sorted(indices):
        src = all_pngs[idx]
        dst = real_dir / f"real_{idx:04d}.png"
        shutil.copy2(src, dst)

    print(f"[Réel] {len(indices)} images copiées dans {real_dir}")
    return real_dir


# ============================================================
# 5. Grille de comparaison visuelle
# ============================================================

def create_comparison_grid(
    before_dir: Path,
    after_dir: Path,
    output_path: Path,
    n: int = 4,
) -> None:
    """Crée une grille visuelle avant/après fine-tuning."""
    before = sorted(before_dir.glob("*.png"))[:n]
    after = sorted(after_dir.glob("*.png"))[:n]

    if not before or not after:
        print("[Grille] Pas assez d'images pour la comparaison.")
        return

    n = min(len(before), len(after))
    imgs_before = [Image.open(p).convert("RGB") for p in before[:n]]
    imgs_after = [Image.open(p).convert("RGB") for p in after[:n]]

    w, h = imgs_before[0].size
    padding = 4
    label_h = 30

    grid_w = n * (w + padding) + padding
    grid_h = 2 * (h + padding) + padding + label_h
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

    for i, img in enumerate(imgs_before):
        x = padding + i * (w + padding)
        grid.paste(img.resize((w, h)), (x, padding + label_h))

    for i, img in enumerate(imgs_after):
        x = padding + i * (w + padding)
        grid.paste(img.resize((w, h)), (x, padding + label_h + h + padding))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(output_path), format="PNG")
    print(f"[Grille] Comparaison sauvegardée → {output_path}")


# ============================================================
# 6. Main
# ============================================================

def main(args: argparse.Namespace) -> None:
    output_dir = PROJECT_ROOT / "outputs" / "lora_merapi"
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Chercher le LoRA : d'abord lora_merapi_results (Colab), sinon lora_merapi
    if args.lora_path:
        lora_path = Path(args.lora_path)
    else:
        colab_path = PROJECT_ROOT / "outputs" / "lora_merapi_results" / "lora_merapi_final"
        local_path = output_dir / "lora_merapi_final"
        lora_path = colab_path if colab_path.exists() else local_path
    ti_path = output_dir / "textual_inversion_merapi" if args.textual_inversion else None

    processed_base = PROJECT_ROOT / "data" / "processed"
    results = {}

    # ----- 1. Générer des images avec et sans LoRA -----
    if args.generate:
        print("\n=== GÉNÉRATION D'IMAGES ===")

        print("\n--- Sans LoRA (baseline) ---")
        baseline_dir = generate_eval_images(
            lora_path=None,
            output_dir=eval_dir,
            num_images=args.num_gen,
            resolution=args.resolution,
        )

        print("\n--- Avec LoRA ---")
        lora_dir = generate_eval_images(
            lora_path=lora_path,
            output_dir=eval_dir,
            num_images=args.num_gen,
            resolution=args.resolution,
            ti_path=ti_path,
        )
    else:
        baseline_dir = eval_dir / "baseline"
        lora_dir = eval_dir / "with_lora"

    # ----- 2. Collecter images réelles -----
    print("\n=== COLLECTE D'IMAGES RÉELLES ===")
    real_dir = collect_real_images(processed_base, eval_dir, max_images=args.num_gen * 2)

    # ----- 3. SSIM -----
    if args.ssim:
        print("\n=== SSIM ===")

        if baseline_dir.exists():
            ssim_base = compute_ssim_batch(real_dir, baseline_dir, max_images=args.num_gen)
            print(f"  SSIM (baseline vs réel) : {ssim_base['ssim_mean']:.4f} "
                  f"± {ssim_base['ssim_std']:.4f} (n={ssim_base['n']})")
            results["ssim_baseline_mean"] = ssim_base["ssim_mean"]
            results["ssim_baseline_std"] = ssim_base["ssim_std"]

        if lora_dir.exists():
            ssim_lora = compute_ssim_batch(real_dir, lora_dir, max_images=args.num_gen)
            print(f"  SSIM (LoRA vs réel)     : {ssim_lora['ssim_mean']:.4f} "
                  f"± {ssim_lora['ssim_std']:.4f} (n={ssim_lora['n']})")
            results["ssim_lora_mean"] = ssim_lora["ssim_mean"]
            results["ssim_lora_std"] = ssim_lora["ssim_std"]

    # ----- 4. FID -----
    if args.fid:
        print("\n=== FID ===")

        if baseline_dir.exists():
            fid_base = compute_fid(real_dir, baseline_dir, max_images=args.num_gen)
            print(f"  FID (baseline vs réel) : {fid_base:.2f}")
            results["fid_baseline"] = fid_base

        if lora_dir.exists():
            fid_lora = compute_fid(real_dir, lora_dir, max_images=args.num_gen)
            print(f"  FID (LoRA vs réel)     : {fid_lora:.2f}")
            results["fid_lora"] = fid_lora

    # ----- 5. Grille visuelle -----
    print("\n=== COMPARAISON VISUELLE ===")
    samples_before = output_dir / "samples" / "before_ft"
    samples_after = output_dir / "samples" / "after_ft"

    if samples_before.exists() and samples_after.exists():
        create_comparison_grid(
            samples_before, samples_after,
            eval_dir / "comparison_grid.png",
            n=4,
        )

    # ----- 6. Rapport -----
    if results:
        report_path = eval_dir / "evaluation_report.csv"
        pd.DataFrame([results]).to_csv(report_path, index=False)
        print(f"\n[Rapport] Métriques sauvegardées → {report_path}")

    print("\n=== ÉVALUATION TERMINÉE ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Évaluation du LoRA fine-tuné sur le dataset Merapi"
    )
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Chemin vers le LoRA (défaut: outputs/lora_merapi/lora_merapi_final)")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-gen", type=int, default=50,
                        help="Nombre d'images à générer pour l'évaluation")
    parser.add_argument("--generate", action="store_true", default=True,
                        help="Générer de nouvelles images (par défaut)")
    parser.add_argument("--no-generate", action="store_true",
                        help="Utiliser les images déjà générées")
    parser.add_argument("--ssim", action="store_true", default=True,
                        help="Calculer SSIM")
    parser.add_argument("--fid", action="store_true",
                        help="Calculer FID (nécessite torchvision + scipy)")
    parser.add_argument("--textual-inversion", action="store_true",
                        help="Charger le token TI pour l'évaluation")

    args = parser.parse_args()
    if args.no_generate:
        args.generate = False
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
