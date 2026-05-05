#!/usr/bin/env python3
"""
prepare_dataset_lora.py — Prépare le dataset Merapi pour le fine-tuning LoRA.

Ce script :
  1. Filtre les images utilisables depuis l'index
  2. Sépare jour/nuit en sous-dossiers
  3. Crée un split train/val (90/10)
  4. Génère les fichiers metadata.jsonl (format HuggingFace datasets)
  5. Vérifie l'intégrité des images (PNG, taille, mode)

Usage :
    python prepare_dataset_lora.py
    python prepare_dataset_lora.py --val-ratio 0.15 --resolution 512
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent


# Prompts pour les captions
CAPTION_DAY = (
    "surveillance camera photo of Mount Merapi volcano, daytime, "
    "gray volcanic terrain, lava flow deposits, smoke plume, "
    "Canon EOS 1100D, scientific monitoring"
)
CAPTION_NIGHT = (
    "surveillance camera photo of Mount Merapi volcano, nighttime, "
    "incandescent lava flow, glowing volcanic activity, dark terrain, "
    "Canon EOS 1100D, scientific monitoring"
)


def prepare(args: argparse.Namespace) -> None:
    index_path = PROJECT_ROOT / "data" / "index" / "index.csv"
    processed_base = PROJECT_ROOT / "data" / "processed"
    output_base = PROJECT_ROOT / "data" / "lora_dataset"

    # Nettoyer si existant
    if output_base.exists() and args.clean:
        shutil.rmtree(output_base)
        print(f"[Clean] Supprimé {output_base}")

    # Charger l'index
    df = pd.read_csv(index_path, dtype=str, na_values=["", "None", "nan"])
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")

    # Filtrer les images utilisables
    df = df[df["quality_flag"] == "usable"].copy()
    print(f"[Index] {len(df)} images utilisables")

    # Résoudre les chemins et vérifier l'existence
    records = []
    skipped = 0
    for _, row in df.iterrows():
        y = row.get("year")
        m = row.get("month")
        fn = row.get("filename", "")
        if pd.isna(y) or pd.isna(m) or not fn:
            skipped += 1
            continue

        png_stem = Path(fn).stem + ".png"
        png_path = processed_base / str(int(y)) / f"{int(m):02d}" / png_stem
        if not png_path.exists():
            skipped += 1
            continue

        # Vérifier l'image
        try:
            img = Image.open(png_path)
            if img.size[0] < 64 or img.size[1] < 64:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        hour = row.get("hour")
        is_day = True
        if pd.notna(hour):
            is_day = 6 <= int(hour) < 18

        records.append({
            "src_path": png_path,
            "filename": fn,
            "is_day": is_day,
            "year": int(y),
            "month": int(m),
        })

    print(f"[Vérif] {len(records)} images valides, {skipped} ignorées")

    if not records:
        print("[ERREUR] Aucune image valide trouvée.")
        sys.exit(1)

    # Séparer jour/nuit
    day_records = [r for r in records if r["is_day"]]
    night_records = [r for r in records if not r["is_day"]]
    print(f"[Split] Jour: {len(day_records)}, Nuit: {len(night_records)}")

    # Split train/val déterministe
    rng = np.random.default_rng(42)

    def split_records(recs, val_ratio):
        indices = rng.permutation(len(recs))
        n_val = max(1, int(len(recs) * val_ratio))
        val_idx = set(indices[:n_val])
        train = [recs[i] for i in range(len(recs)) if i not in val_idx]
        val = [recs[i] for i in range(len(recs)) if i in val_idx]
        return train, val

    day_train, day_val = split_records(day_records, args.val_ratio)
    night_train, night_val = split_records(night_records, args.val_ratio)

    print(f"[Split] Jour  — train: {len(day_train)}, val: {len(day_val)}")
    print(f"[Split] Nuit  — train: {len(night_train)}, val: {len(night_val)}")

    # Copier les images et créer les metadata
    subsets = {
        "train/day": (day_train, CAPTION_DAY),
        "train/night": (night_train, CAPTION_NIGHT),
        "val/day": (day_val, CAPTION_DAY),
        "val/night": (night_val, CAPTION_NIGHT),
    }

    for subset_name, (recs, caption) in subsets.items():
        subset_dir = output_base / subset_name
        subset_dir.mkdir(parents=True, exist_ok=True)

        metadata = []
        for i, rec in enumerate(recs):
            src = rec["src_path"]
            dst_name = f"{i:05d}.png"
            dst = subset_dir / dst_name

            # Copier et convertir si nécessaire
            img = Image.open(src).convert("L")
            if img.size != (args.resolution, args.resolution):
                img = img.resize((args.resolution, args.resolution), Image.LANCZOS)

            # Sauvegarder en RGB (SD attend 3 canaux)
            img_rgb = Image.merge("RGB", [img, img, img])
            img_rgb.save(str(dst), format="PNG")

            metadata.append({
                "file_name": dst_name,
                "text": caption,
                "original_filename": rec["filename"],
                "is_day": rec["is_day"],
                "year": rec["year"],
                "month": rec["month"],
            })

        # Écrire metadata.jsonl
        meta_path = subset_dir / "metadata.jsonl"
        with open(meta_path, "w", encoding="utf-8") as f:
            for entry in metadata:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"  [{subset_name}] {len(recs)} images → {subset_dir}")

    # Résumé
    total = len(day_train) + len(night_train) + len(day_val) + len(night_val)
    print(f"\n{'='*50}")
    print(f"Dataset préparé : {total} images")
    print(f"  Train : {len(day_train) + len(night_train)} "
          f"(jour={len(day_train)}, nuit={len(night_train)})")
    print(f"  Val   : {len(day_val) + len(night_val)} "
          f"(jour={len(day_val)}, nuit={len(night_val)})")
    print(f"  Sortie: {output_base}")
    print(f"{'='*50}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prépare le dataset Merapi pour le fine-tuning LoRA"
    )
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Ratio de validation (défaut: 0.1)")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Résolution cible (256 ou 512)")
    parser.add_argument("--clean", action="store_true",
                        help="Supprimer le dataset existant avant de recréer")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(args)
