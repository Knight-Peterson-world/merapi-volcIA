"""
physics_prompts.py — Rich natural language prompt engineering for
                     physics-conditioned Merapi image generation.

Module partagé entre l'entraînement, l'évaluation, le générateur et
l'interface Streamlit.

Principe de conception :
  CLIP (l'encodeur texte de SD 1.5) a été entraîné sur du langage naturel,
  PAS sur des paires clé=valeur structurées.
  "lava_intensity=very_high" ne signifie rien pour CLIP, mais
  "bright orange-red incandescent lava flowing intensely" est bien compris.

  Ce module traduit les paramètres physiques → descriptions en langage naturel
  → embeddings CLIP efficaces.

Paramètres physiques supportés :
  - camera           : point de vue (Suki, Kalor, Kali)
  - time_of_day      : moment de la journée
  - brightness       : luminosité estimée
  - lava_intensity   : intensité de l'activité lavique
  - slope            : pente visible
  - weather          : conditions météo
  - viscosity        : viscosité de la lave (NEW)
  - temperature      : température apparente (NEW)
  - eruption_type    : type d'éruption (NEW)
  - plume            : panache volcanique (NEW)
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
from PIL import Image

# ─── Negative prompt anti-hallucinations ──────────────────────
PHYSICS_NEGATIVE_PROMPT = (
    "extra volcanoes, duplicate peaks, multiple mountains, fantasy landscape, "
    "buildings, houses, city, people, vehicles, text, watermark, logo, signature, "
    "cartoon, anime, drawing, painting, illustration, digital art, 3D render, "
    "low quality, blurry, pixelated, jpeg artifacts, wrong perspective, "
    "oversaturated, neon colors, unrealistic colors"
)


# ═══════════════════════════════════════════════════════════════
# Paramètre → Description en langage naturel (CLIP-friendly)
# ═══════════════════════════════════════════════════════════════

CAMERA_DESCRIPTIONS = {
    "Suki": "south face view from Suki monitoring station, looking up at the summit cone",
    "Kalor": "west face view from Kalor station, lateral perspective showing the crater rim",
    "Kali": "east face view from Kali station, showing the eastern slope and ravines",
}

TIME_DESCRIPTIONS = {
    "early_morning": "early morning with soft warm golden light and long shadows across the slopes",
    "midday": "bright midday under overhead sunlight with strong contrast and sharp shadows",
    "afternoon": "warm afternoon light with golden tone on the volcanic terrain",
    "dusk": "dramatic dusk with orange and purple sky, volcano silhouette against vivid sunset colors",
    "night": "dark nighttime scene with the mountain barely visible against the dark sky",
}

BRIGHTNESS_DESCRIPTIONS = {
    "daylight": "well-lit in natural daylight with good visibility of terrain details and texture",
    "bright_with_incandescence": "bright daylight with visible volcanic incandescence glowing on the slopes",
    "incandescent_glow": "volcanic incandescence illuminating the dark slopes with intense orange-red glow",
    "dim_glow": "faint dim volcanic glow barely visible against the dark mountain silhouette",
    "dark": "very dark with minimal visibility, mountain silhouette barely discernible",
}

LAVA_DESCRIPTIONS = {
    "none": "No visible lava, calm dormant volcanic slopes with gray rocky terrain",
    "low": "Faint traces of lava visible as small orange-red dots scattered on the upper slope",
    "moderate": "Moderate lava flow creating visible bright orange streams down the mountainside channels",
    "high": "Intense lava flow with bright orange-yellow streams cascading down multiple drainage channels",
    "very_high": (
        "Massive incandescent lava flow covering large portions of the slope, "
        "extremely bright white-orange hot material streaming downhill in wide channels"
    ),
}

WEATHER_DESCRIPTIONS = {
    "clear": "clear sky with excellent visibility",
    "overcast": "thick cloud cover partially obscuring the summit, diffuse soft lighting",
    "hazy": "hazy atmospheric conditions with mist and reduced visibility",
    "clear_night": "clear dark night sky",
}

VISCOSITY_DESCRIPTIONS = {
    "low": "thin fluid rapidly flowing lava streams moving quickly downslope",
    "medium": "moderately viscous flowing lava at steady speed",
    "high": "thick slow-moving viscous lava with rough blocky aa-type texture",
}

TEMPERATURE_DESCRIPTIONS = {
    "low": "dim dark red glow indicating cooling solidifying lava at the surface",
    "moderate": "steady bright orange glow from actively flowing molten lava",
    "high": "bright orange-yellow incandescence from very hot volcanic material",
    "extreme": "brilliant white-hot incandescence from extremely hot freshly erupted material",
}

ERUPTION_DESCRIPTIONS = {
    "none": "",
    "effusive": "Effusive eruption with steady continuous lava outflow from the summit crater",
    "explosive": "Explosive eruption with ejected rocks and pyroclastic material flying upward",
    "phreatic": "Phreatic steam eruption with dense white vapor cloud billowing from the crater",
}

PLUME_DESCRIPTIONS = {
    "none": "",
    "low": "A small wispy steam plume rises from the crater rim",
    "medium": "A visible gray ash plume extends above the summit into the sky",
    "high": "A tall dense volcanic ash column rises high into the atmosphere",
}

# ─── Prompt templates (sélection aléatoire pour la diversité) ───
_TEMPLATES = [
    (
        "Surveillance camera photograph of Mount Merapi volcano, {camera_desc}. "
        "{time_desc}, {weather_desc}. {brightness_desc}. {lava_desc}. "
        "{extra}Rocky volcanic terrain with gray andesitic rock. "
        "Realistic monitoring camera image, natural photographic quality."
    ),
    (
        "Mount Merapi volcano monitoring webcam image, {camera_desc}. "
        "{time_desc}. {weather_desc}. {brightness_desc}. "
        "{lava_desc}. {extra}Volcanic landscape with rocky slopes. "
        "Real surveillance photograph, detailed natural image."
    ),
    (
        "Webcam photograph of Merapi stratovolcano in Java Indonesia, {camera_desc}. "
        "{time_desc}, {weather_desc}. {brightness_desc}. "
        "{lava_desc}. {extra}Andesitic volcanic terrain. "
        "High resolution monitoring station photograph."
    ),
]


# ═══════════════════════════════════════════════════════════════
# Construction de prompt
# ═══════════════════════════════════════════════════════════════

def build_rich_prompt(
    camera: str = "Suki",
    time_of_day: str = "midday",
    brightness: str = "daylight",
    lava_intensity: str = "none",
    slope: str = "30deg_south",
    weather: str = "clear",
    viscosity: str = "medium",
    temperature: str = "moderate",
    eruption_type: str = "none",
    plume: str = "none",
    *,
    template_index: int | None = None,
) -> str:
    """Construit un prompt riche en langage naturel depuis les paramètres physiques.

    Args:
        template_index: None → sélection aléatoire (diversité).
                        int  → template fixe (reproductibilité).
    Returns:
        Prompt en langage naturel compréhensible par CLIP.
    """
    camera_desc = CAMERA_DESCRIPTIONS.get(camera, f"view from {camera} station")
    time_desc = TIME_DESCRIPTIONS.get(time_of_day, time_of_day)
    brightness_desc = BRIGHTNESS_DESCRIPTIONS.get(brightness, brightness)
    lava_desc = LAVA_DESCRIPTIONS.get(lava_intensity, lava_intensity)
    weather_desc = WEATHER_DESCRIPTIONS.get(weather, weather)

    # Extra descriptions pour les scénarios actifs
    extra_parts = []
    if lava_intensity not in ("none",):
        visc = VISCOSITY_DESCRIPTIONS.get(viscosity, "")
        temp = TEMPERATURE_DESCRIPTIONS.get(temperature, "")
        if visc and temp:
            extra_parts.append(f"{visc}, {temp}.")

    if eruption_type != "none":
        erupt = ERUPTION_DESCRIPTIONS.get(eruption_type, "")
        if erupt:
            extra_parts.append(erupt)

    if plume != "none":
        pl = PLUME_DESCRIPTIONS.get(plume, "")
        if pl:
            extra_parts.append(pl)

    extra = " ".join(extra_parts) + (" " if extra_parts else "")

    # Sélection du template
    if template_index is None:
        template = random.choice(_TEMPLATES)
    else:
        template = _TEMPLATES[template_index % len(_TEMPLATES)]

    return template.format(
        camera_desc=camera_desc,
        time_desc=time_desc,
        brightness_desc=brightness_desc,
        lava_desc=lava_desc,
        weather_desc=weather_desc,
        extra=extra,
    )


def build_prompt_from_metadata(
    row: dict,
    *,
    template_index: int | None = 0,
) -> str:
    """Construit un prompt riche depuis une ligne de index.csv.

    Pour l'entraînement, template_index=0 (fixe → cache prompt).
    Pour l'inférence, template_index=None (aléatoire → diversité).
    """
    hour = int(row.get("hour", 12))
    anomaly = float(row.get("anomaly_score", 0.0))
    filename = str(row.get("filename", ""))
    quality = str(row.get("quality_flag", "usable"))

    camera = camera_from_filename(filename)
    tod = time_of_day_from_hour(hour)
    brightness = brightness_from_hour_and_anomaly(hour, anomaly)
    lava_int = lava_intensity_from_anomaly(anomaly)
    slope = slope_from_camera(camera)
    weather = weather_from_quality_and_hour(quality, hour)

    # Dériver viscosité et température depuis anomaly
    if anomaly > 0.4:
        viscosity, temperature = "low", "high"
    elif anomaly > 0.2:
        viscosity, temperature = "medium", "moderate"
    elif anomaly > 0.05:
        viscosity, temperature = "high", "low"
    else:
        viscosity, temperature = "medium", "low"

    # Type d'éruption
    eruption_type = "effusive" if anomaly > 0.35 else "none"

    # Panache
    if anomaly > 0.5:
        plume = "medium"
    elif anomaly > 0.3:
        plume = "low"
    else:
        plume = "none"

    return build_rich_prompt(
        camera=camera,
        time_of_day=tod,
        brightness=brightness,
        lava_intensity=lava_int,
        slope=slope,
        weather=weather,
        viscosity=viscosity,
        temperature=temperature,
        eruption_type=eruption_type,
        plume=plume,
        template_index=template_index,
    )


# ═══════════════════════════════════════════════════════════════
# Extraction de métadonnées (fonctions utilitaires partagées)
# ═══════════════════════════════════════════════════════════════

def camera_from_filename(filename: str) -> str:
    fn = filename.lower()
    if fn.startswith("suki"):
        return "Suki"
    elif fn.startswith("kalor"):
        return "Kalor"
    elif fn.startswith("kali"):
        return "Kali"
    return "unknown"


def time_of_day_from_hour(hour: int) -> str:
    if 6 <= hour < 10:
        return "early_morning"
    elif 10 <= hour < 14:
        return "midday"
    elif 14 <= hour < 18:
        return "afternoon"
    elif 18 <= hour < 21:
        return "dusk"
    return "night"


def brightness_from_hour_and_anomaly(hour: int, anomaly: float) -> str:
    if 6 <= hour < 18:
        return "bright_with_incandescence" if anomaly > 0.3 else "daylight"
    if anomaly > 0.3:
        return "incandescent_glow"
    elif anomaly > 0.1:
        return "dim_glow"
    return "dark"


def lava_intensity_from_anomaly(anomaly: float) -> str:
    if anomaly > 0.5:
        return "very_high"
    elif anomaly > 0.3:
        return "high"
    elif anomaly > 0.15:
        return "moderate"
    elif anomaly > 0.05:
        return "low"
    return "none"


def slope_from_camera(camera: str) -> str:
    return {
        "Suki": "30deg_south",
        "Kalor": "25deg_west",
        "Kali": "35deg_east",
    }.get(camera, "30deg_unknown")


def weather_from_quality_and_hour(quality: str, hour: int) -> str:
    if quality == "cloudy":
        return "overcast"
    if quality == "dark":
        return "clear_night" if hour >= 18 or hour < 6 else "hazy"
    return "clear"


# ═══════════════════════════════════════════════════════════════
# Augmentation couleur physiquement motivée
# ═══════════════════════════════════════════════════════════════

def apply_physics_color(
    img_gray: Image.Image,
    hour: int,
    anomaly: float,
    *,
    jitter_strength: float = 0.12,
) -> Image.Image:
    """Applique une coloration physiquement motivée sur des images en niveaux de gris.

    Enseigne au modèle les associations couleur ↔ condition :
      - Jour : tons neutres/chauds naturels
      - Nuit + activité : lueur orange-rouge volcanique
      - Nuit + calme : tons bleu-sombre froids
      - Crépuscule : tons dorés/ambrés
      - Matin : lumière chaude douce

    Args:
        img_gray: Image PIL en niveaux de gris.
        hour: Heure de capture (0-23).
        anomaly: Score d'anomalie (0.0-1.0).
        jitter_strength: Force de la variation aléatoire de couleur.

    Returns:
        Image PIL RGB avec coloration condition-appropriée.
    """
    arr = np.array(img_gray, dtype=np.float32) / 255.0
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)  # Assurer 2D

    # Multiplicateurs de base par canal
    if 6 <= hour < 10:          # Matin
        r_m, g_m, b_m = 1.08, 1.02, 0.88
    elif 10 <= hour < 14:       # Midi
        r_m, g_m, b_m = 1.02, 1.00, 0.95
    elif 14 <= hour < 18:       # Après-midi
        r_m, g_m, b_m = 1.06, 0.98, 0.87
    elif 18 <= hour < 21:       # Crépuscule
        r_m, g_m, b_m = 1.20, 0.82, 0.60
    else:                       # Nuit
        if anomaly > 0.3:
            # Lueur volcanique forte
            glow = min(anomaly * 1.8, 1.0)
            r_m = 0.30 + 0.70 * glow
            g_m = 0.15 + 0.35 * glow
            b_m = 0.10
        elif anomaly > 0.1:
            # Lueur faible
            r_m, g_m, b_m = 0.45, 0.30, 0.20
        else:
            # Nuit bleu sombre
            r_m, g_m, b_m = 0.25, 0.30, 0.45

    # Jitter aléatoire pour la diversité
    r_j = 1.0 + random.uniform(-jitter_strength, jitter_strength)
    g_j = 1.0 + random.uniform(-jitter_strength, jitter_strength)
    b_j = 1.0 + random.uniform(-jitter_strength, jitter_strength)

    r = np.clip(arr * r_m * r_j, 0, 1)
    g = np.clip(arr * g_m * g_j, 0, 1)
    b = np.clip(arr * b_m * b_j, 0, 1)

    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


# ═══════════════════════════════════════════════════════════════
# Listes de valeurs valides (pour l'UI)
# ═══════════════════════════════════════════════════════════════

CAMERAS = ["Suki", "Kalor", "Kali"]
TIMES_OF_DAY = ["early_morning", "midday", "afternoon", "dusk", "night"]
BRIGHTNESSES = ["daylight", "bright_with_incandescence", "incandescent_glow", "dim_glow", "dark"]
LAVA_INTENSITIES = ["none", "low", "moderate", "high", "very_high"]
WEATHERS = ["clear", "overcast", "hazy", "clear_night"]
VISCOSITIES = ["low", "medium", "high"]
TEMPERATURES = ["low", "moderate", "high", "extreme"]
ERUPTION_TYPES = ["none", "effusive", "explosive", "phreatic"]
PLUMES = ["none", "low", "medium", "high"]
