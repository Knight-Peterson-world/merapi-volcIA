"""
attention_maps.py — Cartes d'attention DINOv2 pour l'interprétabilité.

Méthode : Attention du dernier bloc transformer, moyennée sur toutes les têtes.
          Le token CLS "regarde" les patches les plus discriminants →
          la distribution d'attention identifie les zones actives.

Pas de Grad-CAM : DINOv2 est utilisé comme extracteur figé (sans classifieur
supervisé attaché). Grad-CAM nécessiterait un backprop sur une loss de classification
qui n'existe pas dans ce contexte non-supervisé.

Usage :
    from src.features.attention_maps import get_dino_attention_map, overlay_attention_on_image

    attn = get_dino_attention_map("path/to/image.jpg")  # np.ndarray (16, 16)
    overlay = overlay_attention_on_image(img_gray, attn)  # np.ndarray BGR
    metrics = zone_metrics(attn)  # dict avec surface active, centroïde, etc.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
from pathlib import Path

# Forcer l'implémentation Python de protobuf et désactiver TF dans transformers
# pour éviter les conflits protobuf/tensorflow sur conda base.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import numpy as np

logger = logging.getLogger("attention_maps")

# Taille de patch DINOv2-small : 14px → 224/14 = 16 patches par côté
DINO_MODEL_NAME = "facebook/dinov2-small"
DINO_INPUT_SIZE = 224
DINO_PATCH_SIZE = 14
DINO_N_PATCHES = DINO_INPUT_SIZE // DINO_PATCH_SIZE  # 16


# ─── Protection timeout PIL ──────────────────────────────────────────────

@contextlib.contextmanager
def _pil_timeout(seconds: int = 2):
    """
    Context manager qui interrompt toute opération PIL si elle dépasse `seconds`
    secondes. Utilise SIGALRM (Unix/macOS uniquement — no-op transparent sur Windows).

    Pourquoi SIGALRM et pas threading.Timer ?
    PIL délègue le décodage JPEG à libjpeg (code C). Un thread Python ne peut pas
    interrompre du code C bloqué — seul un signal OS-level le peut. SIGALRM est
    envoyé par le noyau directement au processus, ce qui garantit l'interruption
    même si Python est dans une extension C (libjpeg, libtiff, etc.).

    Contrainte : SIGALRM ne fonctionne que depuis le thread principal. Si appelé
    depuis un thread secondaire, le yield s'exécute sans timeout (dégradé gracieux).
    """
    if not hasattr(signal, "SIGALRM"):
        # Windows : SIGALRM n'existe pas, on continue sans timeout
        yield
        return

    # Vérifier que nous sommes dans le thread principal (SIGALRM ne peut être
    # installé que depuis le thread principal de CPython).
    import threading
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"PIL image load bloqué depuis {seconds}s — image ignorée")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)                            # annuler l'alarme
        signal.signal(signal.SIGALRM, old_handler) # restaurer le handler précédent


# ─── Chargement robuste d'images ──────────────────────────────────────────

def safe_load_image(image_input) -> "PILImage.Image | None":
    """
    Charge une image de manière robuste depuis un chemin ou un array numpy.

    Gère les cas problématiques courants :
      - JPEGs tronqués (fin de fichier prématurée)
      - Progressive JPEGs mal formés
      - Fichiers avec mauvais magic bytes malgré l'extension .jpg
      - Images RGBA / P / CMYK / LAB → converties en RGB
      - Métadonnées EXIF de rotation (ImageOps.exif_transpose)

    La cause racine du warning "cannot identify image file" est que
    PIL.Image.open() est LAZY : il lit uniquement l'en-tête. Le vrai
    décodage des pixels (et donc l'erreur) intervient lors du premier accès
    aux pixels (convert/load/resize). C'est pourquoi on force .load()
    immédiatement pour détecter les fichiers corrompus dès cette fonction.

    Returns:
        PIL.Image en mode "RGB" prêt pour DINOv2, ou None si non récupérable.
    """
    from PIL import Image as PILImage, ImageFile, ImageOps, UnidentifiedImageError

    # Autoriser la lecture des JPEGs tronqués (EOF prématuré fréquent sur les
    # images de surveillance réseau). Sans ce flag, PIL lève une exception même
    # si 90% des données sont intactes.
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    try:
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                logger.debug("Image introuvable : %s", path)
                return None

            # Vérifier la taille minimale du fichier (< 1 KB = probablement corrompu)
            if path.stat().st_size < 1024:
                logger.debug("Fichier trop petit (%d B), probablement corrompu : %s",
                             path.stat().st_size, path.name)
                return None

            pil_img = PILImage.open(path)

        elif isinstance(image_input, np.ndarray):
            # Construire une PIL Image depuis un array numpy
            arr = image_input
            if arr.ndim == 2:
                pil_img = PILImage.fromarray(arr, mode="L")
            elif arr.ndim == 3 and arr.shape[2] == 4:
                pil_img = PILImage.fromarray(arr.astype(np.uint8), mode="RGBA")
            else:
                pil_img = PILImage.fromarray(arr.astype(np.uint8))
        else:
            logger.debug("safe_load_image: type non supporté %s", type(image_input))
            return None

        # Forcer le décodage complet des pixels MAINTENANT (pas lazy).
        # _pil_timeout(2) : SIGALRM après 2s si libjpeg/libtiff bloque en C
        # (cas : JPEG progressif malformé → boucle infinie dans le décodeur C).
        with _pil_timeout(2):
            pil_img.load()

        # Corriger l'orientation EXIF (rotation, miroir) si présente.
        # Évite que des images "correctes visuellement" soient retournées
        # dans le pipeline (les caméras de surveillance encodent souvent la
        # rotation dans l'EXIF sans l'appliquer aux pixels).
        try:
            pil_img = ImageOps.exif_transpose(pil_img)
        except Exception:
            pass  # exif_transpose peut échouer sur des EXIF mal formés → ignorer

        # Normaliser vers RGB quelle que soit la source :
        # L (grayscale), P (palette), RGBA, CMYK, LAB, HSV → RGB
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        return pil_img

    except (UnidentifiedImageError, OSError, IOError, SyntaxError) as exc:
        _name = Path(image_input).name if isinstance(image_input, (str, Path)) else "array"
        logger.debug("safe_load_image: image non lisible (%s) — %s", _name, exc)
        return None
    except Exception as exc:
        _name = Path(image_input).name if isinstance(image_input, (str, Path)) else "array"
        logger.warning("safe_load_image: erreur inattendue (%s) — %s", _name, exc)
        return None


# ─── Chargement du modèle (singleton en cache) ────────────────────────────

_dino_model = None
_dino_processor = None
_dino_load_failed = False   # True après le premier échec → ne pas retenter


def _load_dino():
    """Charge DINOv2 une seule fois. Si ça échoue, ne retente jamais (évite le spam de logs)."""
    global _dino_model, _dino_processor, _dino_load_failed
    if _dino_model is not None:
        return _dino_model, _dino_processor
    if _dino_load_failed:
        return None, None  # déjà échoué, ne pas retenter
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        _dino_processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
        _dino_model = AutoModel.from_pretrained(
            DINO_MODEL_NAME, output_attentions=True
        )
        _dino_model.eval()
        logger.info("DINOv2-small chargé (%s)", DINO_MODEL_NAME)
        return _dino_model, _dino_processor
    except Exception as exc:
        _dino_load_failed = True
        logger.warning(
            "DINOv2 non disponible (sera ignoré) : %s\n"
            "  → Conseil : pip install 'protobuf>=3.20,<4' pour corriger le conflit protobuf.",
            exc,
        )
        return None, None


# ─── Extraction d'attention ───────────────────────────────────────────────

def get_dino_attention_map(
    image_input,
    layer: int = -1,
) -> np.ndarray:
    """
    Retourne la carte d'attention DINOv2 normalisée [0, 1].

    Résolution de sortie : (DINO_N_PATCHES, DINO_N_PATCHES) = (16, 16).
    Pour overlay, upscaler avec cv2.resize vers la taille de l'image originale.

    Args:
        image_input: chemin (str/Path) ou np.ndarray (grayscale ou RGB).
        layer: indice du bloc transformer à utiliser (-1 = dernier, recommandé).

    Returns:
        np.ndarray float32 shape (16, 16), valeurs dans [0, 1].
        Retourne un tableau de zéros si le modèle n'est pas disponible.
    """
    model, processor = _load_dino()
    if model is None:
        return np.zeros((DINO_N_PATCHES, DINO_N_PATCHES), dtype=np.float32)

    try:
        import torch

        pil_img = safe_load_image(image_input)
        if pil_img is None:
            _name = Path(image_input).name if isinstance(image_input, (str, Path)) else "array"
            logger.warning("get_dino_attention_map: image invalide ignorée — %s", _name)
            return np.zeros((DINO_N_PATCHES, DINO_N_PATCHES), dtype=np.float32)

        pil_img = pil_img.resize((DINO_INPUT_SIZE, DINO_INPUT_SIZE))

        # ── Inférence ─────────────────────────────────────────────────
        inputs = processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        # outputs.attentions : tuple de tenseurs (1, n_heads, n_tokens, n_tokens)
        # n_tokens = 1 CLS + 16*16 = 257
        attn_layer = outputs.attentions[layer]  # (1, heads, 257, 257)
        attn_heads = attn_layer[0]              # (heads, 257, 257)

        # Moyenne sur toutes les têtes → (257, 257)
        attn_mean = attn_heads.mean(dim=0)

        # Attention du token CLS vers les patches (exclut CLS→CLS)
        cls_attn = attn_mean[0, 1:].numpy()    # (256,) = 16×16 patches

        # Reshape et normalisation
        cls_attn = cls_attn.reshape(DINO_N_PATCHES, DINO_N_PATCHES)
        vmin, vmax = cls_attn.min(), cls_attn.max()
        if vmax > vmin:
            cls_attn = (cls_attn - vmin) / (vmax - vmin)
        else:
            cls_attn = np.zeros_like(cls_attn)

        return cls_attn.astype(np.float32)

    except Exception as exc:
        logger.warning("get_dino_attention_map failed : %s", exc)
        return np.zeros((DINO_N_PATCHES, DINO_N_PATCHES), dtype=np.float32)


def get_dino_patch_features(
    image_input,
) -> np.ndarray:
    """
    Extrait les features patch-level DINOv2 (pour PatchCore).

    Returns:
        np.ndarray float32 shape (256, 384) — 256 patches, dim=384.
        Retourne un tableau de ZÉROS si le modèle ou l'image est indisponible.
        L'appelant (PatchCoreDetector.fit) filtre les features nulles via
        ``patches.max() != 0`` — les images invalides sont donc silencieusement
        exclues du coreset sans crasher le pipeline.
    """
    model, processor = _load_dino()
    if model is None:
        return np.zeros((DINO_N_PATCHES * DINO_N_PATCHES, 384), dtype=np.float32)

    try:
        import torch

        pil_img = safe_load_image(image_input)
        if pil_img is None:
            _name = Path(image_input).name if isinstance(image_input, (str, Path)) else "array"
            logger.warning("get_dino_patch_features: image invalide ignorée — %s", _name)
            return np.zeros((DINO_N_PATCHES * DINO_N_PATCHES, 384), dtype=np.float32)

        pil_img = pil_img.resize((DINO_INPUT_SIZE, DINO_INPUT_SIZE))
        inputs = processor(images=pil_img, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        # last_hidden_state : (1, 257, 384) — CLS + 256 patches
        patch_features = outputs.last_hidden_state[0, 1:].numpy()  # (256, 384)
        return patch_features.astype(np.float32)

    except Exception as exc:
        _name = Path(image_input).name if isinstance(image_input, (str, Path)) else "array"
        logger.warning("get_dino_patch_features failed (%s) : %s", _name, exc)
        return np.zeros((DINO_N_PATCHES * DINO_N_PATCHES, 384), dtype=np.float32)


# ─── Overlay visuel ───────────────────────────────────────────────────────

def overlay_attention_on_image(
    img_gray: np.ndarray,
    attn_map: np.ndarray,
    alpha: float = 0.45,
    colormap: int | None = None,
) -> np.ndarray:
    """
    Superpose la carte d'attention sur l'image originale.

    Args:
        img_gray: image grayscale uint8 ou float32, shape (H, W).
        attn_map: carte d'attention float32 [0,1], shape (h, w) (upscalée automatiquement).
        alpha: poids de la heatmap (0 = image seule, 1 = heatmap seule).
        colormap: colormap OpenCV (défaut: COLORMAP_JET).

    Returns:
        np.ndarray uint8 BGR shape (H, W, 3).
    """
    try:
        import cv2 as _cv2

        if colormap is None:
            colormap = _cv2.COLORMAP_JET

        H, W = img_gray.shape[:2]
        img_u8 = _to_uint8(img_gray)
        img_bgr = _cv2.cvtColor(img_u8, _cv2.COLOR_GRAY2BGR) if img_u8.ndim == 2 else img_u8

        # Upscale attention map vers la taille de l'image
        attn_resized = _cv2.resize(attn_map, (W, H), interpolation=_cv2.INTER_LINEAR)
        attn_u8 = (attn_resized * 255).astype(np.uint8)
        heatmap = _cv2.applyColorMap(attn_u8, colormap)

        return _cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)

    except ImportError:
        # Fallback sans OpenCV : retourner image RGB avec heatmap matplotlib
        return _overlay_matplotlib_fallback(img_gray, attn_map, alpha)


def _overlay_matplotlib_fallback(
    img_gray: np.ndarray,
    attn_map: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fallback overlay sans OpenCV, utilise matplotlib."""
    import matplotlib.cm as cm

    H, W = img_gray.shape[:2]
    img_u8 = _to_uint8(img_gray)

    # Upscale avec PIL
    from PIL import Image as PILImage
    attn_pil = PILImage.fromarray((attn_map * 255).astype(np.uint8)).resize((W, H))
    attn_np = np.array(attn_pil) / 255.0

    colorized = (cm.jet(attn_np)[:, :, :3] * 255).astype(np.uint8)
    img_rgb = np.stack([img_u8, img_u8, img_u8], axis=2)
    blended = (img_rgb * (1 - alpha) + colorized * alpha).astype(np.uint8)
    return blended


# ─── Métriques de zone active ─────────────────────────────────────────────

def zone_metrics(attn_map: np.ndarray, threshold: float = 0.70) -> dict[str, float]:
    """
    Quantifie la zone active à partir de la carte d'attention.

    Ces métriques sont stockables dans index.csv et traçables dans le temps.

    Args:
        attn_map: carte d'attention float32 [0,1], shape quelconque.
        threshold: seuil pour définir la zone "active" (top 30% par défaut).

    Returns:
        dict avec :
          - active_surface_pct : % de la surface détectée comme active
          - centroid_x         : position x normalisée [0,1] du centroïde
          - centroid_y         : position y normalisée [0,1] du centroïde
          - max_attention      : valeur max de la carte (intensité de la zone peak)
          - n_active_patches   : nombre de patches au-dessus du seuil
    """
    active_mask = attn_map > threshold
    n_active = int(active_mask.sum())
    total = attn_map.size

    if n_active > 0:
        ys, xs = np.where(active_mask)
        centroid_x = float(xs.mean()) / attn_map.shape[1]
        centroid_y = float(ys.mean()) / attn_map.shape[0]
    else:
        centroid_x = 0.5
        centroid_y = 0.5

    return {
        "active_surface_pct": float(n_active / total * 100),
        "centroid_x":          centroid_x,
        "centroid_y":          centroid_y,
        "max_attention":       float(attn_map.max()),
        "n_active_patches":    n_active,
    }


# ─── Worker multiprocessing pour l'audit (doit être picklable) ─────────────

def _check_image_worker(path_str: str) -> tuple[str, bool]:
    """
    Teste si une image est lisible. Exécuté dans un sous-process indépendant.

    Doit être défini au niveau du module (top-level) pour être picklable par
    ProcessPoolExecutor. Ne pas déplacer dans une classe ou une fonction imbriquée.

    Deux niveaux de protection contre les blocages :
      1. _pil_timeout(2) : SIGALRM après 2s (niveau intra-process)
      2. future.result(timeout=5) dans le process parent (niveau inter-process)
    """
    try:
        img = safe_load_image(path_str)
        return path_str, (img is not None)
    except Exception:
        return path_str, False


# ─── Audit du dataset ─────────────────────────────────────────────────────

def audit_image_dataset(
    raw_dir: str | Path,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    verbose: bool = True,
    n_workers: int = 4,
    per_image_timeout: float = 5.0,
    on_invalid: str = "ignore",
    quarantine_dir: str | Path | None = None,
) -> dict:
    """
    Scanne un dossier récursivement et teste chaque image. 100% anti-freeze.

    Architecture anti-blocage (3 couches) :
      1. safe_load_image() vérifie taille, LOAD_TRUNCATED_IMAGES, exif_transpose
      2. _pil_timeout(2) : SIGALRM OS-level si PIL bloque > 2s dans du code C
      3. ProcessPoolExecutor : chaque image dans un sous-process isolé ;
         future.result(timeout=per_image_timeout) abandonne après N secondes
         sans tuer le process principal

    Args:
        raw_dir            : dossier racine (ex: 'data/raw/')
        extensions         : extensions image à tester (case-insensitive)
        verbose            : affiche la progression en temps réel (flush=True)
        n_workers          : nombre de sous-process parallèles (défaut: 4)
        per_image_timeout  : secondes max par image avant abandon (défaut: 5.0)
        on_invalid         : que faire des images invalides :
                             'ignore' (défaut) — rien
                             'move'   — déplacer vers quarantine_dir
                             'delete' — supprimer définitivement
        quarantine_dir     : dossier de quarantaine (on_invalid='move').
                             Défaut : <raw_dir>/../quarantine/

    Returns:
        dict avec :
          n_total      : fichiers trouvés
          n_ok         : images valides
          n_failed     : images rejetées
          failed_paths : liste des chemins problématiques

    Exemples CLI :
        # Audit simple
        python -c "
        from src.features.attention_maps import audit_image_dataset
        r = audit_image_dataset('data/raw/')
        print(r['n_failed'], 'invalides sur', r['n_total'])
        "

        # Audit + déplacement des invalides en quarantaine
        python -c "
        from src.features.attention_maps import audit_image_dataset
        audit_image_dataset('data/raw/', on_invalid='move')
        "
    """
    import shutil
    from concurrent.futures import ProcessPoolExecutor, TimeoutError as _FutureTimeout, as_completed

    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        print(f"[audit] ERREUR : dossier introuvable — {raw_dir}", flush=True)
        return {"n_total": 0, "n_ok": 0, "n_failed": 0, "failed_paths": []}

    # ── Collecte des fichiers ────────────────────────────────────────────
    all_files = sorted(
        p for p in raw_dir.rglob("*")
        if p.suffix.lower() in extensions and p.is_file()
    )
    n_total = len(all_files)
    print(f"[audit] {n_total} images trouvées dans {raw_dir}", flush=True)

    if n_total == 0:
        return {"n_total": 0, "n_ok": 0, "n_failed": 0, "failed_paths": []}

    failed_paths: list[str] = []
    n_done = 0

    # ── Scan parallèle avec sous-process isolés ──────────────────────────
    # ProcessPoolExecutor isole chaque image dans un process séparé.
    # Si PIL bloque (JPEG progressif corrompu, libjpeg boucle infinie) :
    #   - dans le sous-process : _pil_timeout(2) lève TimeoutError via SIGALRM
    #   - dans le process parent : future.result(timeout=per_image_timeout)
    #     lève concurrent.futures.TimeoutError si le sous-process ne répond plus
    # Le process zombie est nettoyé à la fermeture du context manager (executor).
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_path = {
            executor.submit(_check_image_worker, str(p)): p
            for p in all_files
        }

        for future in as_completed(future_to_path):
            fpath = future_to_path[future]
            n_done += 1
            ok = False

            try:
                _, ok = future.result(timeout=per_image_timeout)
            except _FutureTimeout:
                ok = False
                print(f"[audit] TIMEOUT   [{n_done}/{n_total}] {fpath.name}", flush=True)
            except Exception as exc:
                ok = False
                print(f"[audit] ERREUR    [{n_done}/{n_total}] {fpath.name} — {exc}", flush=True)

            if not ok:
                failed_paths.append(str(fpath))
                if verbose:
                    print(f"[audit] INVALIDE  [{n_done}/{n_total}] {fpath}", flush=True)
            elif verbose and (n_done % 200 == 0 or n_done == n_total):
                print(
                    f"[audit] {n_done}/{n_total} — "
                    f"{len(failed_paths)} invalides jusqu'ici",
                    flush=True,
                )

    n_failed = len(failed_paths)
    n_ok = n_total - n_failed
    print(
        f"\n[audit] Terminé — {n_ok} valides  {n_failed} invalides  (total {n_total})",
        flush=True,
    )

    # ── Nettoyage optionnel des images invalides ─────────────────────────
    if on_invalid in ("move", "delete") and n_failed > 0:
        if on_invalid == "move":
            qdir = Path(quarantine_dir) if quarantine_dir else raw_dir.parent / "quarantine"
            qdir.mkdir(parents=True, exist_ok=True)
            print(f"[audit] Déplacement de {n_failed} images → {qdir}", flush=True)
            for p_str in failed_paths:
                p = Path(p_str)
                dst = qdir / p.name
                # Éviter les collisions de noms en préfixant avec le sous-dossier
                if dst.exists():
                    dst = qdir / f"{p.parent.name}_{p.name}"
                try:
                    shutil.move(str(p), str(dst))
                except OSError as exc:
                    print(f"[audit] Impossible de déplacer {p.name} : {exc}", flush=True)

        elif on_invalid == "delete":
            print(f"[audit] Suppression de {n_failed} images invalides …", flush=True)
            for p_str in failed_paths:
                try:
                    Path(p_str).unlink(missing_ok=True)
                except OSError as exc:
                    print(f"[audit] Impossible de supprimer {Path(p_str).name} : {exc}", flush=True)

    if n_failed > 0:
        logger.warning(
            "[audit] %d images invalides. Relancez avec on_invalid='move' pour les "
            "déplacer en quarantaine, ou on_invalid='delete' pour les supprimer.",
            n_failed,
        )

    return {
        "n_total": n_total,
        "n_ok": n_ok,
        "n_failed": n_failed,
        "failed_paths": failed_paths,
    }


# ─── Helper ───────────────────────────────────────────────────────────────

def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if img.max() <= 1.0:
        return (img * 255).astype(np.uint8)
    return img.astype(np.uint8)
