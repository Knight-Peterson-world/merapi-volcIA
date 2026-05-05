"""
streamlit_app.py — VolcIA : outil d'analyse et de simulation volcanologique.

Application Streamlit pour :
  - Exploration des images du Merapi (10 ans, 2014–2024)
  - Détection d'anomalies et scores baseline
  - Génération d'images par IA (text-to-image) — prototype
  - Simulation simplifiée d'écoulements volcaniques
  - Statistiques et documentation du projet

Lancement (depuis le dossier merapi_anomaly/) :
    /opt/anaconda3/bin/streamlit run app/streamlit_app.py

  ⚠ Ne pas utiliser la commande `streamlit` seule si python et pip ne
  pointent pas vers le même environnement (risque Python 3.12 vs 3.10).
"""

from __future__ import annotations

import os
# Éviter les imports TF inutiles (accélère le démarrage).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import sys
from pathlib import Path

# ----------------------------------------------------------
# Racine du projet
# ----------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import plotly.graph_objects as go
import plotly.express as px

# --- Compatibilité Streamlit : use_container_width pour st.image ---
# Ajouté en 1.38+ pour st.image ; avant ça le paramètre s'appelait use_column_width.
import inspect as _inspect
_img_params = _inspect.signature(st.image).parameters
_IMG_WIDTH_KW = "use_container_width" if "use_container_width" in _img_params else "use_column_width"

try:
    import seaborn as sns
except ImportError:
    sns = None

from src.utils import load_config, get_index_path, parse_filename_datetime
from src.preprocessing import MerapiPreprocessor

# ----------------------------------------------------------
# Configuration globale Streamlit
# ----------------------------------------------------------
st.set_page_config(
    page_title="VolcIA — Merapi",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------
# CSS personnalisé
# ----------------------------------------------------------
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    [data-testid="stSidebar"] { min-width: 280px; }
    [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.1);
        padding: 0.5rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Noms de pages — regroupées par logique métier
# [0-2]  Navigation + Exploration
# [3-6]  Détection + Analyse volcanique
# [7-9]  IA (génération + reconstruction + simulation)
# [10]   Statistiques
# [11]   À propos (TOUJOURS EN DERNIER)
PAGES = [
    "🏠 Accueil",                       # [0]
    "🔍 Exploration",                    # [1]
    "🖼️ Galerie temporelle",            # [2]
    "🚨 Anomalies",                      # [3]
    "🔬 DINOv2 + PatchCore",            # [4]
    "📊 Timeline PatchCore",             # [5]
    "⚡ Early Warning",                  # [6]
    "🌋 Analyse volcanique avancée",    # [7]
    "🧪 Analyse avancée",               # [8]
    "🌋 Simulation écoulements",         # [9]
    "📖 À propos",                       # [10] ← TOUJOURS EN DERNIER
]


# ----------------------------------------------------------
# Chargement des données (mis en cache)
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_index() -> pd.DataFrame:
    """Charge l'index CSV et enrichit les colonnes temporelles."""
    config = load_config()
    index_path = get_index_path(config)
    demo_path = PROJECT_ROOT / "data" / "index" / "index_demo.csv"

    if index_path.exists():
        df = pd.read_csv(index_path, dtype=str, na_values=["", "None", "nan"])
    elif demo_path.exists():
        df = pd.read_csv(demo_path, dtype=str, na_values=["", "None", "nan"])
    else:
        return pd.DataFrame()

    # Conversions
    for c in ["year", "month", "day", "hour", "minute", "second"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fallback : extraire date depuis le nom de fichier
    if "filename" in df.columns:
        needs_fill = df["day"].isna() | df["hour"].isna()
        if needs_fill.any():
            parsed = df.loc[needs_fill, "filename"].apply(
                lambda fn: parse_filename_datetime(str(fn)) or {}
            )
            for col in ["day", "hour", "minute", "second"]:
                vals = pd.to_numeric(
                    parsed.apply(lambda d, c=col: d.get(c)), errors="coerce"
                )
                mask = df[col].isna() & needs_fill
                df.loc[mask, col] = vals.loc[mask]
                df[col] = pd.to_numeric(df[col], errors="coerce")

    df["downloaded"] = df["downloaded"].map(lambda x: str(x).lower() == "true")
    df["file_size_bytes"] = pd.to_numeric(df.get("file_size_bytes", pd.Series(dtype=float)), errors="coerce")
    df["anomaly_score"] = pd.to_numeric(df.get("anomaly_score", pd.Series(dtype=float)), errors="coerce")

    # ── Conversion patchcore_score en numérique ─────────────────────────
    if "patchcore_score" in df.columns:
        df["patchcore_score"] = pd.to_numeric(df["patchcore_score"], errors="coerce")

    # ── Enrichissement patchcore depuis patchcore_scores.csv ───────────
    # (couvre les nouvelles images 2019+ qui n'ont pas encore de score dans l'index)
    scores_csv = PROJECT_ROOT / "outputs" / "scores" / "patchcore_scores.csv"
    if scores_csv.exists() and "filename" in df.columns:
        try:
            pc = pd.read_csv(scores_csv)
            if not pc.empty and "patchcore_score" in pc.columns:
                pc["patchcore_score"] = pd.to_numeric(pc["patchcore_score"], errors="coerce")
                if "patchcore_score" not in df.columns:
                    df["patchcore_score"] = np.nan
                missing_mask = df["patchcore_score"].isna()
                if missing_mask.any():
                    pc_map = pc.set_index("filename")["patchcore_score"].to_dict()
                    df.loc[missing_mask, "patchcore_score"] = (
                        df.loc[missing_mask, "filename"].map(pc_map)
                    )
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
            pass  # fichier en cours d'écriture ou corrompu → continuer sans scores

    # ── Peupler anomaly_score depuis patchcore_score quand absent/NaN ─────
    # Permet aux pages qui lisent anomaly_score de voir les données PatchCore
    if "patchcore_score" in df.columns:
        if "anomaly_score" not in df.columns:
            df["anomaly_score"] = df["patchcore_score"]
        else:
            df["anomaly_score"] = df["anomaly_score"].fillna(df["patchcore_score"])

    # ── Colonnes optionnelles — initialiser si absentes pour éviter KeyError ──
    for _col in ["patchcore_score", "anomaly_score", "quality_flag", "is_night"]:
        if _col not in df.columns:
            df[_col] = np.nan

    # ── Filtre caméra Kalor (inclut les fichiers ech_kalor_Canon_... 2019+) ─
    if "filename" in df.columns:
        df = df[df["filename"].str.lower().str.contains("kalor")].copy()

    # ── Vérification existence réelle des fichiers sur disque ─────────────
    # L'index contient des entrées marquées downloaded=True mais inexistantes
    # sur disque (doublons macOS « 2.jpg », images Suki/autres caméras jamais
    # téléchargées, fichiers déplacés ou supprimés).
    # Cette colonne permet de distinguer « disponible dans l'index » de
    # « réellement accessible pour PatchCore ».
    if "local_path" in df.columns:
        _has_lp = df["local_path"].notna()
        df["on_disk"] = False
        if _has_lp.any():
            df.loc[_has_lp, "on_disk"] = df.loc[_has_lp, "local_path"].apply(
                lambda p: Path(str(p)).exists()
            )
    else:
        df["on_disk"] = False

    return df


@st.cache_data(ttl=300)
def load_scores(year: int, month: int) -> pd.DataFrame:
    """Charge les scores baseline pré-calculés pour un mois."""
    config = load_config()
    scores_path = PROJECT_ROOT / config["paths"]["scores"] / f"baselines_{year}_{month:02d}.csv"
    return safe_read_csv(scores_path)


@st.cache_resource
def _cached_load_volcano_classifier(clf_path_str: str):
    """Charge le VolcanoClassifier une seule fois en mémoire (évite les rechargements inutiles)."""
    from src.models.volcano_classifier import VolcanoClassifier
    return VolcanoClassifier.load(clf_path_str)


def resolve_paths(row: pd.Series, config: dict) -> tuple[Path, Path]:
    """Retourne (raw_path, processed_path) pour une ligne de l'index."""
    local = str(row.get("local_path", ""))
    if not local:
        return Path(), Path()
    # Si le local_path est absolu, l'utiliser directement
    if Path(local).is_absolute():
        raw_path = Path(local)
    else:
        raw_path = PROJECT_ROOT / local
    proc_rel = str(local).replace(
        config["paths"]["data_raw"],
        config["paths"]["data_processed"],
    )
    # Priorité au format PNG
    if Path(proc_rel).is_absolute():
        proc_path = Path(proc_rel)
    else:
        proc_path = PROJECT_ROOT / proc_rel
    png_path = proc_path.with_suffix(".png")
    if png_path.exists():
        proc_path = png_path
    return raw_path, proc_path


def load_image_for_display(path: Path) -> np.ndarray | None:
    """Charge une image pour l'affichage (PNG prioritaire, fallback JPG)."""
    from PIL import Image

    # 1. PNG (format cible après refactoring)
    png_path = path.with_suffix(".png")
    if png_path.exists():
        try:
            return np.array(Image.open(png_path))
        except Exception:
            pass

    # 2. Rétro-compatibilité : .npy (anciens fichiers)
    npy_path = path.with_suffix(".npy")
    if npy_path.exists():
        return np.load(str(npy_path)).astype(np.float32)

    # 3. Fichier original (raw .jpg ou chemin exact)
    if path.exists():
        try:
            return np.array(Image.open(path))
        except Exception:
            return None
    return None


def find_image_path(row: pd.Series) -> Path | None:
    """Retourne le chemin réel de l'image (raw ou processed) à partir d'une ligne d'index.

    Ordre de priorité :
      1. local_path tel quel (absolu ou relatif)
      2. raw  data/raw/{year}/{month:02d}/{filename}
      3. proc data/processed/{year}/{month:02d}/{stem}.png
      4. proc data/processed/{year}/{month:02d}/{filename}
    Retourne None si aucun fichier n'est trouvé.
    """
    filename = str(row.get("filename", ""))
    year = row.get("year")
    month = row.get("month")

    candidates: list[Path] = []

    # 1. local_path dans l'index
    local = str(row.get("local_path", "") or "")
    if local:
        p = Path(local) if Path(local).is_absolute() else PROJECT_ROOT / local
        candidates.append(p)

    # 2. Construire les chemins raw + processed depuis l'année/mois/filename
    if filename and year and month:
        try:
            y, m = int(year), int(month)
            raw_p = PROJECT_ROOT / "data" / "raw" / str(y) / f"{m:02d}" / filename
            stem = Path(filename).stem
            proc_png = PROJECT_ROOT / "data" / "processed" / str(y) / f"{m:02d}" / f"{stem}.png"
            proc_jpg = PROJECT_ROOT / "data" / "processed" / str(y) / f"{m:02d}" / filename
            candidates += [raw_p, proc_png, proc_jpg]
        except (ValueError, TypeError):
            pass

    for p in candidates:
        if p.exists():
            return p
    return None


def safe_read_csv(path: "Path | str", **kwargs) -> pd.DataFrame:
    """Lecture CSV robuste : retourne un DataFrame vide si absent/vide/corrompu."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, **kwargs)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError, UnicodeDecodeError):
        return pd.DataFrame()


# ==============================================================
# Helpers
# ==============================================================

def sidebar_data_filters(df: pd.DataFrame) -> tuple[int, int, pd.DataFrame]:
    """Filtres année/mois/jour/qualité dans la sidebar. Retourne (year, month, df_filtered)."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Filtres données")

    years = sorted(df["year"].dropna().unique().astype(int))
    if not years:
        return 0, 0, pd.DataFrame()

    sel_year = st.sidebar.selectbox("Année", years, index=len(years) - 1, key="filter_year")
    df_year = df[df["year"] == sel_year]

    months = sorted(df_year["month"].dropna().unique().astype(int))
    if not months:
        return sel_year, 0, df_year

    sel_month = st.sidebar.selectbox("Mois", months, index=0,
                                     format_func=lambda m: f"{m:02d}", key="filter_month")
    df_month = df_year[df_year["month"] == sel_month]

    # ── Filtre Jour (optionnel) ──────────────────────────────────────────
    days_available = sorted(df_month["day"].dropna().unique().astype(int))
    if days_available:
        _use_day_filter = st.sidebar.checkbox(
            "Filtrer par jour", value=False, key="filter_use_day",
            help="Affiche uniquement les images d'un jour précis du mois."
        )
        if _use_day_filter:
            sel_day = st.sidebar.selectbox(
                "Jour",
                days_available,
                format_func=lambda d: f"{d:02d}",
                key="filter_day",
            )
            df_month = df_month[df_month["day"] == sel_day]

    quality_flags = df_month["quality_flag"].dropna().unique().tolist()
    if quality_flags:
        sel_quality = st.sidebar.multiselect(
            "Qualité", sorted(quality_flags), default=sorted(quality_flags),
            key="filter_quality",
        )
        if sel_quality:
            df_month = df_month[df_month["quality_flag"].isin(sel_quality) | df_month["quality_flag"].isna()]

    only_dl = st.sidebar.checkbox("Uniquement téléchargées", value=False, key="filter_downloaded")
    if only_dl:
        df_month = df_month[df_month["downloaded"] == True]

    st.sidebar.metric("Images filtrées", len(df_month))
    return sel_year, sel_month, df_month


# ==============================================================
# ==============================================================
# Helpers PatchCore — heatmap & cluster visualization
# ==============================================================

@st.cache_resource(show_spinner=False)
def _load_patchcore_detector():
    """Charge le détecteur PatchCore depuis outputs/models/patchcore.npz."""
    model_path = PROJECT_ROOT / "outputs" / "models" / "patchcore.npz"
    if not model_path.exists():
        print(f"[PatchCore] Fichier introuvable : {model_path}")
        return None
    try:
        from src.models.patchcore_detector import PatchCoreDetector
        # load() est une méthode d'instance — instancier d'abord, puis charger
        return PatchCoreDetector().load(str(model_path))
    except Exception as exc:
        print(f"[PatchCore] Impossible de charger le modèle : {exc}")
        return None


@st.cache_resource(show_spinner=False)
def _load_diffusion_pipeline(lora_path_str: str | None = None):
    """
    Charge le pipeline SD 1.5 img2img UNE SEULE FOIS (cache Streamlit persistant).

    Le cache est invalidé uniquement si lora_path_str change.
    Changer 'strength' ou 'quality_mode' NE recharge PAS le modèle.

    Returns:
        (pipeline, backend_str) — pipeline est None si diffusers indisponible.
    """
    from src.models.diffusion_reconstructor import build_img2img_pipeline
    lora_path = None
    if lora_path_str:
        from pathlib import Path as _Path
        lp = _Path(lora_path_str)
        lora_path = lp if lp.exists() else None
    print(f"[DiffusionPipeline] Chargement SD 1.5 (lora={lora_path_str or 'none'})…")
    pipe, backend = build_img2img_pipeline(lora_path=lora_path, device="auto")
    print(f"[DiffusionPipeline] Prêt — backend={backend}")
    return pipe, backend


def _anomaly_level(score: float) -> str:
    """Retourne le niveau d'anomalie textuel pour un score normalisé [0, 1]."""
    if score < 0.2:
        return "Normal 🟢"
    elif score < 0.4:
        return "Légère 🟡"
    elif score < 0.6:
        return "Modérée 🟠"
    elif score < 0.8:
        return "Forte 🔴"
    return "Critique ⚠️"


# ─── Helper UX : bouton d'aide contextuel ─────────────────────────────────

def help_tooltip(title: str, content: str, key: str) -> None:
    """
    Affiche un bouton ❓ qui ouvre un expander d'aide contextuelle.

    Usage :
        help_tooltip(
            "Score PatchCore",
            "Le score mesure la distance entre les features DINOv2...",
            key="help_patchcore_score",
        )

    Args:
        title   : titre de la section d'aide (ex: "Tableau des anomalies")
        content : texte Markdown (définitions, métriques, contexte)
        key     : clé Streamlit unique (évite les conflits widget)
    """
    with st.expander(f"❓ {title}", expanded=False):
        st.markdown(content)


def build_heatmap_figure(
    patch_map: np.ndarray,
    img_array: np.ndarray,
    opacity: float = 0.45,
    colorscale: str = "Hot",
) -> go.Figure:
    """
    Construit une figure Plotly : image originale + heatmap PatchCore superposée.

    Args:
        patch_map:  Carte (H_p, W_p) des scores normalisés [0, 1].
        img_array:  Image RGB numpy (H, W, 3).
        opacity:    Opacité de la heatmap [0, 1].
        colorscale: Colorscale Plotly (Hot, Viridis, RdYlBu_r…).

    Returns:
        Figure Plotly prête à afficher avec st.plotly_chart.
    """
    import base64
    import io
    from PIL import Image as _PILImg
    from scipy.ndimage import zoom

    H, W = img_array.shape[:2]
    scale_h = H / patch_map.shape[0]
    scale_w = W / patch_map.shape[1]
    patch_full = zoom(patch_map.astype(float), (scale_h, scale_w), order=1)

    # customdata riche pour le hover
    y_idx, x_idx = np.indices(patch_full.shape)
    pi_h = (y_idx / scale_h).astype(int).clip(0, patch_map.shape[0] - 1)
    pi_w = (x_idx / scale_w).astype(int).clip(0, patch_map.shape[1] - 1)
    custom = np.stack(
        [patch_full, x_idx.astype(float), y_idx.astype(float),
         pi_h.astype(float), pi_w.astype(float)],
        axis=-1,
    )

    # Encode l'image en base64 pour fond Plotly
    buf = io.BytesIO()
    _PILImg.fromarray(img_array).save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    fig = go.Figure()
    fig.add_layout_image(
        source=f"data:image/jpeg;base64,{img_b64}",
        xref="x", yref="y",
        x=0, y=H, sizex=W, sizey=H,
        sizing="stretch", layer="below",
    )
    fig.add_trace(go.Heatmap(
        z=patch_full,
        colorscale=colorscale,
        opacity=opacity,
        zmin=0.0, zmax=1.0,
        customdata=custom,
        hovertemplate=(
            "<b>%{customdata[0]:.3f}</b><br>"
            "Pixel : (%{customdata[1]:.0f}, %{customdata[2]:.0f})<br>"
            "Patch : [%{customdata[3]:.0f}, %{customdata[4]:.0f}]<extra></extra>"
        ),
        showscale=True,
        colorbar=dict(title="Score", thickness=14, len=0.75),
    ))
    fig.update_layout(
        xaxis=dict(range=[0, W], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, H], showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x"),
        margin=dict(l=0, r=50, t=30, b=0),
        height=520,
        plot_bgcolor="black",
    )
    return fig


def build_cluster_figure(
    coreset: np.ndarray,
    query_features: np.ndarray | None = None,
    patch_scores_query: np.ndarray | None = None,
) -> go.Figure:
    """
    Projette le coreset (mémoire normale) et les features d'une image via PCA 2D.

    Args:
        coreset:            (N_coreset, D) features du coreset.
        query_features:     (N_patches, D) features de l'image courante.
        patch_scores_query: (N_patches,) scores normalisés des patches de l'image.

    Returns:
        Figure Plotly scatter interactive.
    """
    from sklearn.decomposition import PCA

    n_coreset = len(coreset)
    all_feats = coreset if query_features is None else np.vstack([coreset, query_features])

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(all_feats)
    var = pca.explained_variance_ratio_

    fig = go.Figure()

    # Coreset (mémoire normale)
    c = coords[:n_coreset]
    fig.add_trace(go.Scatter(
        x=c[:, 0], y=c[:, 1],
        mode="markers",
        name="Mémoire normale (coreset)",
        marker=dict(color="#2980b9", size=4, opacity=0.45),
        hovertemplate="Coreset #%{pointNumber}<br>PC1=%{x:.3f} PC2=%{y:.3f}<extra></extra>",
    ))

    # Patches de l'image courante
    if query_features is not None:
        q = coords[n_coreset:]
        colors = patch_scores_query if patch_scores_query is not None else np.zeros(len(q))
        fig.add_trace(go.Scatter(
            x=q[:, 0], y=q[:, 1],
            mode="markers",
            name="Patches image (anomalie = rouge)",
            marker=dict(
                color=colors,
                colorscale="RdYlBu_r",
                size=9, opacity=0.85,
                colorbar=dict(title="Score", thickness=12, len=0.6, x=1.08),
                line=dict(width=0.5, color="white"),
                showscale=True,
            ),
            hovertemplate=(
                "Patch #%{pointNumber}<br>"
                "PC1=%{x:.3f} PC2=%{y:.3f}<br>"
                "Score=%{marker.color:.3f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=f"Espace latent DINOv2 (PCA — {var[0]*100:.1f}% + {var[1]*100:.1f}% variance)",
        xaxis_title=f"PC1 ({var[0]*100:.1f}%)",
        yaxis_title=f"PC2 ({var[1]*100:.1f}%)",
        height=480,
        legend=dict(orientation="h", y=-0.18),
        hovermode="closest",
    )
    return fig


def load_f1_metrics(df: pd.DataFrame | None = None) -> dict:
    """
    Charge (ou calcule) les métriques F1 de l'évaluation PatchCore.

    Ordre de priorité :
      1. evaluation_report.csv (f1_macro_evaluate / f1_anomaly / f1_normal)
      2. Fallback : recalcul dynamique depuis df (quality_flag + patchcore_score P90)

    Returns:
        dict avec f1_macro, f1_anomaly, f1_normal, source ("report" | "dynamic" | "unavailable")
    """
    result: dict = {
        "f1_macro":   float("nan"),
        "f1_anomaly": float("nan"),
        "f1_normal":  float("nan"),
        "source": "unavailable",
    }

    report_path = PROJECT_ROOT / "outputs" / "scores" / "evaluation_report.csv"
    if report_path.exists():
        try:
            rpt = pd.read_csv(report_path)
            if not rpt.empty:
                for dest, col in [
                    ("f1_macro",   "f1_macro_evaluate"),
                    ("f1_anomaly", "f1_anomaly"),
                    ("f1_normal",  "f1_normal"),
                ]:
                    if col in rpt.columns and pd.notna(rpt[col].iloc[0]):
                        result[dest] = float(rpt[col].iloc[0])
                if pd.notna(result["f1_macro"]):
                    result["source"] = "report"
                    return result
        except Exception:
            pass

    # Fallback dynamique
    if df is not None and "patchcore_score" in df.columns and "quality_flag" in df.columns:
        try:
            from sklearn.metrics import f1_score as _f1_score
            df_bin = df[
                df["quality_flag"].isin(["dark", "usable"]) & df["patchcore_score"].notna()
            ].copy()
            if not df_bin.empty and df_bin["quality_flag"].eq("dark").any():
                threshold = float(np.percentile(df_bin["patchcore_score"].values, 90))
                y_true = (df_bin["quality_flag"] == "dark").astype(int).values
                y_pred = (df_bin["patchcore_score"].values > threshold).astype(int)
                result["f1_macro"]   = float(_f1_score(y_true, y_pred, average="macro",  labels=[0, 1], zero_division=0))
                result["f1_anomaly"] = float(_f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0))
                result["f1_normal"]  = float(_f1_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0))
                result["source"] = "dynamic"
        except Exception:
            pass

    return result


def compute_patch_metadata(patch_map: np.ndarray) -> pd.DataFrame:
    """
    Calcule un DataFrame de métadonnées pour tous les patches d'une image.

    Args:
        patch_map: (H_p, W_p) scores normalisés [0, 1].

    Returns:
        DataFrame avec colonnes :
            patch_id, ligne, col, score, niveau, rank
    """
    flat = patch_map.flatten()
    h, w = patch_map.shape
    rows = []
    for idx, score in enumerate(flat):
        li, co = divmod(idx, w)
        rows.append({
            "patch_id": idx,
            "ligne": li,
            "col": co,
            "score": round(float(score), 5),
            "niveau": _anomaly_level(float(score)),
        })
    df_meta = pd.DataFrame(rows)
    df_meta["rank"] = df_meta["score"].rank(ascending=False, method="min").astype(int)
    return df_meta.sort_values("rank").reset_index(drop=True)


# ==============================================================
# Page 1 — Accueil
# ==============================================================

def page_accueil(df: pd.DataFrame) -> None:
    """Page d'accueil unifiée : vue globale + statistiques avancées du dataset."""
    st.title("🌋 VolcIA — Détection d'anomalies volcaniques & IA générative")
    st.markdown(
        "> *Détection d'anomalies et génération text-to-image à partir des images de surveillance "
        "du **Merapi** (Indonésie) — Caméra **Kalor**, 2014–2020*"
    )

    if df.empty:
        st.info(
            "Aucune donnée indexée pour le moment.\n\n"
            "Pour démarrer, exécutez le pipeline :\n"
            "`python run_full_pipeline.py`"
        )
        return

    # ── Métriques globales ─────────────────────────────────────────────────
    _n_disk = int(df["on_disk"].sum()) if "on_disk" in df.columns else int(df["downloaded"].sum())
    _n_scored = (
        int(df["patchcore_score"].notna().sum())
        if "patchcore_score" in df.columns
        else int(df["anomaly_score"].notna().sum())
    )
    _cov = _n_scored / _n_disk * 100 if _n_disk > 0 else 0.0
    years_range = (
        f"{int(df['year'].min())}–{int(df['year'].max())}"
        if df["year"].notna().any() else "—"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total indexées", f"{len(df):,}")
    c2.metric(
        "Sur disque",
        f"{_n_disk:,}",
        help=(
            "Fichiers réellement présents sur disque (≠ 'downloaded=True' dans l'index, "
            "qui peut inclure des entrées fantômes)."
        ),
    )
    c3.metric("Années", years_range)
    c4.metric("Scorées PatchCore", f"{_n_scored:,}")
    c5.metric(
        "Couverture disque",
        f"{_cov:.0f}%",
        delta=(
            f"{_n_disk - _n_scored:+,} manquantes"
            if _n_disk > _n_scored else "✓ complet"
        ),
        delta_color="inverse",
    )

    if _n_disk > 0 and _cov < 90:
        st.warning(
            f"⚠️ **Couverture PatchCore incomplète** : {_n_scored:,} / {_n_disk:,} ({_cov:.0f}%). "
            f"**{_n_disk - _n_scored:,} fichiers présents localement sans score.**\n\n"
            "```bash\npython run_v1_pipeline.py --step patchcore\n```\n\n"
            "Voir onglet **🔍 Couverture pipeline** ci-dessous pour le détail par année."
        )
    elif _n_disk > 0:
        st.success(f"✅ Couverture PatchCore : {_cov:.0f}% — tous les fichiers sur disque sont scorés.")

    st.markdown("---")

    # ── Tabs principaux ────────────────────────────────────────────────────
    tab_global, tab_time, tab_quality, tab_coverage, tab_insights = st.tabs([
        "🌍 Vue globale",
        "📅 Temporel",
        "🎯 Qualité & Fichiers",
        "🔍 Couverture pipeline",
        "💡 Insights clés",
    ])

    # ──────────────────────────────────────────────────────────────────────
    with tab_global:
        st.subheader("📊 Images par année")
        # Toujours rendre st.plotly_chart — figure vide annotée si pas de données (anti-removeChild)
        if df["year"].notna().any():
            year_counts = df.groupby("year").size().reset_index(name="count")
            year_counts["year"] = year_counts["year"].astype(int)
            fig_yr = go.Figure(go.Bar(
                x=year_counts["year"].tolist(),
                y=year_counts["count"].tolist(),
                text=year_counts["count"].tolist(),
                textposition="outside",
                marker_color="#3498db",
                hovertemplate="<b>%{x}</b><br>%{y:,} images<extra></extra>",
            ))
            fig_yr.update_layout(
                xaxis_title="Année",
                yaxis_title="Nombre d'images",
                height=380,
                margin=dict(l=0, r=0, t=20, b=40),
                bargap=0.25,
            )
        else:
            fig_yr = go.Figure()
            fig_yr.add_annotation(
                text="Aucune donnée temporelle disponible",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=14, color="gray"),
            )
            fig_yr.update_layout(height=380, xaxis_visible=False, yaxis_visible=False,
                                 margin=dict(l=0, r=0, t=20, b=40))
        st.plotly_chart(fig_yr, use_container_width=True, key="accueil_bar_year")

    # ──────────────────────────────────────────────────────────────────────
    with tab_time:
        st.subheader("Distribution temporelle")
        # Toujours rendre st.pyplot — figure annotée si pas de données (anti-removeChild)
        fig, ax = plt.subplots(figsize=(14, 4))
        if df["year"].notna().any():
            year_month = df.groupby(["year", "month"]).size().reset_index(name="count")
            year_month["year"] = year_month["year"].astype(int)
            year_month["month"] = year_month["month"].astype(int)
            year_month["label"] = year_month.apply(
                lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1
            )
            ax.bar(range(len(year_month)), year_month["count"], color="#3498db", edgecolor="white", width=0.8)
            ax.set_xlabel("Mois")
            ax.set_ylabel("Images")
            ax.set_title("Images par mois (toutes années)")
            tick_idx = list(range(0, len(year_month), 6))
            ax.set_xticks(tick_idx)
            ax.set_xticklabels(
                [year_month["label"].iloc[i] for i in tick_idx], rotation=45, fontsize=7
            )
        else:
            ax.text(0.5, 0.5, "Aucune donnée temporelle disponible",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_axis_off()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Distribution horaire — toujours rendre st.pyplot (anti-removeChild)
        fig_hr, ax_hr = plt.subplots(figsize=(10, 3))
        if df["hour"].notna().any():
            hour_counts = df.groupby("hour").size()
            ax_hr.bar(hour_counts.index.astype(int), hour_counts.values, color="#2ecc71", edgecolor="white")
            ax_hr.set_xlabel("Heure UTC")
            ax_hr.set_ylabel("Images")
            ax_hr.set_title("Distribution horaire")
            ax_hr.set_xticks(range(0, 24))
        else:
            ax_hr.text(0.5, 0.5, "Aucune donnée horaire disponible",
                       ha="center", va="center", transform=ax_hr.transAxes, fontsize=12, color="gray")
            ax_hr.set_axis_off()
        plt.tight_layout()
        st.pyplot(fig_hr)
        plt.close(fig_hr)

    # ──────────────────────────────────────────────────────────────────────
    with tab_quality:
        _colors_q = {
            "usable": "#2ecc71", "dark": "#34495e",
            "cloudy": "#95a5a6", "corrupted": "#e74c3c", "unknown": "#f39c12",
        }
        # Toujours rendre st.columns(2) — figures annotées si pas de données (anti-removeChild)
        _has_quality = df["quality_flag"].notna().any()
        col_qp, col_qy = st.columns(2)
        with col_qp:
            st.subheader("Distribution qualité")
            fig, ax = plt.subplots(figsize=(6, 4))
            if _has_quality:
                qcounts = df["quality_flag"].value_counts()
                bars = ax.bar(
                    qcounts.index, qcounts.values,
                    color=[_colors_q.get(q, "#3498db") for q in qcounts.index],
                    edgecolor="white",
                )
                for b in bars:
                    ax.text(
                        b.get_x() + b.get_width() / 2, b.get_height() + 1,
                        str(int(b.get_height())), ha="center", fontsize=9,
                    )
                ax.set_ylabel("Nombre d'images")
                ax.set_title("Qualité globale")
            else:
                ax.text(0.5, 0.5, "Classification qualité\nnon disponible",
                        ha="center", va="center", transform=ax.transAxes, fontsize=11, color="gray")
                ax.set_axis_off()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_qy:
            st.subheader("Qualité par année")
            fig, ax = plt.subplots(figsize=(8, 4))
            if _has_quality and df["year"].notna().any():
                cross = pd.crosstab(df["year"].dropna().astype(int), df["quality_flag"])
                cross.plot(
                    kind="bar", stacked=True, ax=ax,
                    color=[_colors_q.get(c, "#3498db") for c in cross.columns],
                )
                ax.set_xlabel("Année")
                ax.set_ylabel("Nombre d'images")
                ax.set_title("Qualité par année")
                ax.legend(fontsize=8, bbox_to_anchor=(1, 1))
            else:
                ax.text(0.5, 0.5, "Données insuffisantes",
                        ha="center", va="center", transform=ax.transAxes, fontsize=11, color="gray")
                ax.set_axis_off()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")
        st.subheader("Tailles de fichiers")
        _sizes = (
            df["file_size_bytes"].dropna()
            if "file_size_bytes" in df.columns
            else pd.Series(dtype=float)
        )
        # Toujours rendre st.pyplot + st.columns(3) + métriques (anti-removeChild)
        fig, ax = plt.subplots(figsize=(10, 3))
        if not _sizes.empty:
            ax.hist(_sizes / 1024, bins=50, edgecolor="white", alpha=0.8, color="#9b59b6")
            ax.set_xlabel("Taille (Ko)")
            ax.set_ylabel("Fréquence")
            ax.set_title(
                f"Distribution — µ={_sizes.mean()/1024:.0f} Ko, σ={_sizes.std()/1024:.0f} Ko"
            )
            ax.axvline(_sizes.mean() / 1024, color="red", ls="--", label="Moyenne")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "Pas de données de taille de fichier",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_axis_off()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        cs1, cs2, cs3 = st.columns(3)
        cs1.metric("Taille totale",   f"{_sizes.sum() / 1e6:.1f} Mo"   if not _sizes.empty else "—")
        cs2.metric("Taille moyenne",  f"{_sizes.mean() / 1024:.0f} Ko" if not _sizes.empty else "—")
        cs3.metric("Taille max",      f"{_sizes.max() / 1024:.0f} Ko"  if not _sizes.empty else "—")

        st.markdown("---")
        st.subheader("Progression du pipeline")
        _total = len(df)
        _dl = int(df["downloaded"].sum()) if "downloaded" in df.columns else 0
        _quality_n = int(df["quality_flag"].notna().sum())
        _scored_n = (
            int(df["anomaly_score"].notna().sum())
            if "anomaly_score" in df.columns else 0
        )
        for _label, _done, _tot in [
            ("Phase 1 — Indexation", _total, _total),
            ("Phase 1 — Téléchargement", _dl, _total),
            ("Phase 3 — Classification qualité", _quality_n, _total),
            ("Phase 4 — Scoring PatchCore", _scored_n, _total),
        ]:
            _pct = _done / _tot * 100 if _tot > 0 else 0
            st.markdown(f"**{_label}** — {_done}/{_tot} ({_pct:.0f}%)")
            st.progress(_pct / 100)

        if "on_disk" in df.columns and "downloaded" in df.columns:
            _phantom = int(df["downloaded"].sum()) - _n_disk
            if _phantom > 0:
                st.info(
                    f"ℹ️ **{_phantom:,} entrées fantômes** : marquées `downloaded=True` "
                    "mais fichier absent sur disque. Causes : doublons macOS `' 2.jpg'`, "
                    "images d'autres caméras, fichiers supprimés."
                )

    # ──────────────────────────────────────────────────────────────────────
    with tab_coverage:
        st.subheader("🔍 Couverture pipeline par année")
        st.caption(
            "**Indexées** = total dans l'index CSV | "
            "**Sur disque ✓** = fichiers accessibles | "
            "**Scorées PatchCore** = score calculé. "
            "La couverture = Scorées ÷ Sur disque."
        )
        if df["year"].notna().any():
            _cov_rows = []
            for _yr in sorted(df["year"].dropna().unique()):
                _d = df[df["year"] == _yr]
                _n_tot = len(_d)
                _n_dl = int(_d["downloaded"].sum()) if "downloaded" in _d.columns else 0
                _n_dsk = int(_d["on_disk"].sum()) if "on_disk" in df.columns else _n_dl
                _n_sc = (
                    int(_d["patchcore_score"].notna().sum())
                    if "patchcore_score" in df.columns
                    else int(_d["anomaly_score"].notna().sum())
                )
                _n_usp = int(_d["quality_flag"].eq("usable").sum())
                _cv = _n_sc / _n_dsk * 100 if _n_dsk > 0 else 0.0
                _cov_rows.append({
                    "Année": int(_yr),
                    "Indexées": _n_tot,
                    "Téléch. (index)": _n_dl,
                    "Sur disque ✓": _n_dsk,
                    "Scorées PatchCore": _n_sc,
                    "Usable": _n_usp,
                    "Couverture disque": f"{_cv:.0f}%",
                    "_cv_float": _cv,
                })
            _cov_df = pd.DataFrame(_cov_rows)
            st.dataframe(_cov_df.drop(columns=["_cv_float"]), hide_index=True,
                         use_container_width=True, key="accueil_coverage_table")
            st.info(
                "**Téléch. (index)** : marquées `downloaded=True` — peut inclure des entrées fantômes.\n\n"
                "**Sur disque ✓** : fichiers réellement présents et lisibles.\n\n"
                "**Couverture** = Scorées PatchCore ÷ Sur disque ✓"
            )
            try:
                _fig, _axes = plt.subplots(1, 2, figsize=(14, 4))
                _years_c = [int(r["Année"]) for r in _cov_rows]
                _x = list(range(len(_years_c)))
                _idx_v = [r["Indexées"] for r in _cov_rows]
                _dsk_v = [r["Sur disque ✓"] for r in _cov_rows]
                _sc_v = [r["Scorées PatchCore"] for r in _cov_rows]
                _cv_v = [r["_cv_float"] for r in _cov_rows]

                _ax1 = _axes[0]
                _ax1.bar(_x, _idx_v, label="Indexées", color="#bdc3c7", width=0.6)
                _ax1.bar(_x, _dsk_v, label="Sur disque", color="#3498db", width=0.6)
                _ax1.bar(_x, _sc_v, label="Scorées", color="#2ecc71", width=0.6)
                _ax1.set_xticks(_x)
                _ax1.set_xticklabels([str(y) for y in _years_c], rotation=45, fontsize=9)
                _ax1.set_ylabel("Nombre d'images")
                _ax1.set_title("Indexées / Sur disque / Scorées")
                _ax1.legend(fontsize=8)

                _ax2 = _axes[1]
                _bar_cols = [
                    "#2ecc71" if c >= 90 else "#f39c12" if c >= 50 else "#e74c3c"
                    for c in _cv_v
                ]
                _ax2.bar(_x, _cv_v, color=_bar_cols, edgecolor="white", width=0.6)
                _ax2.axhline(90, color="#27ae60", ls="--", alpha=0.7, label="Seuil 90%")
                _ax2.set_xticks(_x)
                _ax2.set_xticklabels([str(y) for y in _years_c], rotation=45, fontsize=9)
                _ax2.set_ylabel("Couverture disque (%)")
                _ax2.set_title("Couverture PatchCore par année")
                _ax2.set_ylim(0, 115)
                _ax2.legend(fontsize=8)
                for _xi, _val, _dsk_i in zip(_x, _cv_v, _dsk_v):
                    if _dsk_i > 0:
                        _ax2.text(_xi, _val + 2, f"{_val:.0f}%", ha="center", fontsize=8)
                plt.tight_layout()
                st.pyplot(_fig)
                plt.close(_fig)
            except Exception as _exc_cov:
                st.warning(f"Graphique couverture indisponible : {_exc_cov}")

            _bad_years = [
                r["Année"] for r in _cov_rows
                if r["Sur disque ✓"] > 0 and r["_cv_float"] < 90
            ]
            if _bad_years:
                st.warning(
                    f"⚠️ Couverture < 90% pour : **{_bad_years}**\n\n"
                    "```bash\npython run_v1_pipeline.py --step patchcore\n```"
                )
        else:
            # Toujours rendre un placeholder (anti-removeChild)
            st.caption("Aucune donnée temporelle disponible pour calculer la couverture.")

    # ──────────────────────────────────────────────────────────────────────
    with tab_insights:
        st.subheader("💡 Insights clés")
        report_path = PROJECT_ROOT / "outputs" / "scores" / "evaluation_report.csv"
        if report_path.exists():
            rpt = safe_read_csv(report_path)
            if not rpt.empty:
                _auc_pc_i = rpt["auc_pr_patchcore"].iloc[0] if "auc_pr_patchcore" in rpt.columns else float("nan")
                _improv_i = rpt["improvement_pct"].iloc[0] if "improvement_pct" in rpt.columns else float("nan")
                _mwp_i = rpt["mann_whitney_p"].iloc[0] if "mann_whitney_p" in rpt.columns else float("nan")
                _effect_i = rpt["effect_size"].iloc[0] if "effect_size" in rpt.columns else float("nan")
                _auc_roc_i = rpt["auc_roc_patchcore"].iloc[0] if "auc_roc_patchcore" in rpt.columns else float("nan")

                ia1, ia2, ia3 = st.columns(3)
                ia1.metric(
                    "AUC-PR PatchCore",
                    f"{_auc_pc_i:.4f}" if pd.notna(_auc_pc_i) else "N/A",
                    help="Aire sous la courbe Précision-Rappel.",
                )
                ia2.metric(
                    "Amélioration vs aléatoire",
                    f"{_improv_i:.0f}%" if pd.notna(_improv_i) else "N/A",
                    help="Gain relatif vs classifieur aléatoire (AUC-PR = prévalence).",
                )
                ia3.metric(
                    "AUC-ROC",
                    f"{_auc_roc_i:.4f}" if pd.notna(_auc_roc_i) else "N/A",
                    help="> 0.5 confirme la direction des scores.",
                )
                st.markdown("---")
                if pd.notna(_effect_i) and _effect_i > 0:
                    st.success(f"✅ **Signal détecté** — RBC = {_effect_i:.3f} (dark > usable).")
                if pd.notna(_mwp_i) and _mwp_i < 0.05:
                    st.success(f"✅ **Significatif** — Mann-Whitney p = {_mwp_i:.2e}.")
                if pd.notna(_improv_i):
                    st.success(f"✅ **PatchCore {1 + _improv_i/100:.1f}× mieux** que le classifieur aléatoire.")
        else:
            st.info("Rapport d'évaluation non disponible. Voir **🔬 DINOv2 + PatchCore** pour les métriques.")



# ==============================================================
# Page 2 — Exploration
# ==============================================================

def page_exploration(df: pd.DataFrame, config: dict) -> None:
    sel_year, sel_month, df_filt = sidebar_data_filters(df)

    st.header(f"🔍 Exploration — {sel_year}/{sel_month:02d}")

    # Toujours rendre le squelette UI — values = 0 si pas de données (anti-removeChild)
    _expl_has_data = not df_filt.empty

    # Résumé rapide — conteneur stable pour éviter le NotFoundError
    metrics_container = st.container()
    with metrics_container:
        c1, c2, c3 = st.columns(3)
        c1.metric("Images", len(df_filt) if _expl_has_data else 0)
        c2.metric("Téléchargées", int(df_filt["downloaded"].sum()) if _expl_has_data else 0)
        scored = int(df_filt["anomaly_score"].notna().sum()) if _expl_has_data else 0
        c3.metric("Avec score", scored)
    if not _expl_has_data:
        st.caption("Aucune image pour cette sélection — modifiez les filtres.")

    # Table de données
    st.subheader("📋 Index des images")
    display_cols = ["filename", "day", "hour", "minute", "quality_flag", "anomaly_score", "downloaded", "file_size_bytes"]
    display_cols = [c for c in display_cols if c in df_filt.columns]
    st.dataframe(
        df_filt[display_cols].sort_values(["day", "hour"], na_position="last"),
        use_container_width=True,
        hide_index=True,
        height=400,
        key="expl_index_table",
    )

    # Couverture temporelle
    st.subheader("🗓️ Couverture temporelle")
    # Toujours rendre st.pyplot — figure annotée si données insuffisantes (anti-removeChild)
    fig, ax = plt.subplots(figsize=(14, 4))
    if df_filt["day"].notna().any() and df_filt["hour"].notna().any():
        pivot = df_filt.pivot_table(
            index="hour", columns="day", values="filename", aggfunc="count"
        )
        if sns is not None:
            sns.heatmap(
                pivot, cmap="Blues", linewidths=0.3, ax=ax, annot=True, fmt=".0f",
                cbar_kws={"label": "Nombre d'images"},
            )
        else:
            ax.imshow(pivot.fillna(0).values, aspect="auto", cmap="Blues")
        ax.set_title(f"Couverture jour × heure — {sel_year}/{sel_month:02d}")
        ax.set_xlabel("Jour du mois")
        ax.set_ylabel("Heure (UTC)")
    else:
        ax.text(0.5, 0.5, "Données temporelles (jour / heure) insuffisantes",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_axis_off()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Export CSV
    st.markdown("---")
    csv_data = df_filt.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Télécharger la sélection (CSV)", csv_data,
        file_name=f"merapi_{sel_year}_{sel_month:02d}.csv",
        mime="text/csv",
    )


# ==============================================================
# Page 2 — Galerie temporelle
# ==============================================================

def page_galerie(df: pd.DataFrame, config: dict) -> None:
    sel_year, sel_month, df_filt = sidebar_data_filters(df)

    st.header(f"🖼️ Galerie temporelle — {sel_year}/{sel_month:02d}")

    if df_filt.empty:
        st.info("Aucune image pour cette sélection.")
        return

    df_sorted = df_filt.sort_values(["day", "hour", "minute"]).reset_index(drop=True)

    # Navigation
    col_nav, col_info = st.columns([3, 1])
    with col_nav:
        idx_sel = st.slider(
            "Navigation", 0, max(0, len(df_sorted) - 1), 0,
            format=f"Image %d / {len(df_sorted)}",
        )
    with col_info:
        row = df_sorted.iloc[idx_sel]
        day_v = int(row["day"]) if pd.notna(row.get("day")) else "?"
        hour_v = int(row["hour"]) if pd.notna(row.get("hour")) else "?"
        st.metric("Date", f"Jour {day_v}, {hour_v}h")

    img_path_gal = find_image_path(row)

    # Affichage côte à côte
    col_raw, col_proc = st.columns(2)

    with col_raw:
        st.markdown("**📷 Image disponible**")
        img_raw = load_image_for_display(img_path_gal) if img_path_gal is not None else None
        if img_raw is not None:
            st.image(img_raw, **{_IMG_WIDTH_KW: True})
            st.caption(f"`{img_path_gal.name}`")
        else:
            _gal_fn = row.get("filename", "?")
            _gal_dl = bool(row.get("downloaded", False))
            if not _gal_dl:
                st.warning(
                    f"⬇️ Non téléchargée : `{_gal_fn}`\n\n"
                    "Relancez : `python run_full_pipeline.py --step download`"
                )
            else:
                _gal_y = int(row.get("year", 0)) if pd.notna(row.get("year")) else 0
                _gal_m = int(row.get("month", 0)) if pd.notna(row.get("month")) else 0
                st.warning(f"⚠️ Fichier absent du disque : `{_gal_fn}`")
                if _gal_y and _gal_m:
                    with st.expander("🔍 Chemins vérifiés", expanded=False):
                        for _gp in [
                            PROJECT_ROOT / "data" / "raw" / str(_gal_y) / f"{_gal_m:02d}" / _gal_fn,
                            PROJECT_ROOT / "data" / "processed" / str(_gal_y) / f"{_gal_m:02d}" / Path(_gal_fn).with_suffix(".png").name,
                        ]:
                            _icon = "OK" if _gp.exists() else "ABSENT"
                            st.code(f"[{_icon}] {_gp}")

    with col_proc:
        st.markdown("**🟢 Version prétraitée (PNG)**")
        proc_img = None
        try:
            y, m = int(row["year"]), int(row["month"])
            stem = Path(str(row["filename"])).stem
            proc_png = PROJECT_ROOT / "data" / "processed" / str(y) / f"{m:02d}" / f"{stem}.png"
            if proc_png.exists():
                proc_img = load_image_for_display(proc_png)
        except (ValueError, TypeError, KeyError):
            pass
        if proc_img is not None:
            st.image(proc_img, clamp=True, **{_IMG_WIDTH_KW: True})
        else:
            st.info("Pas de version prétraitée (PNG)")

    # Métadonnées
    with st.expander("📋 Métadonnées complètes", expanded=False):
        meta_cols = ["filename", "url", "day", "hour", "minute", "second",
                     "quality_flag", "is_night", "anomaly_score", "file_size_bytes"]
        meta = {c: row.get(c, "—") for c in meta_cols if c in row.index}
        st.json(meta)

    # Navigation rapide
    st.markdown("---")
    if len(df_sorted) > 1:
        n_thumb = min(10, len(df_sorted))
        st.markdown(f"**Aperçu rapide** (premières {n_thumb} images)")
        thumb_cols = st.columns(n_thumb)
        for i, (_, r) in enumerate(df_sorted.head(n_thumb).iterrows()):
            _rp = find_image_path(r)
            img = load_image_for_display(_rp) if _rp is not None else None
            with thumb_cols[i]:
                if img is not None:
                    d = int(r["day"]) if pd.notna(r.get("day")) else "?"
                    h = int(r["hour"]) if pd.notna(r.get("hour")) else "?"
                    st.image(img, caption=f"j{d} {h}h", **{_IMG_WIDTH_KW: True})
                else:
                    st.text("—")


# ==============================================================
# Page 3 — Anomalies
# ==============================================================
# AUDIT — Décision architecture :
#   ❌ combined_score / anomaly_score legacy  : baselines pixel (SSIM, MAD, nuit)
#      → quasi-vides en pratique (aucun CSV baselines générés), non interprétables
#      → SUPPRIMÉ comme source primaire
#   ✅ patchcore_score                        : DINOv2 + k-NN features, seule vérité
#      → cohérent avec pages DINOv2+PatchCore et Analyse avancée
#   Décision : PatchCore = unique source de vérité dans cette page
# ==============================================================

def page_anomalies(df: pd.DataFrame, config: dict) -> None:
    sel_year, sel_month, df_filt = sidebar_data_filters(df)

    st.header(f"🚨 Anomalies PatchCore — {sel_year}/{sel_month:02d}")

    # ── Source de données : PatchCore uniquement ──────────────────────────
    _pc_col = "patchcore_score"

    # Chercher les scores PatchCore dans df_filt ou dans le CSV global
    if _pc_col in df_filt.columns and df_filt[_pc_col].notna().any():
        display_df = df_filt[df_filt[_pc_col].notna()].copy()
    else:
        _pc_path = PROJECT_ROOT / "outputs" / "scores" / "patchcore_scores.csv"
        if _pc_path.exists():
            _pc = safe_read_csv(_pc_path)
            if not _pc.empty and _pc_col in _pc.columns and "filename" in _pc.columns:
                _pc[_pc_col] = pd.to_numeric(_pc[_pc_col], errors="coerce")
                merged = df_filt.drop(columns=[_pc_col], errors="ignore").merge(
                    _pc[["filename", _pc_col]], on="filename", how="inner"
                )
                display_df = merged[merged[_pc_col].notna()].copy()
            else:
                display_df = pd.DataFrame()
        else:
            display_df = pd.DataFrame()

    # ── Aucun score → guide utilisateur ──────────────────────────────────
    if display_df.empty:
        n_total = len(df_filt)
        n_dl = int(df_filt["downloaded"].sum()) if "downloaded" in df_filt.columns else 0
        if n_dl == 0:
            st.warning(
                f"**{sel_year}/{sel_month:02d}** — {n_total} images indexées, "
                "**aucune téléchargée localement**.\n\n"
                "PatchCore requiert les images raw pour extraire les features DINOv2.\n\n"
                "```bash\npython run_v1_pipeline.py --step patchcore\n```"
            )
        else:
            st.warning(
                f"**{sel_year}/{sel_month:02d}** — {n_total} images indexées, "
                f"{n_dl} téléchargées, **0 scorées par PatchCore**.\n\n"
                "```bash\npython run_v1_pipeline.py --step patchcore\n```"
            )
        if not df_filt.empty:
            st.subheader("📋 Images indexées (sans score)")
            display_cols = [c for c in ["filename", "day", "hour", "downloaded", "quality_flag"]
                            if c in df_filt.columns]
            st.dataframe(
                df_filt[display_cols].sort_values(["day", "hour"], na_position="last"),
                use_container_width=True, hide_index=True, height=300,
                key="anom_no_score_table",
            )
        return

    # ── Statistiques ──────────────────────────────────────────────────────
    # BUGFIX : le seuil μ+2σ est calculé sur la DISTRIBUTION GLOBALE (tout le dataset),
    # pas sur la période filtrée. Raison : sur un mois calme, μ_local+2σ_local serait
    # toujours > tous les scores locaux (→ 0 anomalie), ce qui n'est pas informatif.
    # Le seuil global = "ce qui est anormal par rapport à TOUTE la surveillance".
    _scores = display_df[_pc_col].dropna()
    _mean   = float(_scores.mean())
    _std    = float(_scores.std())
    _max    = float(_scores.max())
    _p90    = float(_scores.quantile(0.90))

    _all_pc_global = (
        df[_pc_col].dropna()
        if _pc_col in df.columns and df[_pc_col].notna().sum() > 20
        else _scores
    )
    _global_mean = float(_all_pc_global.mean())
    _global_std  = float(_all_pc_global.std())
    _thr = _global_mean + 2 * _global_std  # seuil sur données GLOBALES
    _n_anom = int((_scores > _thr).sum())  # dépassements dans la période filtrée

    help_tooltip(
        "Métriques anomalies",
        f"""
| Métrique | Définition |
|---|---|
| **Images scorées** | Nombre d'images ayant un score PatchCore calculé |
| **Score moyen** | Moyenne locale des distances k-NN DINOv2 pour la période |
| **Score max** | Distance maximale — image la plus anormale de la période |
| **Anomalies (>2σ)** | Images dont le score **dépasse le seuil GLOBAL** μ+2σ = {_thr:.4f} |

**Score PatchCore** : distance k-NN entre les features DINOv2 de l'image et le
coreset des images normales. Plus le score est élevé, plus l'image est anormale.

**Pourquoi le seuil est GLOBAL ?** Le seuil μ+2σ est calculé sur l'ensemble du
dataset ({len(_all_pc_global):,} images scorées), pas uniquement sur le mois affiché.
Ainsi, une période calme peut avoir 0 anomalies — ce qui est **informatif** (mois normal),
contrairement au seuil local qui serait toujours ~5% par construction.

> μ_global = {_global_mean:.4f} · σ_global = {_global_std:.4f} → seuil = {_thr:.4f}
        """,
        key="help_anomalies_metrics",
    )

    _delta_mean = _mean - _global_mean
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images scorées", f"{len(display_df):,}")
    c2.metric("Score moyen (local)", f"{_mean:.4f}",
              delta=f"{_delta_mean:+.4f} vs global",
              delta_color="inverse" if _delta_mean > 0 else "normal")
    c3.metric("Score max", f"{_max:.4f}")
    c4.metric(
        "Anomalies (>2σ global)",
        f"{_n_anom:,}",
        delta=f"seuil {_thr:.4f}",
        delta_color="off",
    )

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_dist, tab_top, tab_heatmap = st.tabs(["📊 Distribution", "🔝 Top anomalies", "🗺️ Heatmap temporelle"])

    # ── Tab 1 : Distribution (Plotly interactif) ──────────────────────────
    with tab_dist:
        help_tooltip(
            "Distribution des scores",
            """
**Histogramme** : répartition des scores PatchCore sur la période sélectionnée.

- **Seuil μ+2σ** (ligne rouge) : images au-dessus = potentiellement anomales
- **P90** (ligne orange) : seuil percentile utilisé dans la page DINOv2+PatchCore

**Timeline** : évolution chronologique du score. Les pics sont des candidats
à l'anomalie volcanique (éruption, éboulement, changement de fumée).
            """,
            key="help_anom_dist",
        )

        # Histogramme Plotly
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=_scores,
            nbinsx=40,
            marker_color="#e74c3c",
            opacity=0.8,
            name="Score PatchCore",
            hovertemplate="Score : %{x:.4f}<br>Fréquence : %{y}<extra></extra>",
        ))
        fig_hist.add_vline(
            x=_thr, line_dash="dash", line_color="red",
            annotation_text=f"μ+2σ global = {_thr:.4f}",
            annotation_position="top right",
        )
        fig_hist.add_vline(
            x=_mean, line_dash="dot", line_color="blue",
            annotation_text=f"μ local = {_mean:.4f}",
            annotation_position="top left",
        )
        fig_hist.add_vline(
            x=_p90, line_dash="dot", line_color="orange",
            annotation_text=f"P90 local = {_p90:.4f}",
            annotation_position="bottom right",
        )
        fig_hist.update_layout(
            title=f"Distribution des scores PatchCore — {sel_year}/{sel_month:02d}",
            xaxis_title="Score PatchCore",
            yaxis_title="Fréquence",
            height=320,
            margin=dict(t=45, b=40),
        )
        st.plotly_chart(fig_hist, use_container_width=True, key="anom_hist")

        # Timeline interactive
        _df_sorted = display_df.sort_values(["day", "hour", "minute"], na_position="last").reset_index(drop=True)
        _label_x = _df_sorted.apply(
            lambda r: f"j{int(r.get('day',0)):02d} {int(r.get('hour',0)):02d}h"
            if pd.notna(r.get("day")) else str(r.name),
            axis=1,
        )
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=_df_sorted.index,
            y=_df_sorted[_pc_col],
            mode="lines+markers",
            marker=dict(
                size=5,
                color=_df_sorted[_pc_col],
                colorscale="YlOrRd",
                showscale=True,
                colorbar=dict(title="Score"),
            ),
            line=dict(color="#3498db", width=1),
            customdata=_label_x,
            hovertemplate="<b>%{customdata}</b><br>Score : %{y:.4f}<extra></extra>",
            name="Score PatchCore",
        ))
        fig_line.add_hline(
            y=_thr, line_dash="dash", line_color="red",
            annotation_text=f"μ+2σ={_thr:.4f}",
        )
        fig_line.add_hline(
            y=_mean, line_dash="dot", line_color="green",
            annotation_text=f"μ={_mean:.4f}",
        )
        fig_line.update_layout(
            title="Évolution chronologique du score",
            xaxis_title="Index chronologique",
            yaxis_title="Score PatchCore",
            height=300,
            margin=dict(t=45, b=40),
        )
        st.plotly_chart(fig_line, use_container_width=True, key="anom_timeline")

    # ── Tab 2 : Top anomalies ─────────────────────────────────────────────
    with tab_top:
        top_n = st.slider("Nombre d'images prioritaires", 5, 50, 15, key="anom_top_n")
        top = display_df.nlargest(top_n, _pc_col).reset_index(drop=True)

        help_tooltip(
            "Tableau Top anomalies",
            """
| Colonne | Définition |
|---|---|
| **filename** | Nom du fichier image (format `kalor_YYYYMMDDHHMMSS.jpg`) |
| **day** | Jour du mois |
| **hour** | Heure UTC de la prise de vue |
| **patchcore_score** | Score d'anomalie PatchCore (0 = normal, >0.5 = suspect) |
| **quality_flag** | `usable` = exploitable, `dark` = image sombre/nuageuse |

Les images sont classées par score décroissant (la plus anomale en premier).
            """,
            key="help_anom_top_table",
        )

        show_cols = [c for c in ["filename", "day", "hour", _pc_col, "quality_flag"]
                     if c in top.columns]

        # Tableau stylé — gradient de couleur sur le score
        _top_display = top[show_cols].copy()
        if _pc_col in _top_display.columns:
            _top_display[_pc_col] = _top_display[_pc_col].round(4)

        st.dataframe(
            _top_display.style.background_gradient(subset=[_pc_col], cmap="YlOrRd"),
            use_container_width=True,
            hide_index=True,
            key="anom_top_table",
        )

        # Bar chart des top scores (Plotly)
        fig_top = go.Figure(go.Bar(
            x=top[_pc_col].values[::-1],
            y=[f"j{int(r.get('day',0)):02d} {int(r.get('hour',0)):02d}h"
               for _, r in top.iloc[::-1].iterrows()],
            orientation="h",
            marker=dict(
                color=top[_pc_col].values[::-1],
                colorscale="YlOrRd",
                showscale=True,
                colorbar=dict(title="Score"),
            ),
            hovertemplate="<b>%{y}</b><br>Score : %{x:.4f}<extra></extra>",
        ))
        fig_top.update_layout(
            title=f"Top {top_n} images les plus anomales",
            xaxis_title="Score PatchCore",
            height=max(250, top_n * 22),
            margin=dict(t=45, l=90, b=40),
        )
        st.plotly_chart(fig_top, use_container_width=True, key="anom_top_bar")

        # Aperçu visuel
        st.subheader("Aperçu des images prioritaires")
        n_cols = min(5, len(top))
        if n_cols > 0:
            cols = st.columns(n_cols)
            for i, (_, row) in enumerate(top.head(n_cols).iterrows()):
                fn = row.get("filename", "")
                idx_row = df[df["filename"] == fn]
                if idx_row.empty:
                    continue
                _ip = find_image_path(idx_row.iloc[0])
                img = load_image_for_display(_ip) if _ip is not None else None
                with cols[i]:
                    if img is not None:
                        day_v  = int(row["day"])  if pd.notna(row.get("day"))  else "?"
                        hour_v = int(row["hour"]) if pd.notna(row.get("hour")) else "?"
                        score_v = row[_pc_col]
                        st.image(img, caption=f"j{day_v} {hour_v}h — {score_v:.3f}", **{_IMG_WIDTH_KW: True})
                    else:
                        st.caption(f"{fn}\n(non disponible)")

    # ── Tab 3 : Heatmap temporelle (Plotly interactif) ────────────────────
    with tab_heatmap:
        help_tooltip(
            "Heatmap temporelle",
            """
**Axe X** : jour du mois. **Axe Y** : heure UTC.

Chaque cellule = score PatchCore moyen des images prises ce jour à cette heure.

- **Zone rouge / chaude** → activité anormale fréquente à ce créneau horaire
- **Zone bleue / froide** → comportement normal
- **Cellule vide** → aucune image disponible ce créneau

**Utilité volcanique** : permet de repérer des patterns temporels (matin vs soir,
jours particuliers) qui pourraient corréler avec des événements sismiques ou
météorologiques.
            """,
            key="help_anom_heatmap",
        )

        if "day" in display_df.columns and "hour" in display_df.columns:
            pivot = display_df.pivot_table(
                index="hour", columns="day", values=_pc_col, aggfunc="mean",
            )
            fig_hm = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[int(c) for c in pivot.columns],
                y=[int(r) for r in pivot.index],
                colorscale="YlOrRd",
                colorbar=dict(title="Score<br>PatchCore"),
                hovertemplate="Jour %{x} — %{y}h UTC<br>Score moyen : %{z:.4f}<extra></extra>",
                zsmooth=False,
            ))
            fig_hm.update_layout(
                title=f"Heatmap anomalies — {sel_year}/{sel_month:02d}",
                xaxis_title="Jour du mois",
                yaxis_title="Heure UTC",
                height=380,
                margin=dict(t=50, b=40),
            )
        else:
            fig_hm = go.Figure()
            fig_hm.add_annotation(
                text="Données temporelles (jour / heure) insuffisantes pour la heatmap",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=13, color="gray"),
            )
            fig_hm.update_layout(height=380, xaxis_visible=False, yaxis_visible=False,
                                 margin=dict(t=50, b=40))
        st.plotly_chart(fig_hm, use_container_width=True, key="anom_heatmap_plotly")


# ==============================================================
# Page 5 — Simulation d'écoulements
# ==============================================================

def page_simulation() -> None:
    st.header("🌋 Simulation 3D d'écoulements volcaniques")
    st.markdown(
        "Visualisation **3D interactive** d'une coulée de lave basée sur les paramètres "
        "physiques du Merapi. Rotation, zoom et survol des données inclus."
    )

    help_tooltip(
        "Modèle physique utilisé",
        r"""
**Modèle simplifié d'écoulement gravitaire**

La vitesse de front est approximée par :

$$v = \frac{\rho g \sin(\alpha) \cdot h^2}{3 \mu}$$

| Paramètre | Valeur | Unité |
|---|---|---|
| $\rho$ — densité magma | 2 500 | kg/m³ |
| $g$ — gravité | 9.81 | m/s² |
| $\alpha$ — angle de pente | réglable | ° |
| $h$ — épaisseur coulée | calculée | m |
| $\mu$ — viscosité | réglable | Pa·s |

**Limitations** : modèle de démonstration. Ne remplace pas VolcFlow, MOLASSES, etc.

*Référence : Kelfoun & Druitt, 2005 — Modélisation d'écoulements volcaniques*
    """,
        key="help_sim_model",
    )

    col_params, col_viz = st.columns([1, 2])

    with col_params:
        st.subheader("Paramètres")
        slope = st.slider("Pente (°)", 5, 50, 20, key="sim_slope")
        viscosity_val = st.slider("Viscosité (Pa·s)", 100, 100_000, 10_000, step=1000, key="sim_visc")
        volume = st.slider("Volume (×1000 m³)", 1, 500, 50, key="sim_vol")
        duration = st.slider("Durée (heures)", 1, 48, 12, key="sim_dur")
        n_particles = st.slider("Particules", 100, 1000, 400, step=100, key="sim_npart")

        view_mode = st.radio(
            "Vue 3D",
            ["🌋 Surface du terrain + coulée", "🔥 Nuage de particules"],
            index=0, horizontal=True, key="sim_view",
        )
        show_temp = st.checkbox("Colorier par température", value=True, key="sim_temp")

        run_sim = st.button("▶️ Lancer la simulation", type="primary",
                            use_container_width=True, key="sim_run")

    # Invalider le cache si les paramètres changent
    _sim_key = f"{slope}|{viscosity_val}|{volume}|{duration}|{n_particles}|{view_mode}|{show_temp}"
    if st.session_state.get("_sim_params_key") != _sim_key:
        st.session_state.pop("_sim_fig", None)
        st.session_state["_sim_params_key"] = _sim_key

    if run_sim:
        with st.spinner("Simulation 3D en cours…"):
            _fig3d = _run_flow_simulation_3d(
                slope_deg=slope,
                viscosity=viscosity_val,
                volume=volume,
                duration=duration,
                n_particles=n_particles,
                view_mode="surface" if "Surface" in view_mode else "particles",
                colorize_temp=show_temp,
            )
            st.session_state["_sim_fig"] = _fig3d

    with col_viz:
        # Toujours rendre st.plotly_chart — figure annotée si simulation pas encore lancée (anti-removeChild)
        if st.session_state.get("_sim_fig") is not None:
            _sim_display_fig = st.session_state["_sim_fig"]
        else:
            _sim_display_fig = go.Figure()
            _sim_display_fig.add_annotation(
                text="Ajustez les paramètres et cliquez sur ▶️ Lancer la simulation",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=13, color="gray"),
            )
            _sim_display_fig.update_layout(
                height=500, xaxis_visible=False, yaxis_visible=False,
                margin=dict(l=0, r=0, t=20, b=20),
            )
        st.plotly_chart(_sim_display_fig, use_container_width=True, key="sim_3d_chart")

    # Métriques physiques — toujours rendre st.columns(4) + métriques (anti-removeChild)
    st.divider()
    _sim_has_result = st.session_state.get("_sim_fig") is not None
    if _sim_has_result:
        rho = 2500.0
        slope_rad = np.radians(slope)
        h = (volume * 1000 / 100.0) ** (1 / 3)
        v_front = rho * 9.81 * np.sin(slope_rad) * h**2 / (3 * viscosity_val)
        total_dist = min(v_front * duration * 3600, 10000)
        _mv, _dv, _ev, _dv2 = f"{v_front:.3f} m/s", f"{total_dist/1000:.2f} km", f"{h:.1f} m", f"{duration} h"
    else:
        _mv, _dv, _ev, _dv2 = "—", "—", "—", "—"
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Vitesse de front", _mv)
    mc2.metric("Distance parcourue", _dv)
    mc3.metric("Épaisseur estimée", _ev)
    mc4.metric("Durée simulée", _dv2)


def _run_flow_simulation_3d(
    slope_deg: float,
    viscosity: float,
    volume: float,
    duration: float,
    n_particles: int,
    view_mode: str = "surface",
    colorize_temp: bool = True,
) -> go.Figure:
    """
    Génère une visualisation 3D Plotly interactive d'un écoulement volcanique.

    Args:
        view_mode : "surface" (terrain 3D + coulée) | "particles" (nuage 3D)
        colorize_temp : colorier les particules selon la température estimée
    """
    rng = np.random.default_rng(42)

    # Paramètres physiques
    rho = 2500.0
    g = 9.81
    slope_rad = np.radians(slope_deg)
    h = (volume * 1000 / 100.0) ** (1 / 3)
    v_front = rho * g * np.sin(slope_rad) * h**2 / (3 * viscosity)
    total_dist = min(v_front * duration * 3600, 10_000)
    base_width = 50 + (1e5 / max(viscosity, 1)) * volume / 100

    t = np.linspace(0, 1, n_particles)
    x_center = t * total_dist
    spread = base_width * t * (1 + 0.3 * rng.standard_normal(n_particles))
    y_offset = spread * rng.standard_normal(n_particles)
    z_flow = -x_center * np.sin(slope_rad) + h * (1 - t)  # altitude + épaisseur

    temp = 1050 - 350 * t + 30 * rng.standard_normal(n_particles)
    temp = np.clip(temp, 600, 1100)

    if view_mode == "surface":
        # ── Vue Surface 3D ─────────────────────────────────────────────
        # Grille du terrain
        _xs = np.linspace(0, total_dist, 60)
        _ys = np.linspace(-base_width * 1.5, base_width * 1.5, 40)
        _X, _Y = np.meshgrid(_xs, _ys)
        _Z_terrain = -_X * np.sin(slope_rad)

        # Grille de la coulée (zone centrale)
        _flow_width = base_width * np.linspace(0, 1, 60) * (1 + 0.2 * rng.standard_normal(60))
        _flow_width = np.clip(_flow_width, 0, base_width * 2)
        _Z_flow = np.where(
            np.abs(_Y) < _flow_width[np.newaxis, :],
            _Z_terrain + h * (1 - _X / total_dist),
            np.nan,
        )
        # Température sur la grille de coulée
        _temp_grid = 1100 - 500 * _X / total_dist

        fig = go.Figure()

        # Terrain
        fig.add_trace(go.Surface(
            x=_X, y=_Y, z=_Z_terrain,
            colorscale=[[0, "#5d4037"], [0.5, "#795548"], [1, "#a1887f"]],
            opacity=0.6,
            showscale=False,
            name="Terrain",
            hovertemplate="Distance : %{x:.0f} m<br>Y : %{y:.0f} m<br>Alt : %{z:.0f} m<extra>Terrain</extra>",
        ))

        # Coulée de lave
        fig.add_trace(go.Surface(
            x=_X, y=_Y, z=_Z_flow,
            surfacecolor=_temp_grid if colorize_temp else None,
            colorscale="Hot",
            cmin=600, cmax=1100,
            colorbar=dict(title="Temp. °C", len=0.6, x=1.02),
            opacity=0.9,
            name="Coulée",
            hovertemplate="Distance : %{x:.0f} m<br>Temp : %{surfacecolor:.0f} °C<extra>Coulée</extra>",
        ))

        # Évent source
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[h * 0.5],
            mode="markers+text",
            marker=dict(size=10, color="yellow", symbol="diamond"),
            text=["Source"],
            textposition="top center",
            name="Source",
            hoverinfo="name",
        ))

        fig.update_layout(
            title=f"Coulée 3D — pente {slope_deg}°, v={v_front:.3f} m/s, dist={total_dist/1000:.1f} km",
            scene=dict(
                xaxis_title="Distance (m)",
                yaxis_title="Déviation (m)",
                zaxis_title="Altitude (m)",
                camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
                aspectmode="manual",
                aspectratio=dict(x=2, y=1, z=0.5),
            ),
            height=600,
            margin=dict(t=50),
        )

    else:
        # ── Vue nuage de particules 3D ──────────────────────────────────
        fig = go.Figure(go.Scatter3d(
            x=x_center,
            y=y_offset,
            z=z_flow,
            mode="markers",
            marker=dict(
                size=3,
                color=temp if colorize_temp else x_center,
                colorscale="Hot" if colorize_temp else "Blues",
                cmin=600 if colorize_temp else None,
                cmax=1100 if colorize_temp else None,
                colorbar=dict(title="Temp. °C" if colorize_temp else "Distance"),
                opacity=0.75,
            ),
            customdata=np.stack([temp, x_center / 1000], axis=1),
            hovertemplate=(
                "Distance : %{customdata[1]:.2f} km<br>"
                "Température : %{customdata[0]:.0f} °C<extra></extra>"
            ),
        ))

        # Évent source
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[h * 0.5],
            mode="markers",
            marker=dict(size=12, color="yellow", symbol="diamond"),
            name="Source",
        ))

        fig.update_layout(
            title=f"Particules 3D — {n_particles} pts, v={v_front:.3f} m/s",
            scene=dict(
                xaxis_title="Distance (m)",
                yaxis_title="Déviation (m)",
                zaxis_title="Altitude (m)",
                camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
            ),
            height=600,
            margin=dict(t=50),
        )

    return fig


# ==============================================================
# Page 7 — Statistiques
# ==============================================================

def page_statistiques(df: pd.DataFrame) -> None:
    st.header("📈 Statistiques du dataset")

    if df.empty:
        st.warning("Aucune donnée disponible.")
        return

    # Métriques globales
    metrics_container = st.container()
    with metrics_container:
        _n_disk_s = int(df["on_disk"].sum()) if "on_disk" in df.columns else int(df["downloaded"].sum())
        _n_scored_s = int(df["patchcore_score"].notna().sum()) if "patchcore_score" in df.columns else int(df["anomaly_score"].notna().sum())
        _cov_s = _n_scored_s / _n_disk_s * 100 if _n_disk_s > 0 else 0.0
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total images", f"{len(df):,}")
        c2.metric("Sur disque", f"{_n_disk_s:,}")
        c3.metric("Classifiées", f"{int(df['quality_flag'].notna().sum()):,}")
        c4.metric("Scorées PatchCore", f"{_n_scored_s:,}")
        c5.metric("Couverture disque", f"{_cov_s:.0f}%")

    tab_time, tab_quality, tab_files, tab_progress, tab_coverage = st.tabs([
        "📅 Temporal", "🎯 Qualité", "📁 Fichiers", "📊 Progression", "🔍 Couverture pipeline"
    ])

    with tab_time:
        st.subheader("Distribution temporelle")

        # Par année
        if df["year"].notna().any():
            year_month = df.groupby(["year", "month"]).size().reset_index(name="count")
            year_month["year"] = year_month["year"].astype(int)
            year_month["month"] = year_month["month"].astype(int)

            fig, ax = plt.subplots(figsize=(14, 4))
            year_month["label"] = year_month.apply(
                lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1
            )
            ax.bar(range(len(year_month)), year_month["count"], color="#3498db", edgecolor="white", width=0.8)
            ax.set_xlabel("Mois")
            ax.set_ylabel("Images")
            ax.set_title("Images par mois (toutes années)")
            # Labels tous les 6 mois
            tick_idx = list(range(0, len(year_month), 6))
            ax.set_xticks(tick_idx)
            ax.set_xticklabels([year_month["label"].iloc[i] for i in tick_idx], rotation=45, fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Par heure — toujours rendre st.pyplot (anti-removeChild)
        fig_hr2, ax_hr2 = plt.subplots(figsize=(10, 3))
        if df["hour"].notna().any():
            hour_counts = df.groupby("hour").size()
            ax_hr2.bar(hour_counts.index.astype(int), hour_counts.values, color="#2ecc71", edgecolor="white")
            ax_hr2.set_xlabel("Heure UTC")
            ax_hr2.set_ylabel("Images")
            ax_hr2.set_title("Distribution horaire")
            ax_hr2.set_xticks(range(0, 24))
        else:
            ax_hr2.text(0.5, 0.5, "Aucune donnée horaire disponible",
                        ha="center", va="center", transform=ax_hr2.transAxes, fontsize=12, color="gray")
            ax_hr2.set_axis_off()
        plt.tight_layout()
        st.pyplot(fig_hr2)
        plt.close(fig_hr2)

    with tab_quality:
        st.subheader("Répartition par qualité")

        # Toujours rendre st.columns(2) — figures annotées si pas de données (anti-removeChild)
        _stat_has_quality = df["quality_flag"].notna().any()
        colors_q = {
            "usable": "#2ecc71", "dark": "#34495e",
            "cloudy": "#95a5a6", "corrupted": "#e74c3c", "unknown": "#f39c12",
        }
        col_qp, col_qy = st.columns(2)

        with col_qp:
            fig, ax = plt.subplots(figsize=(6, 4))
            if _stat_has_quality:
                qcounts = df["quality_flag"].value_counts()
                bars = ax.bar(
                    qcounts.index, qcounts.values,
                    color=[colors_q.get(q, "#3498db") for q in qcounts.index],
                    edgecolor="white",
                )
                for b in bars:
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                            str(int(b.get_height())), ha="center", fontsize=9)
                ax.set_ylabel("Nombre d'images")
                ax.set_title("Qualité globale")
            else:
                ax.text(0.5, 0.5, "Classification qualité\nnon disponible\n(Phase 3 non exécutée)",
                        ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
                ax.set_axis_off()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_qy:
            fig, ax = plt.subplots(figsize=(8, 4))
            if _stat_has_quality and df["year"].notna().any():
                cross = pd.crosstab(df["year"].dropna().astype(int), df["quality_flag"])
                cross.plot(kind="bar", stacked=True, ax=ax,
                          color=[colors_q.get(c, "#3498db") for c in cross.columns])
                ax.set_xlabel("Année")
                ax.set_ylabel("Nombre d'images")
                ax.set_title("Qualité par année")
                ax.legend(fontsize=8, bbox_to_anchor=(1, 1))
            else:
                ax.text(0.5, 0.5, "Données insuffisantes",
                        ha="center", va="center", transform=ax.transAxes, fontsize=11, color="gray")
                ax.set_axis_off()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with tab_files:
        st.subheader("Tailles de fichiers")
        sizes = df["file_size_bytes"].dropna()
        # Toujours rendre st.pyplot + st.columns(3) + métriques (anti-removeChild)
        fig, ax = plt.subplots(figsize=(10, 3))
        if not sizes.empty:
            ax.hist(sizes / 1024, bins=50, edgecolor="white", alpha=0.8, color="#9b59b6")
            ax.set_xlabel("Taille (Ko)")
            ax.set_ylabel("Fréquence")
            ax.set_title(f"Distribution — µ={sizes.mean()/1024:.0f} Ko, σ={sizes.std()/1024:.0f} Ko")
            ax.axvline(sizes.mean() / 1024, color="red", ls="--", label="Moyenne")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "Pas de données de taille de fichier",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_axis_off()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        c1, c2, c3 = st.columns(3)
        c1.metric("Taille totale",  f"{sizes.sum() / 1e6:.1f} Mo"   if not sizes.empty else "—")
        c2.metric("Taille moyenne", f"{sizes.mean() / 1024:.0f} Ko" if not sizes.empty else "—")
        c3.metric("Taille max",     f"{sizes.max() / 1024:.0f} Ko"  if not sizes.empty else "—")

    with tab_progress:
        st.subheader("Progression du pipeline")

        total = len(df)
        dl = int(df["downloaded"].sum())
        quality = int(df["quality_flag"].notna().sum())
        scored = int(df["anomaly_score"].notna().sum())

        progress_data = [
            ("Phase 1 — Indexation", total, total, "#3498db"),
            ("Phase 1 — Téléchargement", dl, total, "#2ecc71"),
            ("Phase 3 — Classification", quality, total, "#f39c12"),
            ("Phase 4 — Scoring", scored, total, "#e74c3c"),
        ]

        for label, done, tot, color in progress_data:
            pct = done / tot * 100 if tot > 0 else 0
            st.markdown(f"**{label}** — {done}/{tot} ({pct:.0f}%)")
            st.progress(pct / 100)

        if "on_disk" in df.columns:
            _n_disk_p = int(df["on_disk"].sum())
            _n_dl_p = int(df["downloaded"].sum())
            _phantom = _n_dl_p - _n_disk_p
            if _phantom > 0:
                st.info(
                    f"ℹ️ **{_phantom:,} entrées fantômes** dans l'index : marquées "
                    "`downloaded=True` mais le fichier n'existe pas sur disque.\n\n"
                    "Causes fréquentes : doublons macOS `' 2.jpg'`, images d'autres caméras "
                    "(Suki_Canon, etc.) enregistrées dans l'index mais jamais téléchargées, "
                    "fichiers déplacés ou supprimés.\n\n"
                    "La colonne **Sur disque ✓** est la mesure fiable pour la couverture."
                )

    with tab_coverage:
        st.subheader("🔍 Couverture pipeline par année")
        st.caption(
            "Ce tableau distingue **Indexées** (total dans l'index CSV), "
            "**Sur disque ✓** (fichiers réellement accessibles), et "
            "**Scorées PatchCore** (score calculé). "
            "C'est la vue fiable pour diagnostiquer les trous de couverture."
        )

        if df["year"].notna().any():
            _cov_rows = []
            for _yr in sorted(df["year"].dropna().unique()):
                _d = df[df["year"] == _yr]
                _n_tot = len(_d)
                _n_dl = int(_d["downloaded"].sum())
                _n_dsk = int(_d["on_disk"].sum()) if "on_disk" in df.columns else _n_dl
                _n_sc = int(_d["patchcore_score"].notna().sum()) if "patchcore_score" in df.columns else int(_d["anomaly_score"].notna().sum())
                _n_usp = int(_d["quality_flag"].eq("usable").sum())
                _cv = _n_sc / _n_dsk * 100 if _n_dsk > 0 else 0.0
                _cov_rows.append({
                    "Année": int(_yr),
                    "Indexées": _n_tot,
                    "Téléch. (index)": _n_dl,
                    "Sur disque ✓": _n_dsk,
                    "Scorées PatchCore": _n_sc,
                    "Usable": _n_usp,
                    "Couverture disque": f"{_cv:.0f}%",
                    "_cv_float": _cv,
                })

            _cov_df = pd.DataFrame(_cov_rows).drop(columns=["_cv_float"])
            st.dataframe(_cov_df, hide_index=True, use_container_width=True)

            st.info(
                "**Téléch. (index)** : marquées `downloaded=True` dans l'index — peut inclure des "
                "entrées fantômes.\n\n"
                "**Sur disque ✓** : fichiers réellement présents et lisibles au démarrage de l'app.\n\n"
                "**Couverture disque** = Scorées PatchCore ÷ Sur disque ✓"
            )

            try:
                _fig, _axes = plt.subplots(1, 2, figsize=(14, 4))
                _years_c = [int(r["Année"]) for r in _cov_rows]
                _x = range(len(_years_c))
                _idx_v = [r["Indexées"] for r in _cov_rows]
                _dsk_v = [r["Sur disque ✓"] for r in _cov_rows]
                _sc_v = [r["Scorées PatchCore"] for r in _cov_rows]
                _cv_v = [r["_cv_float"] for r in _cov_rows]

                # Graphique 1 : volumes
                _ax1 = _axes[0]
                _ax1.bar(_x, _idx_v, label="Indexées", color="#bdc3c7", width=0.6)
                _ax1.bar(_x, _dsk_v, label="Sur disque", color="#3498db", width=0.6)
                _ax1.bar(_x, _sc_v, label="Scorées", color="#2ecc71", width=0.6)
                _ax1.set_xticks(list(_x))
                _ax1.set_xticklabels([str(y) for y in _years_c], rotation=45, fontsize=9)
                _ax1.set_ylabel("Nombre d'images")
                _ax1.set_title("Indexées / Sur disque / Scorées")
                _ax1.legend(fontsize=8)

                # Graphique 2 : couverture %
                _ax2 = _axes[1]
                _bar_cols = [
                    "#2ecc71" if c >= 90 else "#f39c12" if c >= 50 else "#e74c3c"
                    for c in _cv_v
                ]
                _ax2.bar(_x, _cv_v, color=_bar_cols, edgecolor="white", width=0.6)
                _ax2.axhline(90, color="#27ae60", ls="--", alpha=0.7, label="Seuil 90%")
                _ax2.set_xticks(list(_x))
                _ax2.set_xticklabels([str(y) for y in _years_c], rotation=45, fontsize=9)
                _ax2.set_ylabel("Couverture disque (%)")
                _ax2.set_title("Couverture PatchCore par année")
                _ax2.set_ylim(0, 115)
                _ax2.legend(fontsize=8)
                for _xi, _val in zip(_x, _cv_v):
                    if _dsk_v[_xi] > 0:
                        _ax2.text(_xi, _val + 2, f"{_val:.0f}%", ha="center", fontsize=8)

                plt.tight_layout()
                st.pyplot(_fig)
                plt.close(_fig)
            except Exception as _exc_cov:
                st.warning(f"Graphique couverture indisponible : {_exc_cov}")

            _bad_years = [
                r["Année"] for r in _cov_rows
                if r["Sur disque ✓"] > 0 and r["_cv_float"] < 90
            ]
            if _bad_years:
                st.warning(
                    f"⚠️ Couverture < 90% pour les années : **{_bad_years}**\n\n"
                    "Pour rescorer ces années :\n"
                    "```bash\npython run_v1_pipeline.py --step patchcore\n```"
                )
        else:
            st.info("Aucune donnée temporelle disponible.")


# ==============================================================
# Page 8 — À propos
# ==============================================================

def page_a_propos() -> None:
    st.header("📖 À propos du projet")

    tab_proj, tab_pipeline, tab_sections, tab_guide = st.tabs([
        "🎯 Projet", "⚙️ Pipeline", "🗂️ Sections de l'app", "🔧 Guide fine-tuning"
    ])

    with tab_proj:
        st.markdown("""
    ## IA générative pour la surveillance volcanique du Merapi

    ### Objectif principal
    Développer une approche IA capable de :
    - **Détecter des anomalies volcaniques** sans supervision (DINOv2 + PatchCore)
    - **Reconstruire** des images vers un état « normal » pour isoler les anomalies (SD 1.5 img2img)
    - **Classifier** les événements volcaniques (pyroclastique / lave / nuage / normal)
    - **Anticiper** des événements (Early Warning via signal précurseur PatchCore)

    ---

    ### Contexte scientifique
    Stage de Master 2 — **Laboratoire Magmas et Volcans** (LMV, Clermont-Ferrand).

    ### Données
    - **Source** : réseau de surveillance VELI/TéléVolc (OPGC/LMV) — wwwobs.univ-bpclermont.fr
    - **Caméras** : Kalor, Suki (Canon EOS 1100D) — Merapi, Indonésie
    - **Période** : 2014–2024 (10 ans, ~76 000 images indexées)
    - **Résolution** : 4272×2848 px (brutes), 512×512 (prétraitées)

    ### Technologies utilisées
    Python 3.10+ · Streamlit · PyTorch · DINOv2 · scikit-learn · NumPy · Pandas · Matplotlib
    """)

    with tab_pipeline:
        st.markdown("""
    ### Pipeline de traitement complet

    ```
    Phase 1 : Scraping + Indexation (src/scraper.py, src/indexer.py)
               └→ data/index/index.csv (~76 820 lignes)

    Phase 2 : Téléchargement (src/scraper.py --step download)
               └→ data/raw/{année}/{mois:02d}/*.jpg

    Phase 3 : Prétraitement + Qualité (src/preprocessing.py, src/quality_filter.py)
               └→ data/processed/{année}/{mois:02d}/*.png
               └→ quality_flag : usable / dark / cloudy / corrupted

    Phase 4 : Scores baseline (src/baselines.py)
               └→ outputs/scores/baselines_{année}_{mois:02d}.csv

    Phase 5 : DINOv2 + PatchCore (src/models/patchcore_detector.py)
               └→ outputs/scores/patchcore_scores.csv (~76 821 lignes)

    Phase 6 : Features physiques (src/features/physical_features.py)
               └→ outputs/models/physical_features.csv

    Phase 7 : Classification volcanique (src/models/volcano_classifier.py)
               └→ outputs/models/volcano_clf.pkl

    Phase 8 : LoRA fine-tuning (train_lora_physics.py)
               └→ outputs/lora_merapi_physics/lora_merapi_physics_final/

    Phase 9 : Application Streamlit (app/streamlit_app.py)
    ```

    ### Commandes clés
    ```bash
    # Pipeline complet
    python run_full_pipeline.py

    # Étapes individuelles
    python run_full_pipeline.py --step download
    python run_full_pipeline.py --step patchcore
    python run_v1_pipeline.py --step features

    # Lancer l'app
    USE_TF=0 USE_TORCH=1 streamlit run app/streamlit_app.py
    ```
    """)

    with tab_sections:
        st.markdown("""
    ### Guide des sections de l'application

    | Section | Description | Données requises |
    |---------|-------------|-----------------|
    | **🏠 Accueil** | KPIs globaux, statistiques résumées | Index seul |
    | **🔍 Exploration** | Navigation chronologique des images | Index + images raw |
    | **⚠️ Anomalies** | Top images anormales (baselines + PatchCore) | Scores CSV |
    | **🖼️ Galerie** | Grille d'images raw vs prétraitées | Images raw/processed |
    | **🌋 Simulation** | Modèle d'écoulements de lave 3D | — |
    | **📊 Statistiques** | Distributions, EDA, corrélations | Index + scores |
    | **📖 À propos** | Cette page | — |
    | **🔬 DINOv2 + PatchCore** | Scores sémantiques, cartes d'attention | patchcore_scores.csv |
    | **⚡ Early Warning** | Signal précurseur avant événements BPPTKG | patchcore_scores.csv + events CSV |
    | **🧪 Analyse avancée** | Reconstruction img2img + carte d'anomalie | Images |
    | **🌋 Analyse volcanique avancée** | Classification, heatmap, timeline | patchcore_scores.csv |

    ---

    ### 🧪 Analyse avancée — Reconstruction par différence

    **Principe** : reconstruction par différence d'image pour isoler les zones suspectes.
    1. L'image réelle est chargée.
    2. Une reconstruction "normale" est générée (SD 1.5 si disponible, sinon flou gaussien).
    3. La différence |original − reconstruit| = **carte d'anomalie**.

    **Backend** :
    - `diffusers` disponible → SD 1.5 complet (~20–40s/image sur MPS, ~5s sur GPU)
    - Sinon → fallback flou gaussien (~instantané, résultats indicatifs)

    ---

    ### 🌋 Analyse volcanique avancée

    **Classification** (onglet "Classification") :
    - Modèle **RandomForest** si `outputs/models/volcano_clf.pkl` existe
    - Sinon **heuristique** basée sur seuils physiques calibrés :
      - `patchcore_score > 46.5` → pyroclastique (top 2%)
      - `is_night=True` + `bright_pixel_ratio > 2%` → lave (incandescence)
      - `entropy < 3.5` + `pixdiff < 0.03` → nuage (scène homogène)
      - Sinon → normal
    - ⚡ La classification est déclenchée **manuellement** (bouton) pour éviter de bloquer
      l'interface sur 76k images à chaque rechargement.

    **Heatmap spatiale** (onglet "Zones actives") :
    - Grille 16×16 simulant les 256 patches DINOv2-small (images 224×224, patch_size=14)
    - Si `physical_features.csv` absent → distribution uniforme du score par image
    - Si features présentes → pondération spatiale par gradient thermique, edge density

    **Timeline** (onglet "Timeline d'activité") :
    - Score moyen PatchCore agrégé par jour / semaine / mois
    - Détection des pics >μ+2σ

    **Comparaison** (onglet "Comparaison") :
    - Accord entre classification heuristique et modèle RF sur la période filtrée
    """)

    with tab_guide:
        st.markdown("""
    ### Guide de fine-tuning pour images volcaniques

    #### 1. Préparation du dataset
    | Critère | Recommandation |
    |---------|---------------|
    | Format | PNG uniquement (lossless) |
    | Taille | 200–500 images `quality_flag == "usable"` |
    | Résolution | 256×256 ou 512×512 |
    | Annotations | Prompts descriptifs (ex. `"Merapi volcano crater, normal activity"`) |

    #### 2. Paramètres LoRA recommandés
    | Paramètre | Valeur | Commentaire |
    |-----------|--------|-------------|
    | Learning rate | 1e-5 à 5e-6 | Faible pour éviter catastrophic forgetting |
    | Epochs | 50–200 | Early stopping sur loss |
    | Batch size | 2–4 | Limité VRAM (MPS 8–16 Go) |
    | Résolution | 512×512 | Cohérent avec preprocessing |
    | Mixed precision | fp32 (MPS) / fp16 (CUDA) | MPS ne supporte pas fp16 |

    #### 3. Pièges à éviter
    - ❌ Ne pas mélanger jour/nuit sans conditionnement explicite
    - ❌ `transformers>=5.x` est incompatible avec `diffusers 0.37.x` → garder `transformers<5`
    - ❌ `USE_TF=0 USE_TORCH=1` toujours requis (conflits protobuf TF/PyTorch)
    - ❌ Sur MPS : le premier batch prend ~150s (compilation JIT) — attendre

    #### 4. Évaluation
    - **FID** < 50 acceptable pour images spécialisées
    - **SSIM** vs images réelles du même créneau horaire
    - Inspection visuelle des textures volcaniques

    ### Références
    - Kelfoun & Druitt (2005) — Modélisation d'écoulements pyroclastiques
    - Ho et al. (2020) — Denoising Diffusion Probabilistic Models
    - Kingma & Welling (2014) — Auto-Encoding Variational Bayes
    - Rombach et al. (2022) — High-Resolution Image Synthesis with Latent Diffusion Models
    - Caron et al. (2021) — Self-Supervised Vision Transformers (DINOv2)
    - Roth et al. (2022) — Towards Total Recall in Industrial Anomaly Detection (PatchCore)
    """)


# ==============================================================
# Page 9 — DINOv2 + PatchCore
# ==============================================================

def page_patchcore(df: pd.DataFrame, config: dict) -> None:
    """Page DINOv2 + PatchCore : scores sémantiques, heatmap, espace features, pédagogie."""
    st.title("🔬 DINOv2 + PatchCore")
    st.markdown(
        "Détection d'anomalies **sans supervision** : features DINOv2-small + "
        "mémoire coreset. Plus robuste aux variations photométriques que les baselines pixel."
    )

    # ─── Chargement des scores PatchCore ─────────────────────────────────
    scores_path = PROJECT_ROOT / "outputs" / "scores" / "patchcore_scores.csv"
    if "patchcore_score" not in df.columns or df["patchcore_score"].notna().sum() == 0:
        if scores_path.exists():
            sc = safe_read_csv(scores_path)
            if not sc.empty and "filename" in sc.columns and "patchcore_score" in sc.columns:
                df = df.merge(sc[["filename", "patchcore_score"]], on="filename", how="left")

    has_scores = "patchcore_score" in df.columns and df["patchcore_score"].notna().sum() > 0

    if not has_scores:
        st.warning(
            "Aucun score PatchCore disponible. "
            "Lancez `python run_v1_pipeline.py --step patchcore` pour les calculer."
        )
        st.code("python run_v1_pipeline.py --step patchcore", language="bash")
        return

    df["patchcore_score"] = pd.to_numeric(df["patchcore_score"], errors="coerce")
    df_scored = df[df["patchcore_score"].notna()].copy()

    # ─── Logs console ────────────────────────────────────────────────────
    _n_dark_sc = int((df_scored["quality_flag"] == "dark").sum()) if "quality_flag" in df_scored.columns else 0
    _n_usable_sc = int((df_scored["quality_flag"] == "usable").sum()) if "quality_flag" in df_scored.columns else 0
    _mean_dark_sc = float(df_scored.loc[df_scored.get("quality_flag", pd.Series("", index=df_scored.index)) == "dark", "patchcore_score"].mean()) if _n_dark_sc > 0 else float("nan")
    _mean_usable_sc = float(df_scored.loc[df_scored.get("quality_flag", pd.Series("", index=df_scored.index)) == "usable", "patchcore_score"].mean()) if _n_usable_sc > 0 else float("nan")
    print(
        f"[PatchCore] total={len(df_scored)} | dark={_n_dark_sc}(mean={_mean_dark_sc:.3f}) | "
        f"usable={_n_usable_sc}(mean={_mean_usable_sc:.3f})"
    )

    # ─── Métriques clés ──────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Images scorées", f"{len(df_scored):,}")
    col2.metric("Score médian", f"{df_scored['patchcore_score'].median():.4f}")
    col3.metric("Score max", f"{df_scored['patchcore_score'].max():.4f}")
    threshold_90 = float(df_scored["patchcore_score"].quantile(0.90))
    n_anom = int((df_scored["patchcore_score"] > threshold_90).sum())
    col4.metric("Images > P90", f"{n_anom}")

    st.divider()

    # ─── Onglets dynamiques (dépendent des toggles sidebar) ──────────────
    _show_clusters = st.session_state.get("sidebar_show_clusters", True)
    _show_pedagogy = st.session_state.get("sidebar_show_pedagogy", True)

    _tab_names = ["📊 Scores & Anomalies", "🔥 Heatmap interactive"]
    if _show_clusters:
        _tab_names.append("🧬 Espace des features")
    if _show_pedagogy:
        _tab_names.append("📚 Comprendre PatchCore")

    _tabs = st.tabs(_tab_names)
    tab_scores = _tabs[0]
    tab_heatmap = _tabs[1]
    tab_cluster = _tabs[2] if _show_clusters else None
    tab_pedagogy = _tabs[-1] if _show_pedagogy else None

    # Pour les onglets masqués, utiliser des conteneurs no-op SÉPARÉS.
    # IMPORTANT : un seul st.empty() partagé entre deux tabs provoque le
    # bug removeChild (React essaie de mettre à jour le même nœud DOM
    # depuis deux chemins de rendu différents).
    if tab_cluster is None:
        tab_cluster = st.empty()   # nœud dédié — jamais affiché
    if tab_pedagogy is None:
        tab_pedagogy = st.empty()  # nœud dédié — jamais affiché

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — Scores & Anomalies
    # ══════════════════════════════════════════════════════════════════════
    with tab_scores:
        st.subheader("🎛️ Filtres")
        _pc_years = sorted(df_scored["year"].dropna().unique().astype(int))
        _pc_col_y, _pc_col_m, _pc_col_n = st.columns(3)
        with _pc_col_y:
            _pc_sel_year = st.selectbox(
                "Année", ["Toutes"] + [str(y) for y in _pc_years],
                key="pc_filter_year",
            )
        _df_pc_y = df_scored if _pc_sel_year == "Toutes" else df_scored[df_scored["year"] == int(_pc_sel_year)]
        with _pc_col_m:
            _pc_months = sorted(_df_pc_y["month"].dropna().unique().astype(int))
            _pc_sel_month = st.selectbox(
                "Mois", ["Tous"] + [f"{m:02d}" for m in _pc_months],
                key="pc_filter_month",
            )
        _df_pc_ym = _df_pc_y if _pc_sel_month == "Tous" else _df_pc_y[_df_pc_y["month"] == int(_pc_sel_month)]
        with _pc_col_n:
            _pc_max_n = max(1, len(_df_pc_ym))
            _pc_page_size = st.slider(
                "Images par page", 5, min(100, _pc_max_n), min(20, _pc_max_n),
                key="pc_page_size",
            )

        _df_pc_filtered = _df_pc_ym.sort_values("patchcore_score", ascending=False).reset_index(drop=True)
        _pc_n_total = len(_df_pc_filtered)
        _pc_n_pages = max(1, (_pc_n_total + _pc_page_size - 1) // _pc_page_size)

        _pc_page = st.slider(
            "Page", min_value=1, max_value=max(1, _pc_n_pages),
            value=1, key="pc_page_num", format=f"Page %d / {_pc_n_pages}",
        )
        _pc_start = (_pc_page - 1) * _pc_page_size
        _df_pc_page = _df_pc_filtered.iloc[_pc_start: _pc_start + _pc_page_size]

        st.caption(
            f"**{_pc_n_total:,}** images — Page {_pc_page}/{_pc_n_pages} "
            "(triées par score PatchCore ↓)"
        )
        st.info(
            "📊 La timeline interactive complète est dans **📊 Timeline PatchCore** (sidebar)."
        )
        st.divider()

        st.subheader("Images les plus anomales")
        _top_anom = _df_pc_page.copy()
        for _, row in _top_anom.head(6).iterrows():
            img_path = find_image_path(row)
            if img_path is not None:
                try:
                    from PIL import Image as _PILImage
                    img_pil = _PILImage.open(img_path).convert("RGB")
                    with st.container():
                        c1, c2 = st.columns([1, 3])
                        c1.image(img_pil, caption=f"Score: {row['patchcore_score']:.4f}", **{_IMG_WIDTH_KW: True})
                        c2.write(f"**{row.get('filename', '?')}**")
                        c2.write(f"Date: {row.get('year', '?')}-{row.get('month', '?'):02g}-{row.get('day', '?'):02g}")
                        c2.write(f"Qualité: `{row.get('quality_flag', '?')}`")
                        c2.write(f"Niveau : {_anomaly_level(float(row['patchcore_score']) / max(float(df_scored['patchcore_score'].max()), 1e-8))}")
                except Exception:
                    pass

        st.divider()

        # Carte d'attention DINOv2
        st.subheader("Carte d'attention DINOv2")
        st.info(
            "Sélectionnez une image pour visualiser où DINOv2 focalise son attention. "
            "Les zones chaudes indiquent les patches les plus discriminants."
        )
        filenames_top = _top_anom["filename"].tolist()
        sel_file = st.selectbox("Image à analyser", filenames_top, key="pc_attn_sel")

        # Réinitialiser le cache attention si l'image change
        if st.session_state.get("_attn_last_file") != sel_file:
            st.session_state["_attn_last_file"] = sel_file
            st.session_state.pop("_attn_orig", None)
            st.session_state.pop("_attn_overlay", None)
            st.session_state.pop("_attn_error", None)

        sel_row = _top_anom[_top_anom["filename"] == sel_file].iloc[0] if sel_file else None

        if sel_row is not None:
            img_path = find_image_path(sel_row)
            _attn_disabled = img_path is None
            _attn_btn = st.button(
                "Calculer la carte d'attention", key="btn_attn",
                disabled=_attn_disabled,
            )
            if _attn_disabled:
                st.caption("Image introuvable sur disque.")
            elif _attn_btn:
                with st.spinner("DINOv2 en cours (1re fois ~30s)…"):
                    try:
                        from src.features.attention_maps import get_dino_attention_map, overlay_attention_on_image
                        from PIL import Image as _PILImage
                        import numpy as _np
                        img_pil = _PILImage.open(img_path).convert("RGB")
                        img_np = _np.array(img_pil)
                        img_bgr = img_np[:, :, ::-1].copy()
                        attn = get_dino_attention_map(img_bgr)
                        img_gray = _np.mean(img_np, axis=2).astype(_np.uint8)
                        overlay = overlay_attention_on_image(img_gray, attn, alpha=0.55)
                        overlay_rgb = overlay[:, :, ::-1] if overlay.ndim == 3 else overlay
                        # Stocker en session_state — persiste sur les re-runs suivants
                        st.session_state["_attn_orig"] = img_pil
                        st.session_state["_attn_overlay"] = overlay_rgb
                        st.session_state.pop("_attn_error", None)
                    except Exception as exc:
                        st.session_state["_attn_error"] = str(exc)
                        st.session_state.pop("_attn_orig", None)
                        st.session_state.pop("_attn_overlay", None)

        # Zone de rendu TOUJOURS présente dans le DOM — évite le removeChild React
        _attn_placeholder = st.container()
        with _attn_placeholder:
            if st.session_state.get("_attn_error"):
                st.error(f"Erreur : {st.session_state['_attn_error']}")
            elif st.session_state.get("_attn_orig") is not None:
                col_orig, col_attn = st.columns(2)
                col_orig.image(st.session_state["_attn_orig"], caption="Image originale", **{_IMG_WIDTH_KW: True})
                col_attn.image(st.session_state["_attn_overlay"], caption="Carte d'attention", **{_IMG_WIDTH_KW: True})

        st.divider()

        # Rapport d'évaluation
        report_path = PROJECT_ROOT / "outputs" / "scores" / "evaluation_report.csv"
        if report_path.exists():
            st.subheader("Rapport d'évaluation")
            with st.expander("🔍 Diagnostic distribution classes & scores", expanded=False):
                _diag_dark = df_scored[df_scored["quality_flag"].eq("dark")] if "quality_flag" in df_scored.columns else pd.DataFrame()
                _diag_usable = df_scored[df_scored["quality_flag"].eq("usable")] if "quality_flag" in df_scored.columns else pd.DataFrame()
                _diag_n_wl = int(df_scored["weather_label"].notna().sum()) if "weather_label" in df_scored.columns else 0
                dc1, dc2, dc3, dc4 = st.columns(4)
                dc1.metric("dark (proxy anomalie)", f"{len(_diag_dark):,}")
                dc2.metric("usable (normal)", f"{len(_diag_usable):,}")
                dc3.metric("weather_label", f"{_diag_n_wl:,}")
                dc4.metric("Prévalence dark", f"{len(_diag_dark) / max(len(_diag_dark)+len(_diag_usable),1):.1%}")
                if len(_diag_dark) > 0 and len(_diag_usable) > 0:
                    _dm = _diag_dark["patchcore_score"].mean()
                    _um = _diag_usable["patchcore_score"].mean()
                    st.markdown(
                        f"| Groupe | n | Moyenne | Médiane | P90 |\n"
                        f"|--------|---|---------|---------|-----|\n"
                        f"| dark | {len(_diag_dark)} | {_dm:.3f} | {_diag_dark['patchcore_score'].median():.3f} | {_diag_dark['patchcore_score'].quantile(0.9):.3f} |\n"
                        f"| usable | {len(_diag_usable)} | {_um:.3f} | {_diag_usable['patchcore_score'].median():.3f} | {_diag_usable['patchcore_score'].quantile(0.9):.3f} |"
                    )
                    if _dm > _um:
                        st.success(f"✅ Scores correctement orientés : dark ({_dm:.3f}) > usable ({_um:.3f}).")
                    else:
                        st.error(f"🔴 Inversion détectée : dark ({_dm:.3f}) < usable ({_um:.3f}).")
                st.markdown(
                    "**Note F1 évaluation** : calculé via `compute_f1_patchcore_binary` "
                    "(seuil P90 global, y_true = quality_flag==\"dark\"). "
                    "Ne dépend pas de weather_predictions — toujours disponible après `--step evaluate`."
                )

            rpt = safe_read_csv(report_path)
            if not rpt.empty:
                c1, c2, c3 = st.columns(3)
                _auc_base = rpt["auc_pr_baseline"].iloc[0] if "auc_pr_baseline" in rpt.columns else float("nan")
                _auc_pc = rpt["auc_pr_patchcore"].iloc[0] if "auc_pr_patchcore" in rpt.columns else float("nan")
                _delta = rpt["delta_auc_pr"].iloc[0] if "delta_auc_pr" in rpt.columns else float("nan")
                c1.metric("AUC-PR Baseline", f"{_auc_base:.4f}" if pd.notna(_auc_base) else "N/A")
                c2.metric("AUC-PR PatchCore", f"{_auc_pc:.4f}" if pd.notna(_auc_pc) else "N/A")
                c3.metric("Amélioration", f"{_delta:+.4f}" if pd.notna(_delta) else "N/A",
                          delta=f"{_delta:.4f}" if pd.notna(_delta) else None)
                st.dataframe(rpt, use_container_width=True, key="pc_eval_report_table")

                _effect = rpt["effect_size"].iloc[0] if "effect_size" in rpt.columns else float("nan")
                _mwp = rpt["mann_whitney_p"].iloc[0] if "mann_whitney_p" in rpt.columns else float("nan")
                _improv = rpt["improvement_pct"].iloc[0] if "improvement_pct" in rpt.columns else float("nan")
                _auc_roc_val = rpt["auc_roc_patchcore"].iloc[0] if "auc_roc_patchcore" in rpt.columns else float("nan")

                # ── F1 binaire PatchCore ──────────────────────────────────────
                st.markdown("**📊 F1 — Classification binaire (dark vs usable)**")
                _f1_data = load_f1_metrics(df_scored)
                _f1_src = _f1_data["source"]

                if _f1_src == "unavailable":
                    st.warning(
                        "⚠️ F1 non disponible — exécutez : "
                        "`python run_v1_pipeline.py --step evaluate`"
                    )
                else:
                    _lbl_src = "📄 evaluation_report.csv" if _f1_src == "report" else "⚡ calculé dynamiquement (P90)"
                    fc1, fc2, fc3 = st.columns(3)
                    fc1.metric("F1-macro", f"{_f1_data['f1_macro']:.4f}", help="Moyenne F1 anomalie + F1 normal")
                    fc2.metric("F1 anomalie (dark)", f"{_f1_data['f1_anomaly']:.4f}", help="Rappel/précision sur les images dark")
                    fc3.metric("F1 normal (usable)", f"{_f1_data['f1_normal']:.4f}", help="Rappel/précision sur les images normales")
                    st.caption(_lbl_src)
                    if _f1_data["f1_macro"] < 0.5:
                        st.info(
                            "ℹ️ F1-macro < 0.5 — normal avec peu d'images dark annotées. "
                            "Les métriques AUC-PR et Mann-Whitney sont plus fiables sur données déséquilibrées."
                        )
                    else:
                        st.success(f"✅ F1-macro = {_f1_data['f1_macro']:.4f}")

                # ── Interprétation automatique (AUC / MW) ────────────────────
                st.markdown("**🔍 Interprétation automatique**")
                if pd.notna(_effect):
                    if _effect > 0:
                        st.success(f"✅ Signal détecté : dark > usable (RBC = {_effect:.3f} > 0).")
                    else:
                        st.warning(f"⚠️ Effect-size négatif ({_effect:.3f}) — vérifier la convention de score.")
                if pd.notna(_mwp) and _mwp < 0.05:
                    st.success(f"✅ Signal significatif (Mann-Whitney p = {_mwp:.2e} ≪ 0.05).")
                if pd.notna(_improv):
                    st.success(f"✅ PatchCore **{1 + _improv/100:.1f}× mieux** que le classifieur aléatoire (+{_improv:.0f}% AUC-PR).")
                if pd.notna(_auc_roc_val):
                    if _auc_roc_val > 0.5:
                        st.success(f"✅ AUC-ROC = {_auc_roc_val:.4f} > 0.5 — direction des scores correcte.")
                    else:
                        st.error(f"🔴 AUC-ROC = {_auc_roc_val:.4f} < 0.5 → scores inversés, tester `score = -score`.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — Heatmap interactive
    # ══════════════════════════════════════════════════════════════════════
    with tab_heatmap:
        st.subheader("🔥 Heatmap PatchCore interactive")
        st.markdown(
            "Visualisation spatiale des scores d'anomalie patch par patch. "
            "Chaque case de la grille **16 × 16** correspond à un patch de l'image ; "
            "sa couleur indique l'éloignement de ce patch par rapport à la mémoire normale."
        )

        col_ctrl, col_img_sel = st.columns([1, 2])
        with col_ctrl:
            _sb_opacity = st.session_state.get("sidebar_hm_opacity", 0.45)
            _sb_cmap_val = st.session_state.get("sidebar_hm_cmap", "Hot")
            _hm_opacity = st.slider(
                "Opacité heatmap", 0.0, 1.0, float(_sb_opacity), 0.05, key="hm_opacity"
            )
            _cmaps = ["Hot", "Viridis", "RdYlBu_r", "Plasma", "Inferno", "YlOrRd"]
            _cmap_idx = _cmaps.index(_sb_cmap_val) if _sb_cmap_val in _cmaps else 0
            _hm_cmap = st.selectbox(
                "Colormap",
                _cmaps,
                index=_cmap_idx,
                key="hm_cmap",
            )

        with col_img_sel:
            _hm_years = ["Toutes"] + [str(y) for y in sorted(df_scored["year"].dropna().unique().astype(int))]
            _hm_year = st.selectbox("Filtrer par année", _hm_years, key="hm_year")
            _df_hm = df_scored if _hm_year == "Toutes" else df_scored[df_scored["year"] == int(_hm_year)]
            _df_hm_top = _df_hm.sort_values("patchcore_score", ascending=False).head(100)
            _hm_filenames = _df_hm_top["filename"].tolist()
            _hm_sel = st.selectbox(
                "Image à analyser (top 100 scores)",
                _hm_filenames,
                key="hm_sel_file",
            )

        _hm_row = _df_hm_top[_df_hm_top["filename"] == _hm_sel].iloc[0] if _hm_sel else None
        _hm_btn = st.button(
            "🔥 Calculer la heatmap PatchCore",
            key="btn_heatmap",
            disabled=(_hm_row is None),
        )

        if _hm_btn and _hm_row is not None:
            _hm_img_path = find_image_path(_hm_row)
            if _hm_img_path is None:
                st.warning("Image introuvable sur disque.")
            else:
                with st.spinner("Chargement DINOv2 + calcul des scores patch…"):
                    try:
                        from PIL import Image as _PILImage
                        detector = _load_patchcore_detector()
                        if detector is None:
                            st.error(
                                "Modèle PatchCore introuvable. "
                                "Lancez `python run_v1_pipeline.py --step patchcore`."
                            )
                        else:
                            img_pil = _PILImage.open(_hm_img_path).convert("RGB")
                            img_arr = np.array(img_pil)
                            _, patch_map = detector.score_image(_hm_img_path)
                            fig_hm = build_heatmap_figure(patch_map, img_arr, _hm_opacity, _hm_cmap)
                            st.plotly_chart(fig_hm, use_container_width=True, key="pc_hm_img_chart")

                            # Tableau de détail par patch
                            _score_flat = patch_map.flatten()
                            _patch_df = pd.DataFrame({
                                "Patch (ligne, col)": [
                                    f"[{i//16},{i%16}]" for i in range(len(_score_flat))
                                ],
                                "Score normalisé": _score_flat.round(4),
                                "Niveau": [_anomaly_level(s) for s in _score_flat],
                            }).sort_values("Score normalisé", ascending=False)

                            with st.expander("📋 Détail des 20 patches les plus anomaux"):
                                st.dataframe(_patch_df.head(20), hide_index=True,
                                             use_container_width=True, key="pc_hm_patch_table")

                            # Histogramme des scores
                            _col_hs1, _col_hs2 = st.columns(2)
                            with _col_hs1:
                                fig_hist, ax_hist = plt.subplots(figsize=(5, 2.5))
                                ax_hist.hist(_score_flat, bins=20, color="#e74c3c", edgecolor="white", alpha=0.8)
                                ax_hist.set_xlabel("Score patch")
                                ax_hist.set_ylabel("Fréquence")
                                ax_hist.set_title("Distribution des scores patches")
                                plt.tight_layout()
                                st.pyplot(fig_hist)
                                plt.close(fig_hist)
                            with _col_hs2:
                                _n_crit = int((_score_flat > 0.8).sum())
                                _n_forte = int(((_score_flat > 0.6) & (_score_flat <= 0.8)).sum())
                                _n_mod = int(((_score_flat > 0.4) & (_score_flat <= 0.6)).sum())
                                _n_norm = int((_score_flat <= 0.4).sum())
                                st.markdown(f"""
                                **Répartition des niveaux** (256 patches) :
                                - ⚠️ Critique (>0.8) : **{_n_crit}** patches
                                - 🔴 Forte (0.6–0.8) : **{_n_forte}** patches
                                - 🟠 Modérée (0.4–0.6) : **{_n_mod}** patches
                                - 🟢 Normal (≤0.4) : **{_n_norm}** patches
                                """)
                    except Exception as exc:
                        st.error(f"Erreur lors du calcul heatmap : {exc}")
        else:
            st.info(
                "Sélectionnez une image et cliquez sur **🔥 Calculer la heatmap PatchCore**. "
                "La 1re exécution charge DINOv2 (~30s) ; les suivantes sont quasi-instantanées."
            )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 — Espace des features
    # ══════════════════════════════════════════════════════════════════════
    with tab_cluster:
        st.subheader("🧬 Espace latent DINOv2 (PCA)")
        st.markdown(
            "Projection 2D du **coreset** (mémoire des images normales) par PCA. "
            "Chaque point bleu = un vecteur de feature stocké. "
            "Si vous analysez une image, ses patches apparaissent en rouge/jaune selon leur score."
        )

        _coreset_path = PROJECT_ROOT / "outputs" / "models" / "patchcore.npz"
        if not _coreset_path.exists():
            st.warning("Coreset introuvable (`outputs/models/patchcore.npz`). Lancez `--step patchcore`.")
        else:
            try:
                _npz = np.load(str(_coreset_path))
                _coreset_arr = _npz["coreset"]   # (N_coreset, 384)

                # Option : analyser une image pour voir ses patches
                _cl_years = ["Toutes"] + [str(y) for y in sorted(df_scored["year"].dropna().unique().astype(int))]
                _cl_year = st.selectbox("Filtrer par année", _cl_years, key="cl_year")
                _df_cl = df_scored if _cl_year == "Toutes" else df_scored[df_scored["year"] == int(_cl_year)]
                _df_cl_top = _df_cl.sort_values("patchcore_score", ascending=False).head(50)

                _cl_filenames = ["— Aucune (coreset seul) —"] + _df_cl_top["filename"].tolist()
                _cl_sel = st.selectbox("Image à projeter (optionnel)", _cl_filenames, key="cl_sel_file")

                _show_coreset_only = _cl_sel == "— Aucune (coreset seul) —"
                _cl_btn_label = "🔍 Visualiser l'espace des features" if _show_coreset_only else "🔍 Projeter l'image dans le coreset"
                _cl_btn = st.button(_cl_btn_label, key="btn_cluster")

                # Réinitialiser si la sélection change
                if st.session_state.get("_cl_last_sel") != _cl_sel:
                    st.session_state["_cl_last_sel"] = _cl_sel
                    st.session_state.pop("_cl_result", None)
                    st.session_state.pop("_cl_caption", None)
                    st.session_state.pop("_cl_error", None)

                if _cl_btn:
                    if _show_coreset_only:
                        with st.spinner("PCA sur le coreset…"):
                            try:
                                fig_cl = build_cluster_figure(_coreset_arr)
                                st.session_state["_cl_result"] = fig_cl
                                st.session_state["_cl_caption"] = (
                                    f"Coreset : {len(_coreset_arr):,} vecteurs × {_coreset_arr.shape[1]} dimensions. "
                                    "Chaque point = un patch feature stocké dans la mémoire normale."
                                )
                                st.session_state.pop("_cl_error", None)
                            except Exception as exc:
                                st.session_state["_cl_error"] = str(exc)
                    else:
                        _cl_row = _df_cl_top[_df_cl_top["filename"] == _cl_sel].iloc[0]
                        _cl_img_path = find_image_path(_cl_row)
                        if _cl_img_path is None:
                            st.warning("Image introuvable sur disque.")
                        else:
                            with st.spinner("DINOv2 + PCA en cours…"):
                                try:
                                    from src.features.attention_maps import get_dino_patch_features
                                    _query_feats = get_dino_patch_features(_cl_img_path)
                                    _cl_detector = _load_patchcore_detector()
                                    if _cl_detector is not None:
                                        _, _pm = _cl_detector.score_image(_cl_img_path)
                                        _patch_scores_flat = _pm.flatten()
                                    else:
                                        _patch_scores_flat = None
                                    fig_cl = build_cluster_figure(
                                        _coreset_arr, _query_feats, _patch_scores_flat
                                    )
                                    st.session_state["_cl_result"] = fig_cl
                                    st.session_state["_cl_caption"] = (
                                        f"**{_cl_sel}** — score global : {float(_cl_row['patchcore_score']):.4f}. "
                                        "Les points rouges = patches les plus éloignés de la mémoire normale."
                                    )
                                    st.session_state.pop("_cl_error", None)
                                except Exception as exc:
                                    st.session_state["_cl_error"] = str(exc)

                # Zone de rendu TOUJOURS présente dans le DOM — évite le removeChild React
                _cl_display = st.container()
                with _cl_display:
                    if st.session_state.get("_cl_error"):
                        # Erreur : toujours rendre un chart avec l'erreur en annotation (anti-removeChild)
                        _fig_err = go.Figure()
                        _fig_err.add_annotation(
                            text=f"❌ {st.session_state['_cl_error']}",
                            xref="paper", yref="paper", x=0.5, y=0.5,
                            showarrow=False, font=dict(size=12, color="#e74c3c"),
                        )
                        _fig_err.update_layout(height=350, xaxis_visible=False, yaxis_visible=False)
                        st.plotly_chart(_fig_err, use_container_width=True, key="cl_pca_error_chart")
                    elif st.session_state.get("_cl_result") is not None:
                        st.plotly_chart(
                            st.session_state["_cl_result"],
                            use_container_width=True,
                            key="cl_pca_result_chart",
                        )
                        if st.session_state.get("_cl_caption"):
                            st.caption(st.session_state["_cl_caption"])
                    else:
                        # Pas encore lancé : toujours rendre un chart avec instruction (anti-removeChild)
                        _fig_hint = go.Figure()
                        _fig_hint.add_annotation(
                            text="Cliquez sur 🔍 Visualiser pour afficher la projection PCA du coreset.",
                            xref="paper", yref="paper", x=0.5, y=0.5,
                            showarrow=False, font=dict(size=13, color="gray"),
                        )
                        _fig_hint.update_layout(height=350, xaxis_visible=False, yaxis_visible=False)
                        st.plotly_chart(_fig_hint, use_container_width=True, key="cl_pca_hint_chart")
            except Exception as exc_npz:
                st.error(f"Erreur de chargement du coreset : {exc_npz}")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4 — Comprendre PatchCore
    # ══════════════════════════════════════════════════════════════════════
    with tab_pedagogy:
        st.subheader("📚 Comprendre PatchCore — Guide interactif")
        st.markdown(
            "PatchCore est un algorithme de détection d'anomalies **sans supervision**, "
            "qui n'a besoin que d'images **normales** pour apprendre."
        )

        st.markdown("---")
        st.markdown("### 🔁 Pipeline en 5 étapes")

        step_cols = st.columns(5)
        steps = [
            ("1️⃣", "Extraction\nde patches", "L'image est découpée en grille 16×16 = 256 patches."),
            ("2️⃣", "Encodage\nDINOv2", "Chaque patch → vecteur 384-dim (features sémantiques)."),
            ("3️⃣", "Mémoire\ncoreset", "Lors de l'entraînement : on stocke N vecteurs représentatifs des images normales."),
            ("4️⃣", "Distance\nk-NN", "À l'inférence : distance de chaque patch au voisin le plus proche dans le coreset."),
            ("5️⃣", "Score\nglobal", "Score = max des distances. Carte = distances remises en forme 16×16."),
        ]
        for col, (icon, title, desc) in zip(step_cols, steps):
            col.metric(icon + " " + title.replace("\n", " "), "")
            col.caption(desc)

        st.markdown("---")
        try:
            fig_pipe, axes = plt.subplots(1, 5, figsize=(16, 3))
            titles = ["Image\noriginale", "Grille 16×16\nde patches", "Features\nDINOv2\n(384-dim)", "Mémoire\nnormale\n(coreset)", "Heatmap\nanomalie"]
            colors = ["#2980b9", "#27ae60", "#8e44ad", "#e67e22", "#e74c3c"]
            icons_txt = ["🖼️", "⬛", "📐", "🗃️", "🔥"]
            for ax, title, color, icon in zip(axes, titles, colors, icons_txt):
                ax.add_patch(plt.Rectangle((0.05, 0.15), 0.9, 0.7, color=color, alpha=0.85))
                ax.text(0.5, 0.55, icon, ha="center", va="center", fontsize=20)
                ax.text(0.5, 0.08, title, ha="center", va="top", fontsize=7.5, wrap=True)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis("off")
                if ax != axes[-1]:
                    ax.annotate("", xy=(1.05, 0.5), xytext=(0.95, 0.5),
                                xycoords="axes fraction", textcoords="axes fraction",
                                arrowprops=dict(arrowstyle="->", color="#555", lw=2))
            plt.tight_layout(pad=0.5)
            st.pyplot(fig_pipe)
            plt.close(fig_pipe)
        except Exception:
            pass

        st.markdown("---")
        st.markdown("### 💡 Concepts clés")
        col_k1, col_k2, col_k3 = st.columns(3)
        with col_k1:
            st.info(
                "**Pourquoi DINOv2 ?**\n\n"
                "DINOv2 produit des features **sémantiques** : robustes aux changements "
                "d'éclairage, de nuages, de brume. Un patch de nuage ressemble à un autre "
                "patch de nuage, quelle que soit la luminosité."
            )
        with col_k2:
            st.info(
                "**Pourquoi le max ?**\n\n"
                "Le score global = `max(distances)`. Cela permet de détecter "
                "une anomalie **localisée** même si le reste de l'image est normal. "
                "Un seul patch très éloigné suffit à lever l'alarme."
            )
        with col_k3:
            st.info(
                "**Qu'est-ce que le coreset ?**\n\n"
                "Un sous-ensemble représentatif des features d'entraînement. "
                "Il compresse l'information pour limiter la mémoire (~10k vecteurs "
                "au lieu de millions). Calculé par sélection greedy (couverture maximale)."
            )

        st.markdown("---")
        st.markdown("### 📊 Paramètres de ce modèle")
        _npz_path = PROJECT_ROOT / "outputs" / "models" / "patchcore.npz"
        if _npz_path.exists():
            _npz_info = np.load(str(_npz_path))
            _cs = _npz_info["coreset"]
            _col_p1, _col_p2, _col_p3 = st.columns(3)
            _col_p1.metric("Taille du coreset", f"{len(_cs):,} vecteurs")
            _col_p2.metric("Dimension features", f"{_cs.shape[1]}")
            _col_p3.metric("Espace mémoire", f"{_cs.nbytes / 1e6:.1f} Mo")
            st.caption(
                f"Backbone : DINOv2-small | Grille patches : 16×16 | "
                f"k-NN : k=1 (distance au plus proche voisin)"
            )
        else:
            st.warning("Coreset non trouvé — les paramètres s'afficheront après l'entraînement.")

        st.markdown("---")
        st.markdown("### 🔬 Pour aller plus loin")
        st.markdown("""
        - **Paper** : *Towards Total Recall in Industrial Anomaly Detection* (Roth et al., 2022) — [arxiv.org/abs/2106.08265](https://arxiv.org/abs/2106.08265)
        - **DINOv2** : *DINOv2: Learning Robust Visual Features without Supervision* (Oquab et al., 2023)
        - **PatchCore original** utilise un ResNet WideResNet-50 — cette implémentation utilise **DINOv2-small** pour ses features sémantiques plus adaptées aux scènes naturelles.
        - La normalisation `(patch_map - min) / (max - min)` est appliquée **par image** : le score max = 1 ne signifie pas que l'image est très anomale en absolu, mais que ce patch est le plus éloigné *dans cette image*.
        """)





# ==============================================================
# Page 10 — Timeline PatchCore interactive
# ==============================================================

_TIMELINE_SCRIPT = PROJECT_ROOT / "outputs" / "figures" / "plot_patchcore_timeline_interactive.py"
_TIMELINE_HTML_PATH = PROJECT_ROOT / "outputs" / "figures" / "patchcore_timeline_interactive.html"


@st.cache_data(show_spinner=False, ttl=600)
def _build_timeline_figure(data_mtime: float) -> tuple:  # noqa: ARG001
    """
    Construit la figure Plotly de la timeline PatchCore.

    Le paramètre data_mtime (timestamp de l'index CSV) sert de clé de cache :
    toute modification des données invalide automatiquement la figure.

    Returns:
        (fig, is_synthetic, n_images)
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("_plot_tl", str(_TIMELINE_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    df = mod.load_data()
    is_synth = df is None
    if is_synth:
        df = mod.synthetic_data()
    return mod.build_figure(df, is_synth), is_synth, len(df)


def page_timeline_patchcore() -> None:
    """Page dédiée à la timeline interactive PatchCore."""
    st.title("📊 Timeline interactive des anomalies PatchCore")
    st.markdown(
        "Exploration visuelle des **scores d'anomalie PatchCore** sur l'ensemble "
        "des images du Merapi (2014–2025). Filtrage dynamique par **année** et "
        "par **mois** via les dropdowns du graphique."
    )

    # ── Barre d'information ──────────────────────────────────────────────────
    st.info(
        "**Comment utiliser les filtres ?**\n\n"
        "- 📅 **Dropdown Année** (haut gauche) : restreint l'axe temporel à "
        "l'année choisie.\n"
        "- 📆 **Dropdown Mois** (haut centre) : masque les autres mois.\n"
        "- **Combinaison** : *2018* + *Mai* → uniquement mai 2018.\n"
        "- ↺ **Reset** (haut droite) : réinitialise les deux filtres.\n"
        "- **Rangeslider** (barre sous le graphique) : sélection libre.\n"
        "- 📷 **Export PNG** : icône appareil photo dans la barre d'outils Plotly."
    )

    # ── Légende et seuils ───────────────────────────────────────────────────
    with st.expander("ℹ️ Légende et seuils", expanded=False):
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(
                "**Qualité des images (couleur des points)**\n\n"
                "| Symbole | Type | Interprétation |\n"
                "|---------|------|----------------|\n"
                "| 🔵 cercle | Usable | Conditions normales |\n"
                "| 🔴 losange | Dark | Anomalie volcanique potentielle |\n"
                "| 🟠 carré | Cloudy | Couverture nuageuse |\n"
                "| 🟣 ×  | Corrupted | Fichier dégradé |"
            )
        with col_r:
            st.markdown(
                "**Seuil P90 = 49.975**\n\n"
                "Un score **supérieur au P90** (90e percentile) signale une image "
                "statistiquement anormale. Lors de l'éruption de mai 2018, une "
                "proportion inhabituellement élevée d'images dépasse ce seuil.\n\n"
                "**Ligne jaune** = moyenne mensuelle des scores.\n\n"
                "**Lignes verticales** = événements volcaniques annotés."
            )

    st.markdown("---")

    # ── Bouton rafraîchir ────────────────────────────────────────────────────
    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        if st.button("🔄 Rafraîchir", key="btn_refresh_timeline"):
            _build_timeline_figure.clear()
            st.rerun()

    # ── Vérification du script de génération ────────────────────────────────
    if not _TIMELINE_SCRIPT.exists():
        st.error(
            "⚠️ Script de génération introuvable.\n\n"
            f"Chemin attendu : `{_TIMELINE_SCRIPT}`"
        )
        return

    # ── Clé de cache : timestamp de l'index CSV (ou 0 si absent) ────────────
    _idx_csv = PROJECT_ROOT / "data" / "index" / "index.csv"
    _data_mtime = _idx_csv.stat().st_mtime if _idx_csv.exists() else 0.0

    # ── Construction de la figure (cachée) ──────────────────────────────────
    try:
        fig_tl, is_synthetic, n_images = _build_timeline_figure(_data_mtime)
    except Exception as exc:
        st.error(f"Erreur lors de la génération du graphique : {exc}")
        return

    with col_info:
        label = "synthétiques" if is_synthetic else "réelles"
        st.caption(f"{n_images:,} images {label}")

    if is_synthetic:
        st.warning(
            "⚠️ **Données synthétiques** — `data/index/index.csv` ne contient pas "
            "de colonne `patchcore_score`. Lancez d'abord :\n\n"
            "```bash\npython run_v1_pipeline.py --step patchcore\n```"
        )

    # ── Affichage via st.plotly_chart (pas d'iframe/CDN) ────────────────────
    st.plotly_chart(fig_tl, use_container_width=True, key="timeline_patchcore_main")

    # ── Téléchargement HTML (optionnel) ─────────────────────────────────────
    if _TIMELINE_HTML_PATH.exists():
        st.download_button(
            label="⬇️ Télécharger le graphique HTML",
            data=_TIMELINE_HTML_PATH.read_bytes(),
            file_name="patchcore_timeline_interactive.html",
            mime="text/html",
            key="dl_timeline_html",
        )


# ==============================================================
# Page 11 — Early Warning
# ==============================================================

def page_early_warning(df: pd.DataFrame) -> None:
    """Page Early Warning : précurseurs d'événements volcaniques."""
    st.title("⚡ Early Warning")
    st.markdown(
        "Validation du **signal précurseur** : les scores d'anomalie augmentent-ils "
        "significativement avant les événements documentés par le BPPTKG ?"
    )

    # ─── Sélection du score ──────────────────────────────────────────────
    available_scores = [c for c in ["patchcore_score", "anomaly_score"] if c in df.columns and df[c].notna().sum() > 0]
    if not available_scores:
        # Essayer de charger les scores PatchCore depuis le CSV
        scores_path = PROJECT_ROOT / "outputs" / "scores" / "patchcore_scores.csv"
        if scores_path.exists():
            sc = safe_read_csv(scores_path)
            if not sc.empty and "patchcore_score" in sc.columns:
                df = df.merge(sc[["filename", "patchcore_score"]], on="filename", how="left")
                df["patchcore_score"] = pd.to_numeric(df["patchcore_score"], errors="coerce")
                if df["patchcore_score"].notna().sum() > 0:
                    available_scores = ["patchcore_score"]

    if not available_scores:
        st.warning(
            "Aucun score disponible. "
            "Lancez `python run_v1_pipeline.py --step patchcore` d'abord."
        )
        st.code("python run_v1_pipeline.py --step patchcore", language="bash")
        return

    score_col = st.selectbox("Score à analyser", available_scores, key="ew_score_col")

    for c in ["year", "month", "day", "hour", "minute", "second"]:
        df[c] = pd.to_numeric(df.get(c, pd.Series(0, index=df.index)), errors="coerce").fillna(0).astype(int)
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    # ─── Fichier événements ───────────────────────────────────────────────
    events_path = PROJECT_ROOT / "data" / "events" / "merapi_events_2014_2018.csv"
    if not events_path.exists():
        st.error(f"Fichier d'événements introuvable : `{events_path}`")
        return

    events = safe_read_csv(events_path)
    if events.empty:
        st.error(f"Fichier d'événements vide ou illisible : `{events_path}`")
        return
    events["date"] = pd.to_datetime(events["date"])

    # ─── Scores précurseurs depuis CSV si disponible ──────────────────────
    prec_path = PROJECT_ROOT / "outputs" / "scores" / "early_warning_precursors.csv"
    perm_path = PROJECT_ROOT / "outputs" / "scores" / "early_warning_permutation.csv"

    if prec_path.exists():
        precursors = safe_read_csv(prec_path)
        if not precursors.empty:
            precursors = precursors[precursors["score_col"] == score_col]
    else:
        with st.spinner("Calcul des scores précurseurs..."):
            try:
                from src.evaluation.early_warning import EarlyWarningAnalyzer
                analyzer = EarlyWarningAnalyzer(events_path)
                precursors = analyzer.compute_precursor_scores(df, score_col=score_col)
            except Exception as exc:
                st.error(f"Erreur calcul précurseurs : {exc}")
                precursors = pd.DataFrame()

    if not precursors.empty:
        # ─── Tableau résumé ───────────────────────────────────────────────
        st.subheader("Scores précurseurs par événement")

        help_tooltip(
            "Tableau précurseurs",
            """
| Colonne | Définition |
|---|---|
| **event_date** | Date de l'événement documenté (BPPTKG) |
| **lead_days** | Fenêtre précurseur (J-1, J-3, J-7 avant l'événement) |
| **mean_score** | Score PatchCore moyen dans la fenêtre précurseur |
| **background_score** | Score moyen en dehors des fenêtres (référence "calme") |
| **ratio** | mean_score / background_score — 1 = pas de signal, >1.5 = signal fort |

**Interprétation** : un ratio > 1.5 (ligne rouge) suggère que les scores PatchCore
augmentent AVANT l'événement. Cela valide l'hypothèse de signal précurseur visuel.

**Limites** : corrélation ≠ causalité. Les conditions météo (nuages, nuit) augmentent
aussi les scores. Le test de permutation ci-dessous quantifie la significativité.
            """,
            key="help_ew_table",
        )

        display_cols = ["event_date", "event_type", "event_intensity", "lead_days",
                        "mean_score", "background_score", "ratio", "n_images"]
        display_cols = [c for c in display_cols if c in precursors.columns]

        def _color_ratio(val):
            try:
                v = float(val)
                if v > 1.5:
                    return "background-color: #ff4d4d; color: white"
                elif v > 1.2:
                    return "background-color: #ffa500"
                return ""
            except (TypeError, ValueError):
                return ""

        st.dataframe(
            precursors[display_cols].style.map(_color_ratio, subset=["ratio"] if "ratio" in display_cols else []),
            use_container_width=True,
            key="ew_precursors_table",
        )

        # ─── Ratio par horizon — PLOTLY INTERACTIF ────────────────────────
        st.subheader("Ratio score précurseur / background par horizon")

        help_tooltip(
            "Graphique ratio précurseur",
            """
**Axe X** : date de chaque événement documenté BPPTKG.
**Axe Y** : ratio = score_précurseur / background.

- **1.0** (ligne grise) = aucun signal — score précurseur = score normal
- **1.5** (ligne rouge pointillée) = signal fort — score 50% au-dessus du fond
- **J-1, J-3, J-7** : fenêtres précurseur de 1, 3 et 7 jours avant l'événement

**Navigation** : zoomez sur un événement spécifique en cliquant-glissant.
Survolez les points pour voir le détail (événement, date, ratio exact).
            """,
            key="help_ew_ratio_chart",
        )

        if "lead_days" in precursors.columns and "event_date" in precursors.columns:
            fig_ew = go.Figure()
            lead_colors = {1: "#e74c3c", 3: "#f39c12", 7: "#2980b9"}
            for lead in sorted(precursors["lead_days"].unique()):
                grp = precursors[precursors["lead_days"] == lead].copy()
                grp["event_date"] = pd.to_datetime(grp["event_date"], errors="coerce")
                grp = grp.dropna(subset=["event_date", "ratio"])
                grp = grp.sort_values("event_date")

                _cdata_cols = [c for c in ["event_type", "event_intensity", "mean_score",
                                           "background_score", "n_images"] if c in grp.columns]
                _custom = grp[_cdata_cols].values if _cdata_cols else None

                _hover_parts = (
                    "<b>J-%{customdata[0]}</b><br>" if False else
                    "<br>".join(
                        [f"{c.replace('_', ' ').title()} : %{{customdata[{i}]}}" for i, c in enumerate(_cdata_cols)]
                    ) + "<extra></extra>"
                )

                fig_ew.add_trace(go.Scatter(
                    x=grp["event_date"],
                    y=grp["ratio"],
                    mode="lines+markers",
                    name=f"J-{lead}",
                    line=dict(color=lead_colors.get(lead, "#7f8c8d"), width=2),
                    marker=dict(size=8, symbol="circle"),
                    customdata=grp[_cdata_cols].round(3).values if _cdata_cols else None,
                    hovertemplate=(
                        "<b>%{x|%Y-%m-%d}</b><br>"
                        + "Ratio : <b>%{y:.3f}</b><br>"
                        + ("<br>".join(
                            f"{c.replace('_',' ').title()} : %{{customdata[{i}]}}"
                            for i, c in enumerate(_cdata_cols)
                        ) if _cdata_cols else "")
                        + "<extra>J-" + str(lead) + "</extra>"
                    ),
                ))

            # Lignes de référence
            fig_ew.add_hline(y=1.0, line_dash="dash", line_color="gray",
                             annotation_text="Baseline (ratio=1)", annotation_position="right")
            fig_ew.add_hline(y=1.5, line_dash="dot", line_color="#e74c3c",
                             annotation_text="Signal fort (1.5×)", annotation_position="right")
            fig_ew.add_hline(y=1.2, line_dash="dot", line_color="#f39c12",
                             annotation_text="Signal modéré (1.2×)", annotation_position="right")

            # Zone colorée au-dessus de 1.5 — y_max calculé depuis les données (sans Kaleido)
            _ew_y_max = max(
                precursors["ratio"].dropna().max() * 1.1
                if "ratio" in precursors.columns and precursors["ratio"].notna().any()
                else 3.0,
                2.0,  # minimum visuel confortable
            )
            fig_ew.add_hrect(
                y0=1.5, y1=_ew_y_max,
                fillcolor="rgba(231,76,60,0.07)",
                line_width=0,
                layer="below",
                annotation_text="⚠ Zone signal fort",
                annotation_position="top left",
                annotation_font=dict(color="#e74c3c", size=11),
            )
            # Zone modérée (1.2 → 1.5)
            fig_ew.add_hrect(
                y0=1.2, y1=1.5,
                fillcolor="rgba(243,156,18,0.06)",
                line_width=0,
                layer="below",
            )

            fig_ew.update_layout(
                title="Signal précurseur PatchCore — ratio avant événements BPPTKG",
                xaxis_title="Date de l'événement",
                yaxis_title="Ratio score précurseur / background",
                yaxis=dict(range=[0, _ew_y_max]),
                height=440,
                margin=dict(t=55, b=55, r=120),
                legend=dict(orientation="h", y=-0.20, x=0),
                hovermode="x unified",
                xaxis=dict(
                    rangeslider=dict(visible=True, thickness=0.06),
                    type="date",
                ),
            )
            st.plotly_chart(fig_ew, use_container_width=True, key="ew_ratio_plotly")
        else:
            # Toujours rendre st.plotly_chart — figure annotée si données manquantes (anti-removeChild)
            _fig_ew_empty = go.Figure()
            _fig_ew_empty.add_annotation(
                text="Données de précurseurs incomplètes<br>(colonnes lead_days ou event_date manquantes)",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=13, color="gray"),
            )
            _fig_ew_empty.update_layout(height=440, xaxis_visible=False, yaxis_visible=False,
                                        margin=dict(t=55, b=55))
            st.plotly_chart(_fig_ew_empty, use_container_width=True, key="ew_ratio_empty_chart")

    # ─── Permutation test ─────────────────────────────────────────────────
    st.divider()
    st.subheader("Test statistique de significativité")

    help_tooltip(
        "Permutation test",
        """
**Principe** : si le signal précurseur est réel, le ratio observé devrait être
significativement plus élevé que ce qu'on obtient avec des dates aléatoires.

**Méthode** :
1. Calculer le ratio observé (vrai score précurseur / background)
2. Générer 1000 dates aléatoires dans le même dataset
3. Calculer le ratio pour chaque permutation → distribution nulle
4. **p-value** = proportion des permutations avec ratio ≥ ratio observé

**Interprétation** :
- p < 0.05 : signal précurseur statistiquement significatif
- p > 0.05 : signal non-significatif (peut être dû à la météo, la nuit, etc.)

**Important** : faible nombre d'événements (< 20) → peu de puissance statistique.
        """,
        key="help_ew_permtest",
    )

    if perm_path.exists():
        perm = safe_read_csv(perm_path)
        if not perm.empty:
            perm_filtered = perm[perm.get("lead_days", pd.Series([7])) == 7] if "lead_days" in perm.columns else perm
            if not perm_filtered.empty:
                p_val = float(perm_filtered["p_value"].iloc[0])
                obs = float(perm_filtered["observed_mean"].iloc[0])
                null_m = float(perm_filtered["null_distribution_mean"].iloc[0])
                c1, c2, c3 = st.columns(3)
                c1.metric("Obs. score moyen (J-7)", f"{obs:.4f}")
                c2.metric("Dist. nulle (médiane)", f"{null_m:.4f}")
                c3.metric("p-value", f"{p_val:.4f}",
                          delta="✓ significatif" if p_val < 0.05 else "✗ non significatif",
                          delta_color="normal" if p_val < 0.05 else "off")
                # st.caption stable (anti-removeChild) — jamais alterner st.success/st.info
                st.caption(
                    f"✅ Signal précurseur **statistiquement significatif** (p={p_val:.4f} < 0.05)"
                    if p_val < 0.05 else
                    f"ℹ️ Signal non-significatif au seuil 0.05 (p={p_val:.4f}) — plus de données nécessaires."
                )

                # Histogramme distribution nulle (Plotly)
                if "null_distribution" in perm_filtered.columns:
                    _null_vals = perm_filtered["null_distribution"].iloc[0]
                    if isinstance(_null_vals, str):
                        import json
                        try:
                            _null_arr = np.array(json.loads(_null_vals))
                            fig_perm = go.Figure()
                            fig_perm.add_trace(go.Histogram(
                                x=_null_arr, nbinsx=40, name="Distribution nulle",
                                marker_color="#95a5a6", opacity=0.7,
                            ))
                            fig_perm.add_vline(x=obs, line_color="red", line_dash="dash",
                                               annotation_text=f"Observé={obs:.4f}")
                            fig_perm.update_layout(
                                title="Distribution nulle (permutation test)",
                                xaxis_title="Score moyen", yaxis_title="Fréquence",
                                height=280, margin=dict(t=40, b=30),
                            )
                            st.plotly_chart(fig_perm, use_container_width=True, key="ew_perm_hist")
                        except (json.JSONDecodeError, ValueError):
                            pass

            st.dataframe(perm, use_container_width=True, key="ew_perm_table")
    else:
        lead = st.selectbox("Horizon J-N pour le test", [1, 3, 7], index=2, key="ew_lead")
        if st.button("Lancer le permutation test (1000 it.)"):
            with st.spinner("Test en cours (~5s)..."):
                try:
                    from src.evaluation.early_warning import EarlyWarningAnalyzer
                    analyzer = EarlyWarningAnalyzer(events_path)
                    result = analyzer.permutation_test(df, score_col=score_col, lead=lead)
                    if result:
                        p_val = result["p_value"]
                        st.metric("p-value", f"{p_val:.4f}")
                        if p_val < 0.05:
                            st.success("✅ Signal précurseur statistiquement significatif !")
                        else:
                            st.info("ℹ️ Signal non-significatif — collecter plus de données.")
                        st.json(result)
                except Exception as exc:
                    st.error(f"Erreur : {exc}")



# ==============================================================
# Page 11 — Analyse avancée
# ==============================================================

def page_analyse_avancee(df: pd.DataFrame, config: dict) -> None:
    """Page Analyse avancée : reconstruction diffusion + comparaison original / reconstruit."""
    st.title("🧪 Analyse avancée — Reconstruction diffusion")
    st.markdown(
        "Visualisez la **carte d'anomalie** générée par reconstruction img2img. "
        "L'image est encodée dans l'espace latent de SD 1.5, débruitée vers un état "
        "« volcan normal », puis comparée à l'original pour isoler les zones suspectes."
    )

        # Filtres sidebar — chargés AVANT les tabs pour éviter le removeChild React
    _adv_sel_year, _adv_sel_month, _adv_df_filt = sidebar_data_filters(df)

    tab_reconstruct, tab_compare = st.tabs([
        "🔄 Reconstruction", "📊 Comparaison anomalies",
    ])

    # ────────────────────────────────────────────────────────────────────────
    # Tab 1 — Reconstruction img2img
    # ────────────────────────────────────────────────────────────────────────
    with tab_reconstruct:
        st.subheader("Reconstruction img2img — détection par différence")

        # ── Sélection de l'image source ───────────────────────────────────
        col_src, col_opts = st.columns([2, 1])
        with col_src:
            source_mode = st.radio(
                "Source de l'image",
                ["📋 Top anomalies (PatchCore)", "📂 Sélection manuelle", "📤 Upload"],
                horizontal=True,
                key="adv_source_mode",
            )

        img_path_sel: Path | None = None

        if source_mode == "📋 Top anomalies (PatchCore)":
            score_c = "patchcore_score" if "patchcore_score" in df.columns else "anomaly_score"
            _df_scored = df[df[score_c].notna()]
            if _df_scored.empty:
                st.info("Aucun score disponible — lancez d'abord `python run_v1_pipeline.py --step patchcore`.")
            else:
                # ── Filtres utilisateur (sans sélection aléatoire) ────────
                _top_col_y, _top_col_m, _top_col_n = st.columns(3)
                with _top_col_y:
                    _top_years = sorted(_df_scored["year"].dropna().unique().astype(int), reverse=True)
                    _sel_year_t = st.selectbox("Année", ["Toutes"] + [str(y) for y in _top_years], key="adv_top_year")
                _df_y = _df_scored if _sel_year_t == "Toutes" else _df_scored[_df_scored["year"] == int(_sel_year_t)]
                with _top_col_m:
                    _top_months = sorted(_df_y["month"].dropna().unique().astype(int))
                    _sel_month_t = st.selectbox("Mois", ["Tous"] + [f"{m:02d}" for m in _top_months], key="adv_top_month")
                _df_ym = _df_y if _sel_month_t == "Tous" else _df_y[_df_y["month"] == int(_sel_month_t)]
                with _top_col_n:
                    _max_n = max(1, len(_df_ym))
                    _top_n_images = st.slider("Nb images", 1, min(100, _max_n), min(10, _max_n), key="adv_top_n")
                top_anom = _df_ym.nlargest(_top_n_images, score_c)
                if top_anom.empty:
                    st.info("Aucune image pour cette sélection.")
                else:
                    filenames = top_anom["filename"].tolist()
                    sel_fn = st.selectbox("Image (classée par score PatchCore)", filenames, key="adv_top_sel")
                    sel_row = top_anom[top_anom["filename"] == sel_fn].iloc[0]
                    img_path_sel = find_image_path(sel_row)
                    st.caption(
                        f"Score PatchCore : **{sel_row[score_c]:.4f}** | "
                        f"Date : {int(sel_row.get('year','?'))}-{sel_row.get('month','?'):02g}-{sel_row.get('day','?'):02g}"
                    )

        elif source_mode == "📂 Sélection manuelle":
            if _adv_df_filt.empty:
                st.info("Aucune image pour cette sélection (vérifiez les filtres année/mois dans la sidebar).")
            else:
                fn = st.selectbox("Fichier", _adv_df_filt["filename"].tolist(), key="adv_manual_sel")
                row = _adv_df_filt[_adv_df_filt["filename"] == fn].iloc[0]
                img_path_sel = find_image_path(row)

        else:  # Upload
            uploaded = st.file_uploader("Charger une image (JPG, PNG)", type=["jpg", "jpeg", "png"])
            if uploaded:
                from PIL import Image as _PILImg
                _upload_tmp = PROJECT_ROOT / "outputs" / "generated" / f"_upload_{uploaded.name}"
                _upload_tmp.parent.mkdir(parents=True, exist_ok=True)
                _upload_tmp.write_bytes(uploaded.read())
                img_path_sel = _upload_tmp

        # ── Paramètres de reconstruction ──────────────────────────────────
        with col_opts:
            strength = st.slider(
                "Strength img2img", 0.10, 0.70, 0.35, 0.05,
                help="0.1 = conserve l'original, 0.7 = très modifié",
                key="adv_strength",
            )
            use_lora_rec = False
            _lora_candidates = [
                PROJECT_ROOT / "outputs" / "lora_merapi_physics" / "lora_merapi_physics_final",
                PROJECT_ROOT / "outputs" / "lora_merapi_results" / "lora_merapi_final",
            ]
            _lora_found = next((p for p in _lora_candidates if p.exists()), None)
            use_lora_rec = st.checkbox(
                "Utiliser LoRA volcanique",
                value=bool(_lora_found),
                key="adv_use_lora",
                disabled=not bool(_lora_found),
            )
            st.caption(f"`{_lora_found.name}`" if _lora_found else "_LoRA non détecté_")

            # ── Mode de qualité ─────────────────────────────────────────────
            quality_mode = st.radio(
                "Mode",
                ["⚡ Rapide (384 px, 15 steps)", "🧠 Précis (512 px, 25 steps)"],
                index=0,
                key="adv_quality_mode",
                help="Rapide : ~5–15s sur GPU, ~60s sur MPS. Précis : ~2× plus long.",
                horizontal=True,
            )
            _qmode_val = "fast" if "Rapide" in quality_mode else "precise"

        # ── Options avant le bouton (évite le DOM removeChild) ────────────
        save_figs = st.checkbox(
            "💾 Sauvegarder les figures après reconstruction",
            value=False,
            key="adv_save_figs",
        )

        # ── Avertissement MPS (premier chargement long) ───────────────────
        try:
            import torch as _torch
            if hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
                st.info(
                    "**Apple Silicon (MPS)** : la 1ʳᵉ reconstruction déclenche la "
                    "compilation JIT (~60–150s). Les suivantes seront bien plus rapides. "
                    "Le modèle reste en mémoire entre les clics."
                )
        except ImportError:
            pass

        # ── Bouton toujours rendu (disabled si pas d'image) — évite removeChild ──
        _rec_disabled = img_path_sel is None or (img_path_sel is not None and not img_path_sel.exists())
        _run_rec_btn = st.button(
            "🔄 Lancer la reconstruction", type="primary",
            use_container_width=True, key="adv_run_rec",
            disabled=_rec_disabled,
        )
        if img_path_sel is None:
            st.caption("Sélectionnez ou chargez une image pour continuer.")
        elif not img_path_sel.exists():
            st.caption(f"Fichier introuvable : `{img_path_sel.name}`")

        # ── Invalidation cache si paramètres changent ─────────────────────
        _lora_key_str = str(_lora_found) if (use_lora_rec and _lora_found) else None
        _rec_run_key = f"{img_path_sel}|{strength}|{_lora_key_str}|{_qmode_val}"
        if st.session_state.get("_rec_run_key") != _rec_run_key:
            # Nouveaux paramètres → effacer l'ancien résultat
            st.session_state.pop("_rec_result", None)
            st.session_state.pop("_rec_img_path", None)
            st.session_state["_rec_run_key"] = _rec_run_key

        if _run_rec_btn and not _rec_disabled:
            # ── Pipeline chargé via @st.cache_resource ────────────────────
            # Le modèle est chargé UNIQUEMENT si lora_key_str change.
            # Changer strength ou quality_mode NE recharge PAS SD 1.5.
            _pipe_load_msg = st.empty()
            _pipe_load_msg.info("⏳ Pipeline SD 1.5 en cours de chargement (1ʳᵉ fois seulement)…")
            try:
                _pipe, _backend = _load_diffusion_pipeline(_lora_key_str)
                _pipe_load_msg.empty()
            except Exception as exc:
                _pipe_load_msg.empty()
                st.error(f"Erreur chargement pipeline : {exc}")
                _pipe, _backend = None, "fallback"

            from src.models.diffusion_reconstructor import DiffusionReconstructor
            _rec = DiffusionReconstructor(
                pipeline=_pipe,
                backend=_backend,
                strength=strength,
                quality_mode=_qmode_val,
            )
            _steps_info = {"fast": "15 steps, 384 px", "precise": "25 steps, 512 px"}
            with st.spinner(
                f"Reconstruction en cours ({_steps_info[_qmode_val]}, "
                f"backend={_backend})…"
            ):
                try:
                    result = _rec.reconstruct(img_path_sel, strength=strength, quality_mode=_qmode_val)
                    st.session_state["_rec_result"] = result
                    st.session_state["_rec_img_path"] = str(img_path_sel)
                except Exception as exc:
                    st.error(f"Erreur reconstruction : {exc}")
                    st.session_state.pop("_rec_result", None)

        # ── Affichage persistant des résultats (survit aux reruns) ────────
        _rec_results_slot = st.container()  # slot stable — évite removeChild React
        if "_rec_result" in st.session_state:
            result = st.session_state["_rec_result"]
            _src_label = Path(st.session_state.get("_rec_img_path", "?")).name
            _rec_results_slot.caption(f"Résultat pour : `{_src_label}`")

            orig  = result["original"]
            recon = result["reconstructed"]
            diff  = result["diff_map"]
            score = result["anomaly_score"]
            backend = result["backend"]
            _res_qmode = result.get("quality_mode", _qmode_val)

            # ── Protection NaN / inf sur le score et la diff map ───────────────────
            try:
                _score_safe = float(score)
                if np.isnan(_score_safe) or np.isinf(_score_safe):
                    _score_safe = 0.0
            except (TypeError, ValueError):
                _score_safe = 0.0
            _diff_vals = diff[~np.isnan(diff)] if diff is not None and diff.size > 0 else np.array([0.0])

            with _rec_results_slot:
                col_o, col_r, col_d = st.columns(3)
                col_o.image(orig, caption="Image originale", **{_IMG_WIDTH_KW: True})
                col_r.image(recon, caption=f"Reconstruction ({backend}, {_res_qmode})", **{_IMG_WIDTH_KW: True})

                try:
                    from src.models.diffusion_reconstructor import DiffusionReconstructor as _DR
                    colored_diff = _DR.colorize_diff(diff)
                except Exception:
                    # Fallback colorisation manuelle si import échoue
                    import matplotlib.cm as _cm
                    colored_diff = (_cm.hot(diff)[:, :, :3] * 255).astype(np.uint8)
                col_d.image(colored_diff, caption=f"Carte d'anomalie (score={_score_safe:.2f})", **{_IMG_WIDTH_KW: True})

                m1, m2, m3 = st.columns(3)
                m1.metric("Score reconstruction", f"{_score_safe:.3f}")
                m2.metric("Diff. max (%)", f"{_diff_vals.max()*100:.1f}")
                m3.metric("Diff. P95 (%)", f"{np.percentile(_diff_vals, 95)*100:.1f}")

            # st.caption stable — jamais alterner st.info / st.success (anti-removeChild)
            st.caption(
                "⚠️ Mode fallback (diffusers non disponible) — reconstruction via flou gaussien, résultats indicatifs."
                if backend == "fallback" else
                f"✅ Reconstruction SD 1.5 via **{backend}**."
            )

            # ── Sauvegarde (checkbox définie avant le bouton, stable) ─────
            if save_figs:
                try:
                    from PIL import Image as _PILImg
                    import time as _time
                    out_dir = PROJECT_ROOT / "outputs" / "generated"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    ts = int(_time.time())
                    _PILImg.fromarray(recon).save(str(out_dir / f"recon_{ts}.png"))
                    _PILImg.fromarray(colored_diff).save(str(out_dir / f"diff_{ts}.png"))
                    st.caption(f"✅ Sauvegardées : `outputs/generated/recon_{ts}.png`")
                except Exception as exc:
                    st.warning(f"Sauvegarde impossible : {exc}")

            if st.button("🗑️ Effacer les résultats", key="adv_clear_rec"):
                st.session_state.pop("_rec_result", None)
                st.session_state.pop("_rec_img_path", None)
                st.rerun()

    # ────────────────────────────────────────────────────────────────────────
    # Tab 2 — Comparaison multi-images
    # ────────────────────────────────────────────────────────────────────────
    with tab_compare:
        st.subheader("Comparaison scores PatchCore × Early Warning")

        score_c2 = "patchcore_score" if "patchcore_score" in df.columns else "anomaly_score"
        df_s2 = df[df[score_c2].notna()].copy()

        # Toujours rendre st.pyplot — figure annotée si pas de scores (anti-removeChild)
        fig, ax = plt.subplots(figsize=(10, 3))
        if df_s2.empty:
            ax.text(0.5, 0.5, "Aucun score d'anomalie disponible",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_axis_off()
        else:
            # Distribution PatchCore
            p90 = float(df_s2[score_c2].quantile(0.90))
            p95 = float(df_s2[score_c2].quantile(0.95))
            ax.hist(df_s2[score_c2], bins=50, edgecolor="white", alpha=0.8, color="#9b59b6")
            ax.axvline(p90, color="#e67e22", ls="--", lw=1.5, label=f"P90 ({p90:.3f})")
            ax.axvline(p95, color="#e74c3c", ls="--", lw=2, label=f"P95 ({p95:.3f})")
            ax.set_xlabel("Score PatchCore")
            ax.set_ylabel("Fréquence")
            ax.set_title("Distribution des scores d'anomalie — seuils percentiles")
            ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        if not df_s2.empty:
            # Scatter year × score
            if "year" in df_s2.columns and df_s2["year"].notna().any():
                st.markdown("**Score par année**")
                year_score = df_s2.groupby("year")[score_c2].agg(["mean", "max", "std"]).reset_index()
                year_score["year"] = year_score["year"].astype(int)
                fig2, ax2 = plt.subplots(figsize=(8, 3))
                ax2.errorbar(
                    year_score["year"], year_score["mean"],
                    yerr=year_score["std"], fmt="o-", capsize=4, lw=2, color="#3498db", label="Moyenne ± σ",
                )
                ax2.bar(year_score["year"], year_score["max"], alpha=0.15, color="#e74c3c", label="Max annuel")
                ax2.set_xlabel("Année")
                ax2.set_ylabel("Score PatchCore")
                ax2.set_title("Score d'anomalie par année")
                ax2.legend(fontsize=9)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)

    # ────────────────────────────────────────────────────────────────────────
    # Tab 3 — Simulation (recap)


# ==============================================================
# Page 12 — Analyse volcanique avancée
# ==============================================================

def page_analyse_volcanique(df: pd.DataFrame, config: dict) -> None:
    """Page Analyse volcanique avancée : classification, heatmap, timeline, comparaison."""
    st.title("🌋 Analyse volcanique avancée")
    st.markdown(
        "Classification des événements volcaniques, carte d'activité spatiale "
        "et timeline d'anomalies — basées sur les features physiques v2 et DINOv2+PatchCore."
    )

    if df.empty:
        st.warning("Aucune donnée disponible. Exécutez le pipeline complet d'abord.")
        return

    # ── Chargement des modules analytiques ────────────────────────────────
    try:
        from src.models.volcano_classifier import VolcanoClassifier, VOLCANO_CLASSES, classify_heuristic
        _volcano_clf_available = True
    except ImportError as e:
        st.error(f"Module volcano_classifier non disponible : {e}")
        _volcano_clf_available = False
        VOLCANO_CLASSES = ["pyroclastique", "lave", "nuage", "normal"]

    try:
        from src.analysis.activity_heatmap import (
            compute_activity_heatmap,
            detect_active_clusters,
            timeline_activity,
        )
        _heatmap_available = True
    except ImportError as e:
        st.error(f"Module activity_heatmap non disponible : {e}")
        _heatmap_available = False

    # ── Chemins des fichiers de features et scores ─────────────────────────
    features_path = PROJECT_ROOT / "outputs" / "models" / "physical_features.csv"
    scores_path = PROJECT_ROOT / "outputs" / "scores" / "patchcore_scores.csv"
    clf_path = PROJECT_ROOT / "outputs" / "models" / "volcano_clf.pkl"

    # Charger les features si disponibles
    df_feat = None
    if features_path.exists():
        df_feat = safe_read_csv(features_path)
        if df_feat.empty:
            df_feat = None
            st.sidebar.warning("Features physiques illisibles.\nRelancez : `python run_v1_pipeline.py --step features`")
        else:
            st.sidebar.success(f"Features : {len(df_feat):,} images")
    else:
        st.sidebar.warning("Features physiques non calculées.\nLancez : `python run_v1_pipeline.py --step features`")

    # Charger les scores PatchCore si disponibles
    if scores_path.exists() and "patchcore_score" not in df.columns:
        sc = safe_read_csv(scores_path)
        if not sc.empty and "filename" in sc.columns and "patchcore_score" in sc.columns:
            sc_map = sc.drop_duplicates("filename").set_index("filename")["patchcore_score"]
            df = df.copy()
            df["patchcore_score"] = df["filename"].map(sc_map)

    # ── Tabs principaux ────────────────────────────────────────────────────
    # Filtres sidebar — chargés AVANT les tabs pour éviter le removeChild React
    sel_year_c, sel_month_c, df_filt_c = sidebar_data_filters(df)

    tab_clf, tab_heatmap, tab_timeline, tab_compare = st.tabs([
        "🏷️ Classification", "🗺️ Zones actives", "📅 Timeline d'activité", "🔎 Comparaison"
    ])

    # ==== Tab 1 : Classification ==========================================
    with tab_clf:
        st.subheader("Classification des événements volcaniques")
        st.markdown(
            "Chaque image est classée en **pyroclastique**, **lave**, **nuage** ou **normal** "
            "selon ses features physiques. Si un modèle entraîné est disponible, il est utilisé ; "
            "sinon, un système de règles heuristiques est appliqué."
        )

        help_tooltip(
            "Limites de la classification",
            """
**Méthode** :
- Si `volcano_clf.pkl` est présent : Random Forest entraîné sur features physiques
- Sinon : système de règles heuristiques basé sur les colonnes disponibles

**Seuils heuristiques actuels** (calibrés sur quantiles empiriques) :
- *Pyroclastique* : `patchcore_score > Q90` **ET** variation temporelle élevée
- *Lave* : `is_night=True` **ET** `bright_pixel_ratio > 2%`
- *Nuage* : `cloud_coverage > 70%`
- *Normal* : cas par défaut

**Limitations scientifiques** :
1. Les features physiques (`physical_features.csv`) peuvent être absentes → la classification se
   base uniquement sur `patchcore_score`, dégradant fortement la précision
2. `is_night` est estimé depuis l'heure locale, sans correction solaire → erreur ±1h aux solstices
3. `bright_pixel_ratio` est sensible aux nuages → faux positifs lave possibles
4. Le modèle RandomForest n'a pas de ground-truth validé → métriques de précision non disponibles

**Recommandation** : interpréter comme une **aide à la décision exploratoire**, pas comme une
classification définitive. Toujours croiser avec les bulletins BPPTKG.
            """,
            key="help_vcf_classification",
        )

        if not _volcano_clf_available:
            st.error("Module de classification non disponible.")
        else:
            # ── Bouton déclencheur (évite 76k iterrows à chaque rerun) ────
            col_clf_btn, col_clf_reset, col_clf_info = st.columns([1, 1, 2])
            with col_clf_btn:
                run_clf = st.button(
                    "▶️ Lancer la classification",
                    type="primary", key="vcf_run_btn",
                    help="Lance la classification sur toutes les images (~10–30s)",
                )
            with col_clf_reset:
                if st.button("🗑️ Réinitialiser", key="vcf_reset_btn",
                             help="Efface les résultats en cache (utile après mise à jour de l'heuristique)"):
                    st.session_state.pop("vcf_results", None)
                    st.session_state.pop("_vcf_importances", None)
                    st.rerun()
            with col_clf_info:
                # Toujours st.info (widget stable) — évite removeChild React
                if "vcf_results" in st.session_state:
                    _cached_count = len(st.session_state["vcf_results"])
                    st.info(f"Résultats en cache ({_cached_count:,} images). Cliquez Relancer pour recalculer.")
                else:
                    st.info(f"Cliquez pour classifier {len(df):,} images.")

            if run_clf:
                # Recalcule et écrase le cache
                st.session_state.pop("vcf_results", None)

                # Préparer le DataFrame de features
                df_for_clf = df.copy()
                if df_feat is not None:
                    merge_cols = ["filename"] + [c for c in df_feat.columns if c != "filename"]
                    df_for_clf = df_for_clf.merge(
                        df_feat[merge_cols], on="filename", how="left",
                    )

                # Charger le classifieur
                clf = VolcanoClassifier()
                if clf_path.exists():
                    try:
                        clf = _cached_load_volcano_classifier(str(clf_path))
                        st.success(f"✅ Modèle chargé : `{clf_path.name}`")
                    except Exception as e:
                        st.warning(f"Erreur chargement modèle : {e} — mode heuristique.")

                from src.features.physical_features import FEATURE_NAMES
                feat_present = [c for c in FEATURE_NAMES if c in df_for_clf.columns]
                if not feat_present:
                    st.info(
                        "Aucune feature physique dans l'index. "
                        "La classification heuristique utilise les colonnes disponibles."
                    )

                with st.spinner(f"Classification de {len(df_for_clf):,} images en cours…"):
                    preds = clf.predict(df_for_clf)
                    probas = clf.predict_proba(df_for_clf)

                _df_clf = df_for_clf[["filename"]].copy()
                _df_clf["volcano_class"] = preds
                _df_clf["class_confidence"] = probas.max(axis=1).values
                st.session_state["vcf_results"] = _df_clf
                # Importances en session_state → affichage stable entre reruns
                try:
                    _imp = clf.feature_importances()
                    st.session_state["_vcf_importances"] = _imp
                except Exception:
                    st.session_state["_vcf_importances"] = None

            # ── Affichage des résultats — structure TOUJOURS identique (anti-removeChild) ──
            # Règle absolue : NE JAMAIS modifier la structure UI selon session_state.
            # Calculer les valeurs en amont, toujours rendre les mêmes widgets.
            _vcf_has_results = "vcf_results" in st.session_state
            if _vcf_has_results:
                df_classified = st.session_state["vcf_results"]
                preds = df_classified["volcano_class"].values
                counts = pd.Series(preds).value_counts()
            else:
                df_classified = pd.DataFrame(columns=["filename", "volcano_class", "class_confidence"])
                counts = pd.Series(dtype=int)

            colors_cls = {
                "pyroclastique": "#e74c3c",
                "lave":          "#f39c12",
                "nuage":         "#95a5a6",
                "normal":        "#2ecc71",
            }

            # 4 métriques — toujours rendues (0 quand pas de données)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Pyroclastiques", int(counts.get("pyroclastique", 0)))
            col2.metric("Coulées lave", int(counts.get("lave", 0)))
            col3.metric("Nuages", int(counts.get("nuage", 0)))
            col4.metric("Normaux", int(counts.get("normal", 0)))

            st.divider()  # toujours rendu — jamais conditionnel

            # st.caption toujours rendu (même widget, texte différent) — jamais conditionnel.
            # Un st.info conditionnel entre st.divider et st.columns change la structure
            # DOM entre deux runs → removeChild sur le nœud info disparu.
            st.caption(
                "Classification en cache — modifiez les filtres ou cliquez Réinitialiser."
                if _vcf_has_results
                else "▶️ Cliquez sur Lancer la classification pour démarrer l'analyse."
            )

            # Graphiques — toujours rendus (figure vide si pas de données)
            col_pie, col_bar = st.columns(2)
            with col_pie:
                if _vcf_has_results and counts.sum() > 0:
                    fig_pie = go.Figure(go.Pie(
                        labels=counts.index.tolist(),
                        values=counts.values.tolist(),
                        marker=dict(colors=[colors_cls.get(c, "#3498db") for c in counts.index]),
                        textinfo="label+percent",
                        hole=0.3,
                        hovertemplate="<b>%{label}</b><br>Count : %{value}<br>Part : %{percent}<extra></extra>",
                    ))
                    fig_pie.update_layout(
                        title="Répartition des classes",
                        height=360, margin=dict(t=40, b=10, l=10, r=10),
                        legend=dict(orientation="v", x=1.05),
                    )
                else:
                    fig_pie = go.Figure()
                    fig_pie.update_layout(
                        title="Répartition des classes", height=360,
                        annotations=[dict(text="Lancez la classification", showarrow=False,
                                         font=dict(size=14), xref="paper", yref="paper")],
                    )
                st.plotly_chart(fig_pie, use_container_width=True, key="vcf_pie")

            with col_bar:
                fig_bar_cls = go.Figure()
                if _vcf_has_results:
                    df_merged_cls = df_classified.merge(
                        df[["filename", "year"]].dropna(subset=["year"]),
                        on="filename", how="left",
                    )
                    if df_merged_cls["year"].notna().any():
                        cross = pd.crosstab(
                            df_merged_cls["year"].dropna().astype(int),
                            df_merged_cls["volcano_class"],
                        )
                        for cls_name in cross.columns:
                            fig_bar_cls.add_trace(go.Bar(
                                x=cross.index.tolist(),
                                y=cross[cls_name].values.tolist(),
                                name=cls_name,
                                marker_color=colors_cls.get(cls_name, "#3498db"),
                                hovertemplate="Année %{x}<br>" + cls_name + " : %{y}<extra></extra>",
                            ))
                        fig_bar_cls.update_layout(
                            barmode="stack",
                            title="Classification par année",
                            xaxis_title="Année",
                            yaxis_title="Nombre d'images",
                            height=360,
                            margin=dict(t=40, b=30),
                            legend=dict(orientation="h", y=-0.22),
                        )
                    else:
                        fig_bar_cls.update_layout(
                            title="Classification par année", height=360,
                            annotations=[dict(text="Colonne 'year' absente", showarrow=False,
                                             font=dict(size=14), xref="paper", yref="paper")],
                        )
                else:
                    fig_bar_cls.update_layout(
                        title="Classification par année", height=360,
                        annotations=[dict(text="Lancez la classification", showarrow=False,
                                         font=dict(size=14), xref="paper", yref="paper")],
                    )
                st.plotly_chart(fig_bar_cls, use_container_width=True, key="vcf_bar_year")

            # Tableau pyroclastiques — st.dataframe TOUJOURS rendu (DataFrame vide si pas de données)
            st.subheader("🔝 Images pyroclastiques détectées")
            if _vcf_has_results:
                pyro = df_classified[df_classified["volcano_class"] == "pyroclastique"].copy()
                if not pyro.empty:
                    _pc_col = "patchcore_score" if "patchcore_score" in df.columns else None
                    _merge_cols = ["filename", "year", "month", "day", "hour"]
                    if _pc_col:
                        _merge_cols.append(_pc_col)
                    pyro_full = pyro.merge(
                        df[[c for c in _merge_cols if c in df.columns]],
                        on="filename", how="left",
                    )
                    show_cols = [c for c in ["filename", "year", "month", "day", "hour",
                                              "class_confidence", "patchcore_score"]
                                  if c in pyro_full.columns]
                    pyro_display = pyro_full[show_cols].sort_values(
                        "class_confidence", ascending=False
                    ).round(4)
                else:
                    pyro_display = pd.DataFrame(columns=["filename", "volcano_class", "class_confidence"])
            else:
                pyro_display = pd.DataFrame(columns=["filename", "volcano_class", "class_confidence"])
            st.dataframe(pyro_display, use_container_width=True, hide_index=True, height=300,
                         key="vcf_pyro_table")

            # Expander importances — TOUJOURS rendu, figure vide si pas de modèle RF
            with st.expander("📊 Importances des features (RandomForest)", expanded=False):
                _importances_ss = st.session_state.get("_vcf_importances")
                if _importances_ss is not None:
                    _imp_sorted = _importances_ss.sort_values(ascending=True)
                    fig_imp = go.Figure(go.Bar(
                        x=_imp_sorted.values.tolist(),
                        y=_imp_sorted.index.tolist(),
                        orientation="h",
                        marker_color="#3498db",
                        hovertemplate="%{y} : %{x:.4f}<extra></extra>",
                    ))
                    fig_imp.update_layout(
                        title="Importance des features (RandomForest)",
                        xaxis_title="Importance",
                        height=max(250, 30 * len(_imp_sorted)),
                        margin=dict(t=40, b=30, l=160),
                    )
                else:
                    fig_imp = go.Figure()
                    fig_imp.update_layout(
                        title="Importances des features", height=250,
                        annotations=[dict(text="Disponible après classification avec modèle RF",
                                          showarrow=False, font=dict(size=13),
                                          xref="paper", yref="paper")],
                    )
                st.plotly_chart(fig_imp, use_container_width=True, key="vcf_feat_importance")

    # ==== Tab 2 : Zones actives (heatmap spatiale) ========================
    with tab_heatmap:
        st.subheader("Carte d'activité spatiale (16×16 grille DINOv2)")
        st.markdown(
            "La heatmap représente l'anomalie spatiale agrégée sur l'ensemble des images. "
            "Les zones chaudes correspondent aux régions du volcan les plus souvent anormales."
        )

        if not _heatmap_available:
            st.error("Module activity_heatmap non disponible.")
        else:
            col_params, col_heat = st.columns([1, 2])
            with col_params:
                heat_agg = st.radio("Agrégation", ["mean", "max"], index=0,
                                    help="mean = activité moyenne, max = pic d'activité")
                heat_thresh = st.slider("Seuil cluster", 0.3, 0.9, 0.5, 0.05,
                                        help="Seuil de détection des zones actives")
                heat_min_size = st.number_input("Taille min cluster (cellules)", 1, 20, 2)

            df_for_heat = df.copy()
            if df_feat is not None:
                df_for_heat = df_for_heat.merge(
                    df_feat[["filename"] + [c for c in df_feat.columns if c != "filename"]],
                    on="filename", how="left",
                )

            # Calcul hors des colonnes : st.spinner() inside a column injects
            # a transient React node that triggers removeChild when it disappears.
            _heat_error = None
            try:
                with st.spinner("Calcul de la heatmap…"):
                    heatmap = compute_activity_heatmap(
                        df_for_heat,
                        scores_path=scores_path if scores_path.exists() else None,
                        aggregation=heat_agg,
                    )
            except Exception as _heat_exc:
                _heat_error = str(_heat_exc)
                heatmap = np.zeros((16, 16))

            if _heat_error:
                st.error(f"Erreur calcul heatmap : {_heat_error}")

            with col_heat:
                # ── Heatmap spatiale Plotly ────────────────────────────────────
                _heat_text = np.round(heatmap, 3).astype(str)
                fig_heat = go.Figure(go.Heatmap(
                    z=heatmap,
                    colorscale="YlOrRd",
                    zmin=0, zmax=1,
                    text=_heat_text,
                    texttemplate="%{text}",
                    textfont=dict(size=9),
                    colorbar=dict(title="Score d'activité<br>normalisé", thickness=15),
                    hovertemplate="Ligne %{y} / Col %{x}<br>Score : %{z:.4f}<extra></extra>",
                ))
                fig_heat.update_layout(
                    title="Carte d'activité volcanique (grille 16×16 patches DINOv2)",
                    xaxis_title="Colonne (gauche → droite)",
                    yaxis_title="Ligne (haut → bas)",
                    height=500,
                    margin=dict(t=50, b=40, l=60, r=60),
                    xaxis=dict(side="bottom"),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_heat, use_container_width=True, key="vcf_heatmap_spatial")

            # Clusters actifs
            clusters = detect_active_clusters(
                heatmap,
                threshold=heat_thresh,
                min_size=int(heat_min_size),
            )
            st.subheader(f"Clusters actifs détectés : {len(clusters)}")
            # st.dataframe TOUJOURS rendu — DataFrame vide si aucun cluster.
            # Alterner dataframe ↔ info dans un container change le type de nœud
            # React à chaque mouvement du slider → removeChild.
            if clusters:
                cl_df = pd.DataFrame(clusters)
                cl_df["row"] = cl_df["row"].round(1)
                cl_df["col"] = cl_df["col"].round(1)
                cl_df["score"] = cl_df["score"].round(4)
                cl_display = cl_df
            else:
                cl_display = pd.DataFrame(columns=["row", "col", "score", "size"])
            st.caption(
                f"{len(clusters)} cluster(s) trouvé(s)."
                if clusters
                else f"Aucun cluster avec seuil={heat_thresh:.2f}, taille min={heat_min_size}."
            )
            st.dataframe(cl_display, use_container_width=True, hide_index=True,
                         key="vcf_cluster_table")

    # ==== Tab 3 : Timeline d'activité =====================================
    with tab_timeline:
        st.subheader("Timeline du score d'activité volcanique")
        st.markdown(
            "Score d'anomalie agrégé par période — permet d'identifier "
            "les mois ou semaines à haute activité anormale."
        )

        if not _heatmap_available:
            st.error("Module activity_heatmap non disponible.")
        else:
            # col_info supprimé — colonne fantôme jamais remplie → DOM instable
            resample_opt = st.selectbox(
                "Granularité",
                [("Jour", "D"), ("Semaine", "W"), ("Mois", "ME")],
                format_func=lambda x: x[0],
                index=2,
                key="vcf_timeline_resample",
            )
            resample_freq = resample_opt[1]

            try:
                timeline = timeline_activity(
                    df,
                    scores_path=scores_path if scores_path.exists() else None,
                    resample=resample_freq,
                )
            except Exception as _tl_exc:
                st.error(f"Erreur calcul timeline : {_tl_exc}")
                timeline = pd.DataFrame()

            # st.container() est plus stable que st.empty() : ne crée pas
            # de nœud React spécial, gère le changement de type de contenu
            # (info ↔ plotly_chart) sans removeChild.

            # Valeurs par défaut (rendues même si timeline vide — widgets stables)
            _tl_n_periods = 0
            _tl_max_score = "—"
            _tl_n_peaks = 0
            _tl_csv = b""

            # Pattern obligatoire : toujours st.plotly_chart, figure vide si pas de données.
            # Un container avec info↔plotly alternés est instable (React removeChild).
            if timeline.empty:
                fig_tl = go.Figure()
                fig_tl.update_layout(
                    title="Timeline d'activité volcanique — Merapi Kalor",
                    height=380,
                    annotations=[dict(text="Aucune donnée temporelle disponible",
                                      showarrow=False, font=dict(size=14),
                                      xref="paper", yref="paper")],
                )
            else:
                # ── Timeline Plotly ────────────────────────────────────────────
                mu = timeline["mean_score"].mean()
                sigma = timeline["mean_score"].std()
                threshold_t = mu + 2 * sigma
                peaks = timeline[timeline["mean_score"] > threshold_t]

                fig_tl = go.Figure()
                # Aire remplie
                fig_tl.add_trace(go.Scatter(
                    x=timeline["datetime"], y=timeline["mean_score"],
                    fill="tozeroy", fillcolor="rgba(52,152,219,0.12)",
                    line=dict(color="#3498db", width=1.5),
                    name="Score moyen",
                    mode="lines+markers",
                    marker=dict(size=4),
                    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Score : %{y:.5f}<extra></extra>",
                ))
                # Pics d'anomalie
                if not peaks.empty:
                    fig_tl.add_trace(go.Scatter(
                        x=peaks["datetime"], y=peaks["mean_score"],
                        mode="markers", name=f"{len(peaks)} pics (>2σ)",
                        marker=dict(color="#e74c3c", size=9, symbol="circle-open",
                                    line=dict(width=2, color="#e74c3c")),
                        hovertemplate="<b>PIC : %{x|%Y-%m-%d}</b><br>Score : %{y:.5f}<extra>Anomalie</extra>",
                    ))
                # Seuils
                fig_tl.add_hline(y=threshold_t, line_dash="dash", line_color="red", line_width=1.2,
                                  annotation_text=f"μ+2σ = {threshold_t:.4f}",
                                  annotation_position="right")
                fig_tl.add_hline(y=mu, line_dash="dot", line_color="green", line_width=1.0,
                                  annotation_text=f"μ = {mu:.4f}",
                                  annotation_position="right")
                fig_tl.update_layout(
                    title="Timeline d'activité volcanique — Merapi Kalor",
                    xaxis_title="Date",
                    yaxis_title="Score d'anomalie moyen",
                    height=380,
                    margin=dict(t=50, b=60, r=100),
                    legend=dict(orientation="h", y=-0.22),
                    hovermode="x unified",
                    xaxis=dict(
                        rangeslider=dict(visible=True, thickness=0.05),
                        type="date",
                    ),
                )
                _tl_n_periods = len(timeline)
                _tl_max_score = f"{timeline['mean_score'].max():.4f}"
                _tl_n_peaks = len(peaks) if not peaks.empty else 0
                _tl_csv = timeline.to_csv(index=False).encode("utf-8")

            st.plotly_chart(fig_tl, use_container_width=True, key="vcf_timeline_activity")

            # Métriques et bouton export toujours rendus (DOM stable)
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Périodes analysées", _tl_n_periods, label_visibility="visible")
            col_m2.metric("Score max", _tl_max_score, label_visibility="visible")
            col_m3.metric("Pics >2σ", _tl_n_peaks, label_visibility="visible")
            # _tl_csv = b"" par défaut → disabled=True si timeline vide (pas de NameError)
            st.download_button(
                "📥 Télécharger la timeline (CSV)", _tl_csv,
                file_name="merapi_activity_timeline.csv", mime="text/csv",
                disabled=(_tl_csv == b""),
                key="vcf_timeline_download",
            )

    # ==== Tab 4 : Comparaison =============================================
    with tab_compare:
        st.subheader("Comparaison heuristique vs modèle entraîné")
        # sel_year_c, sel_month_c, df_filt_c sont déjà chargés avant les tabs

        col_comp_btn, col_comp_info = st.columns([1, 3])
        with col_comp_btn:
            _run_compare = st.button(
                "▶️ Comparer", type="primary", key="vcf_compare_btn",
                help="Compare l'heuristique au modèle RF sur la période sélectionnée",
                disabled=(df_filt_c.empty or not _volcano_clf_available),
            )
        with col_comp_info:
            # Toujours st.info — DOM stable
            if "vcf_compare_key" in st.session_state and not df_filt_c.empty:
                _ck = st.session_state["vcf_compare_key"]
                st.info(f"Résultats pour {_ck[0]}/{_ck[1]:02d} ({_ck[2]:,} images).")
            elif df_filt_c.empty:
                st.info("Aucune image pour cette sélection.")
            else:
                st.info(f"Cliquez pour comparer {len(df_filt_c):,} images — {sel_year_c}/{sel_month_c:02d}.")

        if not df_filt_c.empty and _volcano_clf_available and _run_compare:
            df_comp = df_filt_c.copy()
            if df_feat is not None:
                df_comp = df_comp.merge(
                    df_feat[["filename"] + [c for c in df_feat.columns if c != "filename"]],
                    on="filename", how="left",
                )

            # Heuristique
            from src.models.volcano_classifier import classify_heuristic
            with st.spinner(f"Classification heuristique + RF ({len(df_comp):,} images)…"):
                df_comp["class_heuristic"] = [
                    classify_heuristic(row.to_dict())
                    for _, row in df_comp.iterrows()
                ]

                # Modèle (ou heuristique si pas de modèle)
                clf_comp = VolcanoClassifier()
                if clf_path.exists():
                    try:
                        clf_comp = _cached_load_volcano_classifier(str(clf_path))
                    except Exception:
                        pass
                df_comp["class_model"] = clf_comp.predict(df_comp)

            df_comp["agreement"] = df_comp["class_heuristic"] == df_comp["class_model"]
            st.session_state["vcf_compare_df"] = df_comp
            st.session_state["vcf_compare_key"] = (sel_year_c, sel_month_c, len(df_comp))

        # Pattern obligatoire : colonnes et graphiques TOUJOURS rendus.
        # if/else sur session_state modifie la structure DOM → removeChild.
        _ck2 = st.session_state.get("vcf_compare_key", (sel_year_c, sel_month_c, 0))
        _cls_colors = {"pyroclastique": "#e74c3c", "lave": "#f39c12",
                       "nuage": "#95a5a6", "normal": "#2ecc71"}

        if "vcf_compare_df" in st.session_state:
            df_comp = st.session_state["vcf_compare_df"]
            agree_pct = df_comp["agreement"].mean() * 100
            _agree_label = f"{agree_pct:.1f}%"
            _disagree_count = int((~df_comp["agreement"]).sum())
            hc = df_comp["class_heuristic"].value_counts()
            mc = df_comp["class_model"].value_counts()
        else:
            df_comp = pd.DataFrame()
            _agree_label = "—"
            _disagree_count = 0
            hc = pd.Series(dtype=int)
            mc = pd.Series(dtype=int)

        col_a1, col_a2 = st.columns(2)
        col_a1.metric("Accord heuristique/modèle", _agree_label)
        col_a2.metric("Désaccords", _disagree_count)

        # st.caption stable — jamais un st.info conditionnel entre widgets structuraux (anti-removeChild)
        st.caption(
            "Cliquez sur ▶️ Comparer pour lancer l'analyse de la période sélectionnée."
            if df_comp.empty else
            f"Analyse effectuée — {len(df_comp):,} images comparées."
        )

        st.divider()
        col_h, col_m = st.columns(2)

        with col_h:
            st.markdown("**Classification heuristique**")
            if not hc.empty:
                fig_hc = go.Figure(go.Bar(
                    x=hc.index.tolist(), y=hc.values.tolist(),
                    marker_color=[_cls_colors.get(c, "#3498db") for c in hc.index],
                    hovertemplate="%{x} : %{y}<extra></extra>",
                ))
                fig_hc.update_layout(
                    title=f"Heuristique — {_ck2[0]}/{_ck2[1]:02d}",
                    height=300, margin=dict(t=40, b=30), showlegend=False,
                )
            else:
                fig_hc = go.Figure()
                fig_hc.update_layout(title="Heuristique", height=300,
                                     annotations=[dict(text="Aucun résultat", showarrow=False,
                                                       font=dict(size=14), xref="paper", yref="paper")])
            st.plotly_chart(fig_hc, use_container_width=True, key="vcf_compare_heuristic")

        with col_m:
            st.markdown("**Classification modèle (RF / heuristique)**")
            if not mc.empty:
                fig_mc = go.Figure(go.Bar(
                    x=mc.index.tolist(), y=mc.values.tolist(),
                    marker_color=[_cls_colors.get(c, "#3498db") for c in mc.index],
                    hovertemplate="%{x} : %{y}<extra></extra>",
                ))
                fig_mc.update_layout(
                    title=f"Modèle — {_ck2[0]}/{_ck2[1]:02d}",
                    height=300, margin=dict(t=40, b=30), showlegend=False,
                )
            else:
                fig_mc = go.Figure()
                fig_mc.update_layout(title="Modèle", height=300,
                                     annotations=[dict(text="Aucun résultat", showarrow=False,
                                                       font=dict(size=14), xref="paper", yref="paper")])
            st.plotly_chart(fig_mc, use_container_width=True, key="vcf_compare_model")

        # Table des désaccords — subheader + dataframe TOUJOURS rendus.
        # Un subheader conditionnel (absent avant le 1er clic) change la structure DOM
        # de Tab 4 entre les runs → removeChild sur tous les nœuds suivants.
        st.subheader("📋 Images en désaccord")
        if not df_comp.empty:
            _disagree = df_comp[~df_comp["agreement"]].copy() if not df_comp["agreement"].all() else pd.DataFrame()
            if not _disagree.empty:
                _disagree_cols = ["filename", "class_heuristic", "class_model"] + [
                    c for c in ["day", "hour", "patchcore_score"] if c in _disagree.columns
                ]
                disagree_display = _disagree[_disagree_cols]
            else:
                disagree_display = pd.DataFrame(columns=["filename", "class_heuristic", "class_model"])
        else:
            disagree_display = pd.DataFrame(columns=["filename", "class_heuristic", "class_model"])
        st.caption(
            "Accord complet entre heuristique et modèle sur cette période."
            if (not df_comp.empty and ("agreement" in df_comp.columns) and df_comp["agreement"].all())
            else ("Lancez la comparaison pour voir les désaccords." if df_comp.empty else "")
        )
        st.dataframe(disagree_display, use_container_width=True, hide_index=True,
                     key="vcf_compare_disagree")


def main() -> None:
    config = load_config()
    df = load_index()

    # ----- Sidebar -----
    st.sidebar.title("🌋 VolcIA")
    st.sidebar.caption("Merapi — IA Générative & Volcanologie")

    # Bouton vidage cache — utile après recalcul des scores ou mise à jour de l'index
    if st.sidebar.button("🔄 Actualiser les données", help="Vide le cache Streamlit et recharge l'index + scores"):
        st.cache_data.clear()
        st.rerun()

    # ── Indicateur de couverture PatchCore ────────────────────────────────
    if not df.empty and "on_disk" in df.columns:
        _sb_disk = int(df["on_disk"].sum())
        _sb_scored = int(df["patchcore_score"].notna().sum()) if "patchcore_score" in df.columns else 0
        if _sb_disk > 0:
            _sb_cov = _sb_scored / _sb_disk * 100
            if _sb_cov < 90:
                st.sidebar.warning(
                    f"⚠️ Couverture PatchCore : {_sb_cov:.0f}%\n"
                    f"{_sb_scored:,} / {_sb_disk:,} fichiers scorés\n\n"
                    "Voir **🏠 Accueil → 🔍 Couverture pipeline**"
                )
            else:
                st.sidebar.success(
                    f"✅ Couverture {_sb_cov:.0f}%\n{_sb_scored:,} / {_sb_disk:,} scorées"
                )
        st.sidebar.markdown("---")

    page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed", key="nav_page")

    # ── Contrôles contextuels PatchCore ──────────────────────────────────
    if page == PAGES[4]:  # 🔬 DINOv2 + PatchCore
        st.sidebar.markdown("---")
        st.sidebar.markdown("**⚙️ Contrôles PatchCore**")
        st.sidebar.slider(
            "Seuil anomalie (percentile)",
            50, 99, 90, 1,
            key="pc_threshold_pct",
            help="Images au-dessus de ce percentile sont signalées comme anomales.",
        )
        st.sidebar.slider(
            "Opacité heatmap",
            0.0, 1.0, 0.45, 0.05,
            key="sidebar_hm_opacity",
            help="Transparence de la heatmap superposée sur l'image.",
        )
        st.sidebar.selectbox(
            "Colormap heatmap",
            ["Hot", "Viridis", "RdYlBu_r", "Plasma", "Inferno", "YlOrRd"],
            key="sidebar_hm_cmap",
        )
        st.sidebar.toggle(
            "Afficher clusters",
            value=True,
            key="sidebar_show_clusters",
            help="Active la visualisation PCA dans l'onglet Espace des features.",
        )
        st.sidebar.toggle(
            "Module pédagogique",
            value=True,
            key="sidebar_show_pedagogy",
            help="Affiche l'onglet 'Comprendre PatchCore'.",
        )

    # ----- Routing -----
    if page == PAGES[0]:            # Accueil
        page_accueil(df)
    elif page == PAGES[1]:          # Exploration
        page_exploration(df, config)
    elif page == PAGES[2]:          # Galerie temporelle
        page_galerie(df, config)
    elif page == PAGES[3]:          # Anomalies
        page_anomalies(df, config)
    elif page == PAGES[4]:          # DINOv2 + PatchCore
        page_patchcore(df, config)
    elif page == PAGES[5]:          # Timeline PatchCore
        page_timeline_patchcore()
    elif page == PAGES[6]:          # Early Warning
        page_early_warning(df)
    elif page == PAGES[7]:          # Analyse volcanique avancée
        page_analyse_volcanique(df, config)
    elif page == PAGES[8]:          # Analyse avancée
        page_analyse_avancee(df, config)
    elif page == PAGES[9]:          # Simulation
        page_simulation()
    elif page == PAGES[10]:         # À propos (toujours en dernier)
        page_a_propos()


# ==============================================================
# Point d'entrée
# ==============================================================
if __name__ == "__main__":
    main()
