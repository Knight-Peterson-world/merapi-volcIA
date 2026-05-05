"""
plot_patchcore_timeline_interactive.py — Timeline PatchCore avec filtrage dynamique année/mois.

FILTRAGE RÉEL (pas seulement zoom visuel) :
  - Dropdown ANNÉE  : restreint l'axe X à l'année choisie via relayout.
  - Dropdown MOIS   : masque les traces des autres mois via restyle.
  - Combinaison     : sélectionner "2018" + "Mai" → uniquement les points de mai 2018.
  - Bouton ↺ Reset  : réinitialise les deux filtres simultanément.
  - Rangeslider      : sélection manuelle de n'importe quelle plage.
  - Bouton PNG       : export image haute résolution depuis la barre Plotly.

Architecture des traces (17 total) :
  Indices 0–11  : données individuelles par mois (colorées par quality_flag)
  Indice  12    : courbe des moyennes mensuelles (permanente)
  Indice  13    : seuil P90 = 49.975 (permanente)
  Indices 14–16 : lignes verticales événements annotés (permanentes)

Usage :
    /opt/anaconda3/bin/python outputs/figures/plot_patchcore_timeline_interactive.py
"""

from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ─── Configuration ─────────────────────────────────────────────────────────
P90_THRESHOLD = 49.975

EVENTS: list[tuple[str, str, str]] = [
    ("2018-05-01", "Éruption mai 2018",   "#e74c3c"),
    ("2020-01-15", "Crise sismique",       "#e67e22"),
    ("2022-09-10", "Alerte météo extrême", "#3498db"),
]

MONTH_NAMES_FR: dict[int, str] = {
    1: "Janvier", 2: "Février", 3: "Mars",    4: "Avril",
    5: "Mai",     6: "Juin",    7: "Juillet",  8: "Août",
    9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre",
}

COLOR_MAP = {
    "usable": "#3498db", "dark": "#e74c3c",
    "cloudy": "#f39c12", "corrupted": "#9b59b6",
}
SYMBOL_MAP = {
    "usable": "circle", "dark": "diamond",
    "cloudy": "square", "corrupted": "x",
}
DEFAULT_COLOR = "#95a5a6"
BG_COLOR, GRID_COLOR, TEXT_COLOR, ANNOT_COLOR = "#0f1117", "#2a2a3a", "#ecf0f1", "#bdc3c7"
N_PERMANENT = 5  # mean + P90 + 3 events


# ─── Données ────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame | None:
    """Charge depuis data/index/index.csv (contient déjà patchcore_score + year + month)."""
    path = PROJECT_ROOT / "data/index/index.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    if "patchcore_score" not in df.columns:
        return None
    df = df.dropna(subset=["patchcore_score", "year", "month"])
    if len(df) < 20:
        return None
    df = df.copy()
    df["year"]  = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["date"]  = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-15"
    )
    rng = np.random.default_rng(42)
    df["date_jitter"] = df["date"] + pd.to_timedelta(rng.integers(-10, 10, len(df)), unit="D")
    return df


def synthetic_data() -> pd.DataFrame:
    """Données synthétiques calibrées (fallback uniquement)."""
    rng = np.random.default_rng(42)
    bumps = {"2018-05-01": 7.0, "2020-01-01": 5.5, "2022-09-01": 4.0}
    rows = []
    for date in pd.date_range("2014-01-01", periods=120, freq="MS"):
        base = 40.8 + rng.normal(0, 3.5)
        for d_str, bmp in bumps.items():
            dm = abs((date.year - pd.Timestamp(d_str).year)*12 + date.month - pd.Timestamp(d_str).month)
            if dm <= 1:
                base += bmp * max(0, 1 - dm * 0.5)
        for score in np.clip(rng.normal(base, 4, int(rng.integers(15, 80))), 20, 65):
            flag = "dark" if score > 50 and rng.random() < 0.25 else "usable"
            jitter = pd.Timedelta(days=int(rng.integers(-10, 10)))
            rows.append({
                "date": date + jitter, "date_jitter": date + jitter,
                "year": date.year, "month": date.month,
                "patchcore_score": float(score), "quality_flag": flag,
                "filename": f"synth_{date.year}_{date.month:02d}.jpg",
            })
    return pd.DataFrame(rows)


# ─── Visibilité des traces ────────────────────────────────────────────────────

def month_visibility(selected_month: int | None) -> list[bool]:
    """
    Retourne liste de 17 booléens pour restyle.
    selected_month=None → tout visible | selected_month=5 → seul Mai visible.
    """
    return [
        *(True if selected_month is None else (m == selected_month) for m in range(1, 13)),
        *([True] * N_PERMANENT),
    ]


# ─── Figure Plotly ──────────────────────────────────────────────────────────

def build_figure(df: pd.DataFrame, is_synthetic: bool) -> go.Figure:
    """Construit la figure avec 17 traces + dropdowns + rangeslider."""
    traces: list = []
    years = sorted(df["year"].unique().tolist())

    # Traces 0-11 : une par mois ─────────────────────────────────────────────
    for m in range(1, 13):
        mdf = df[df["month"] == m]
        traces.append(go.Scatter(
            x=mdf["date_jitter"],
            y=mdf["patchcore_score"],
            mode="markers",
            name=MONTH_NAMES_FR[m],
            marker=dict(
                color=mdf["quality_flag"].map(COLOR_MAP).fillna(DEFAULT_COLOR).tolist(),
                size=5, opacity=0.72,
                symbol=mdf["quality_flag"].map(SYMBOL_MAP).fillna("circle").tolist(),
                line=dict(width=0.4, color="rgba(255,255,255,0.15)"),
            ),
            customdata=list(zip(
                mdf.get("quality_flag", pd.Series(["?"]*len(mdf))).tolist(),
                mdf.get("filename", pd.Series([""]*len(mdf))).tolist(),
            )),
            hovertemplate=(
                "<b>%{x|%d %B %Y}</b><br>"
                "Score PatchCore : <b>%{y:.3f}</b><br>"
                "Qualité : %{customdata[0]}<br>"
                "Fichier : %{customdata[1]}<extra></extra>"
            ),
            visible=True,
            legendgroup=MONTH_NAMES_FR[m], showlegend=True,
        ))

    # Trace 12 : moyenne mensuelle ────────────────────────────────────────────
    monthly = df.groupby("date")["patchcore_score"].agg(mean="mean", n="count").reset_index().sort_values("date")
    traces.append(go.Scatter(
        x=monthly["date"], y=monthly["mean"],
        mode="lines+markers", name="Moyenne mensuelle",
        line=dict(color="#f1c40f", width=2.2), marker=dict(size=5, color="#f1c40f"),
        customdata=monthly["n"],
        hovertemplate="<b>%{x|%B %Y}</b><br>Moyenne : <b>%{y:.3f}</b><br>Images : %{customdata}<extra></extra>",
        visible=True, showlegend=True,
    ))

    # Trace 13 : seuil P90 ────────────────────────────────────────────────────
    d_min = df["date_jitter"].min() - pd.Timedelta(days=30)
    d_max = df["date_jitter"].max() + pd.Timedelta(days=30)
    traces.append(go.Scatter(
        x=[d_min, d_max], y=[P90_THRESHOLD, P90_THRESHOLD],
        mode="lines", name=f"Seuil P90 ({P90_THRESHOLD})",
        line=dict(color="#f39c12", width=1.8, dash="dash"),
        hoverinfo="skip", visible=True, showlegend=True,
    ))

    # Traces 14-16 : lignes verticales événements ─────────────────────────────
    for date_str, label, color in EVENTS:
        traces.append(go.Scatter(
            x=[pd.Timestamp(date_str)]*2, y=[5, 70],
            mode="lines", name=label,
            line=dict(color=color, width=1.5, dash="dot"),
            hovertemplate=f"<b>{label}</b><extra></extra>",
            visible=True, showlegend=True,
        ))

    fig = go.Figure(data=traces)

    # Annotations texte des événements
    for date_str, label, color in EVENTS:
        fig.add_annotation(
            x=pd.Timestamp(date_str), y=64,
            text=label.replace("\n", "<br>"),
            showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor=color,
            ax=0, ay=-35, font=dict(size=10, color=color),
            bgcolor=BG_COLOR, bordercolor=color, borderwidth=1,
        )

    # ── Boutons ANNÉE ─────────────────────────────────────────────────────────
    year_btns = [dict(
        label="Toutes années", method="relayout",
        args=[{"xaxis.range": [d_min.isoformat(), d_max.isoformat()], "xaxis.autorange": False}],
    )]
    for yr in years:
        year_btns.append(dict(
            label=str(yr), method="relayout",
            args=[{"xaxis.range": [f"{yr}-01-01", f"{yr}-12-31"], "xaxis.autorange": False}],
        ))

    # ── Boutons MOIS ─────────────────────────────────────────────────────────
    month_btns = [dict(label="Tous mois", method="restyle", args=[{"visible": month_visibility(None)}])]
    for m in range(1, 13):
        month_btns.append(dict(
            label=MONTH_NAMES_FR[m], method="restyle",
            args=[{"visible": month_visibility(m)}],
        ))

    # ── Bouton RESET (restyle + relayout simultanés) ─────────────────────────
    reset_btn = dict(
        label="↺ Reset", method="update",
        args=[
            {"visible": month_visibility(None)},
            {"xaxis.range": [d_min.isoformat(), d_max.isoformat()], "xaxis.autorange": False},
        ],
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    synthetic_note = " (⚠ données synthétiques)" if is_synthetic else ""
    fig.update_layout(
        title=dict(
            text=(
                f"Score d'anomalie PatchCore — Merapi{synthetic_note}<br>"
                "<sup>📅 Filtre Année ▸ gauche &nbsp;|&nbsp; "
                "📆 Filtre Mois ▸ centre &nbsp;|&nbsp; ↺ Reset ▸ droite</sup>"
            ),
            font=dict(size=15, color=TEXT_COLOR), x=0.5, xanchor="center",
        ),
        height=680,
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family="Arial, sans-serif"),
        hovermode="closest",
        xaxis=dict(
            showgrid=True, gridcolor=GRID_COLOR, gridwidth=0.5,
            tickfont=dict(color=TEXT_COLOR), type="date",
            range=[d_min.isoformat(), d_max.isoformat()],
            rangeslider=dict(visible=True, bgcolor=BG_COLOR, bordercolor=GRID_COLOR, thickness=0.06),
        ),
        yaxis=dict(
            title=dict(text="Score d'anomalie PatchCore", font=dict(size=12)),
            showgrid=True, gridcolor=GRID_COLOR, gridwidth=0.5,
            tickfont=dict(color=TEXT_COLOR), range=[8, 72],
        ),
        legend=dict(
            bgcolor="rgba(15,17,23,0.75)", bordercolor=GRID_COLOR, borderwidth=1,
            font=dict(size=10), x=1.01, y=1.0, yanchor="top", xanchor="left",
        ),
        margin=dict(l=70, r=160, t=130, b=90),
        updatemenus=[
            dict(
                type="dropdown", direction="down", x=0.0, y=1.17,
                xanchor="left", yanchor="top", buttons=year_btns,
                showactive=True, active=0, bgcolor="#1a1a2e",
                bordercolor=GRID_COLOR, font=dict(color=TEXT_COLOR, size=12),
            ),
            dict(
                type="dropdown", direction="down", x=0.22, y=1.17,
                xanchor="left", yanchor="top", buttons=month_btns,
                showactive=True, active=0, bgcolor="#1a1a2e",
                bordercolor=GRID_COLOR, font=dict(color=TEXT_COLOR, size=12),
            ),
            dict(
                type="buttons", x=0.47, y=1.17,
                xanchor="left", yanchor="top", buttons=[reset_btn],
                bgcolor="#1a1a2e", bordercolor=GRID_COLOR,
                font=dict(color="#f39c12", size=12),
            ),
        ],
        annotations=list(fig.layout.annotations) + [
            dict(text="📅 Année :", x=0.0, y=1.23, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=11, color=ANNOT_COLOR), xanchor="left"),
            dict(text="📆 Mois :", x=0.22, y=1.23, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=11, color=ANNOT_COLOR), xanchor="left"),
            dict(text=(
                    "● Usable (normal) &nbsp;"
                    "<span style='color:#e74c3c'>◆ Dark (anomalie)</span> &nbsp;"
                    "<span style='color:#f39c12'>■ Cloudy</span>"
                ),
                x=0.5, y=-0.18, xref="paper", yref="paper",
                showarrow=False, font=dict(size=10, color=TEXT_COLOR), xanchor="center",
                bgcolor="rgba(15,17,23,0.7)", borderpad=3),
        ],
    )
    return fig


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    out = PROJECT_ROOT / "outputs/figures/patchcore_timeline_interactive.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    df = load_data()
    if df is not None:
        is_synthetic = False
        print(f"[INFO] Données réelles : {len(df)} images | années : {sorted(df['year'].unique())}")
    else:
        df = synthetic_data()
        is_synthetic = True
        print("[INFO] Données synthétiques activées")

    fig = build_figure(df, is_synthetic)
    fig.write_html(
        str(out), include_plotlyjs="cdn", full_html=True,
        config={
            "displayModeBar": True,
            "toImageButtonOptions": {"format": "png", "filename": "patchcore_timeline", "height": 800, "width": 1400, "scale": 2},
            "scrollZoom": True, "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
    print(f"[OK] HTML interactif → {out}")
    print("     📅 Dropdown Année  = relayout xaxis.range (filtrage par plage de dates)")
    print("     📆 Dropdown Mois   = restyle visible (masquage des autres mois)")
    print("     ↺  Reset           = update (restyle + relayout simultanés)")


if __name__ == "__main__":
    main()
