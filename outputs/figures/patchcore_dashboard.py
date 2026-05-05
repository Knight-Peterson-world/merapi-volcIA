"""
patchcore_dashboard.py — Application Dash interactive pour explorer les scores PatchCore.

Fonctionnalités :
  - Dropdown Année  : filtre les données sur l'année sélectionnée
  - Dropdown Mois   : filtre les données sur le mois sélectionné
  - Bouton Reset    : réinitialise les deux filtres
  - Graphique Plotly mis à jour dynamiquement via callbacks Python
  - Histogramme de distribution des scores (panneau du bas)
  - KPI cards : N images, score moyen, score max, % > P90

Usage :
    /opt/anaconda3/bin/python outputs/figures/patchcore_dashboard.py
    → Ouvrir http://127.0.0.1:8050 dans un navigateur

Dépendances :
    pip install dash plotly pandas numpy
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dcc, html

# ─── Configuration ──────────────────────────────────────────────────────────

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
BG_COLOR, GRID_COLOR = "#0f1117", "#2a2a3a"
TEXT_COLOR, ACCENT = "#ecf0f1", "#f1c40f"
CARD_BG = "#1a1a2e"


# ─── Chargement des données ──────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Charge index.csv ou renvoie des données synthétiques."""
    path = PROJECT_ROOT / "data/index/index.csv"
    if path.exists():
        df = pd.read_csv(path, low_memory=False)
        if "patchcore_score" in df.columns:
            df = df.dropna(subset=["patchcore_score", "year", "month"])
            if len(df) >= 20:
                df = df.copy()
                df["year"]  = df["year"].astype(int)
                df["month"] = df["month"].astype(int)
                df["date"]  = pd.to_datetime(
                    df["year"].astype(str) + "-"
                    + df["month"].astype(str).str.zfill(2) + "-15"
                )
                rng = np.random.default_rng(42)
                df["date_jitter"] = df["date"] + pd.to_timedelta(
                    rng.integers(-10, 10, len(df)), unit="D"
                )
                df["quality_flag"] = df.get("quality_flag", pd.Series(["usable"]*len(df)))
                print(f"[Dash] {len(df)} images chargées depuis index.csv")
                return df

    # Fallback synthétique
    print("[Dash] Données synthétiques (index insuffisant)")
    rng = np.random.default_rng(42)
    rows = []
    for date in pd.date_range("2014-01-01", periods=120, freq="MS"):
        base = 40.8 + rng.normal(0, 3.5)
        for score in np.clip(rng.normal(base, 4, int(rng.integers(15, 80))), 20, 65):
            jitter = pd.Timedelta(days=int(rng.integers(-10, 10)))
            rows.append({
                "date": date + jitter, "date_jitter": date + jitter,
                "year": date.year, "month": date.month,
                "patchcore_score": float(score),
                "quality_flag": "dark" if score > 50 and rng.random() < 0.25 else "usable",
                "filename": f"synth_{date.year}_{date.month:02d}.jpg",
            })
    return pd.DataFrame(rows)


DF_FULL = load_data()
ALL_YEARS = sorted(DF_FULL["year"].unique().tolist())


# ─── Composants graphiques ───────────────────────────────────────────────────

def make_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter timeline des scores."""
    fig = go.Figure()

    # Points par quality_flag
    for flag, color in COLOR_MAP.items():
        sub = df[df["quality_flag"] == flag]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["date_jitter"], y=sub["patchcore_score"],
            mode="markers", name=flag.capitalize(),
            marker=dict(
                color=color, size=5, opacity=0.72,
                symbol=SYMBOL_MAP.get(flag, "circle"),
                line=dict(width=0.4, color="rgba(255,255,255,0.15)"),
            ),
            customdata=sub.get("filename", pd.Series([""]*len(sub))).tolist(),
            hovertemplate=(
                "<b>%{x|%d %B %Y}</b><br>"
                "Score : <b>%{y:.3f}</b><br>"
                f"Qualité : {flag}<br>"
                "Fichier : %{customdata}<extra></extra>"
            ),
        ))

    # Moyenne mensuelle
    monthly = df.groupby("date")["patchcore_score"].agg(mean="mean", n="count").reset_index()
    fig.add_trace(go.Scatter(
        x=monthly["date"], y=monthly["mean"],
        mode="lines+markers", name="Moyenne mensuelle",
        line=dict(color=ACCENT, width=2.2), marker=dict(size=5, color=ACCENT),
        customdata=monthly["n"],
        hovertemplate="<b>%{x|%B %Y}</b><br>Moy : <b>%{y:.3f}</b><br>N=%{customdata}<extra></extra>",
    ))

    # Seuil P90
    if not df.empty:
        d_min = df["date_jitter"].min() - pd.Timedelta(days=30)
        d_max = df["date_jitter"].max() + pd.Timedelta(days=30)
        fig.add_trace(go.Scatter(
            x=[d_min, d_max], y=[P90_THRESHOLD, P90_THRESHOLD],
            mode="lines", name=f"P90 = {P90_THRESHOLD}",
            line=dict(color="#f39c12", width=1.8, dash="dash"),
            hoverinfo="skip",
        ))

        # Lignes événements
        for date_str, label, color in EVENTS:
            ts = pd.Timestamp(date_str)
            if d_min <= ts <= d_max:
                fig.add_vline(x=ts, line=dict(color=color, width=1.5, dash="dot"))
                fig.add_annotation(
                    x=ts, y=64, text=label.replace("\n", "<br>"),
                    showarrow=True, arrowhead=2, arrowcolor=color,
                    ax=0, ay=-30, font=dict(size=9, color=color),
                    bgcolor=BG_COLOR, bordercolor=color, borderwidth=1,
                )

    fig.update_layout(
        height=430,
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family="Arial, sans-serif"),
        hovermode="closest",
        xaxis=dict(
            showgrid=True, gridcolor=GRID_COLOR, tickfont=dict(color=TEXT_COLOR), type="date",
        ),
        yaxis=dict(
            title="Score PatchCore", showgrid=True, gridcolor=GRID_COLOR,
            tickfont=dict(color=TEXT_COLOR), range=[8, 72],
        ),
        legend=dict(
            bgcolor="rgba(15,17,23,0.75)", bordercolor=GRID_COLOR, borderwidth=1,
            font=dict(size=10), x=1.01, y=1.0, yanchor="top", xanchor="left",
        ),
        margin=dict(l=60, r=150, t=20, b=40),
    )
    return fig


def make_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogramme de distribution des scores PatchCore."""
    fig = go.Figure()
    for flag, color in COLOR_MAP.items():
        sub = df[df["quality_flag"] == flag]
        if sub.empty:
            continue
        fig.add_trace(go.Histogram(
            x=sub["patchcore_score"], name=flag.capitalize(),
            marker_color=color, opacity=0.7, nbinsx=40,
            hovertemplate=f"[{flag}] %{{x:.1f}} — %{{y}} images<extra></extra>",
        ))

    fig.add_vline(x=P90_THRESHOLD, line=dict(color="#f39c12", dash="dash", width=2))
    fig.add_annotation(
        x=P90_THRESHOLD + 0.5, y=1, yref="paper",
        text=f"P90 = {P90_THRESHOLD}", showarrow=False,
        font=dict(size=10, color="#f39c12"), xanchor="left",
    )
    fig.update_layout(
        height=220, barmode="stack",
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family="Arial, sans-serif"),
        xaxis=dict(title="Score PatchCore", showgrid=True, gridcolor=GRID_COLOR,
                   tickfont=dict(color=TEXT_COLOR)),
        yaxis=dict(title="Images", showgrid=True, gridcolor=GRID_COLOR,
                   tickfont=dict(color=TEXT_COLOR)),
        legend=dict(bgcolor="rgba(15,17,23,0.75)", bordercolor=GRID_COLOR,
                    font=dict(size=10), x=1.01, y=1.0, xanchor="left"),
        margin=dict(l=60, r=150, t=10, b=40),
    )
    return fig


def kpi_card(label: str, value: str, color: str = TEXT_COLOR) -> html.Div:
    return html.Div([
        html.Div(label, style={"fontSize": "11px", "color": "#95a5a6", "marginBottom": "2px"}),
        html.Div(value, style={"fontSize": "22px", "fontWeight": "bold", "color": color}),
    ], style={
        "background": CARD_BG, "borderRadius": "8px",
        "padding": "12px 18px", "minWidth": "130px", "textAlign": "center",
    })


def compute_kpis(df: pd.DataFrame) -> tuple[str, str, str, str]:
    if df.empty:
        return "0", "—", "—", "—"
    n     = str(len(df))
    mean  = f"{df['patchcore_score'].mean():.2f}"
    mx    = f"{df['patchcore_score'].max():.2f}"
    pct   = f"{(df['patchcore_score'] > P90_THRESHOLD).mean()*100:.1f}%"
    return n, mean, mx, pct


# ─── Application Dash ────────────────────────────────────────────────────────

app = Dash(__name__, title="PatchCore — Merapi Dashboard")

YEAR_OPTIONS  = [{"label": "Toutes années", "value": "all"}] + [{"label": str(y), "value": y} for y in ALL_YEARS]
MONTH_OPTIONS = [{"label": "Tous mois", "value": "all"}] + [{"label": MONTH_NAMES_FR[m], "value": m} for m in range(1, 13)]

LABEL_STYLE = {"color": "#bdc3c7", "fontSize": "12px", "marginBottom": "4px"}
DROPDOWN_STYLE = {"backgroundColor": CARD_BG, "color": TEXT_COLOR, "width": "160px"}

app.layout = html.Div(
    style={"backgroundColor": BG_COLOR, "minHeight": "100vh", "padding": "20px 30px",
           "fontFamily": "Arial, sans-serif", "color": TEXT_COLOR},
    children=[

        # ── En-tête ────────────────────────────────────────────────────────
        html.H2("Score d'anomalie PatchCore — Merapi", style={"textAlign": "center", "color": ACCENT, "marginBottom": "4px"}),
        html.P("Dashboard interactif — filtre réel par année et mois",
               style={"textAlign": "center", "color": "#95a5a6", "fontSize": "13px", "marginBottom": "20px"}),

        # ── Filtres ────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Div("📅 Année", style=LABEL_STYLE),
                dcc.Dropdown(id="year-filter", options=YEAR_OPTIONS, value="all",
                             clearable=False, style=DROPDOWN_STYLE,
                             className="dash-dropdown-dark"),
            ], style={"marginRight": "20px"}),
            html.Div([
                html.Div("📆 Mois", style=LABEL_STYLE),
                dcc.Dropdown(id="month-filter", options=MONTH_OPTIONS, value="all",
                             clearable=False, style=DROPDOWN_STYLE,
                             className="dash-dropdown-dark"),
            ], style={"marginRight": "20px"}),
            html.Div([
                html.Div("\u00a0", style=LABEL_STYLE),
                html.Button("↺ Reset", id="reset-btn", n_clicks=0,
                            style={
                                "backgroundColor": CARD_BG, "color": "#f39c12",
                                "border": f"1px solid {GRID_COLOR}", "borderRadius": "6px",
                                "padding": "6px 18px", "cursor": "pointer", "fontSize": "14px",
                            }),
            ]),
        ], style={"display": "flex", "alignItems": "flex-end", "marginBottom": "20px"}),

        # ── KPI Cards ──────────────────────────────────────────────────────
        html.Div(id="kpi-row", style={"display": "flex", "gap": "12px", "marginBottom": "18px"}),

        # ── Graphique principal ────────────────────────────────────────────
        dcc.Graph(id="scatter-plot", config={"displayModeBar": True, "scrollZoom": True,
                                              "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                                              "toImageButtonOptions": {"format": "png", "filename": "patchcore_merapi", "scale": 2}}),

        # ── Histogramme ────────────────────────────────────────────────────
        html.Div("Distribution des scores", style={"color": "#95a5a6", "fontSize": "12px", "marginTop": "10px", "marginBottom": "4px"}),
        dcc.Graph(id="histogram", config={"displayModeBar": False}),
    ],
)


# ─── Callbacks ───────────────────────────────────────────────────────────────

@callback(
    Output("year-filter",  "value"),
    Output("month-filter", "value"),
    Input("reset-btn",     "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(_n: int):
    """Réinitialise les deux dropdowns."""
    return "all", "all"


@callback(
    Output("scatter-plot", "figure"),
    Output("histogram",    "figure"),
    Output("kpi-row",      "children"),
    Input("year-filter",   "value"),
    Input("month-filter",  "value"),
)
def update_charts(year_val, month_val):
    """Filtre les données et met à jour les graphiques + KPIs."""
    df = DF_FULL.copy()
    if year_val != "all":
        df = df[df["year"] == int(year_val)]
    if month_val != "all":
        df = df[df["month"] == int(month_val)]

    n, mean, mx, pct = compute_kpis(df)
    cards = [
        kpi_card("Images", n),
        kpi_card("Score moyen", mean, ACCENT),
        kpi_card("Score max", mx, "#e74c3c" if df.empty else ("#e74c3c" if float(mx) > P90_THRESHOLD else TEXT_COLOR)),
        kpi_card(f"% > P90 ({P90_THRESHOLD})", pct, "#e74c3c" if not df.empty and float(pct.rstrip("%")) > 15 else TEXT_COLOR),
    ]
    return make_scatter(df), make_histogram(df), cards


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  PatchCore Dashboard — Merapi")
    print(f"  {len(DF_FULL)} images | Dash 4.x")
    print("=" * 60)
    print("  Ouvrez : http://127.0.0.1:8050")
    print("  Arrêter : Ctrl+C")
    print("=" * 60)
    app.run(debug=False, host="127.0.0.1", port=8050)
