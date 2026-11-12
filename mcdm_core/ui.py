"""Briques d'interface Streamlit communes à toutes les pages."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

INK, PAPER, RUST, TEAL = "#1b1f2a", "#f6f3ec", "#9a3412", "#1f4e5f"
PALETTE = ["#1f4e5f", "#9a3412", "#6b7f3a", "#7a4e7e", "#b08900", "#3b6ea5", "#8a5a44", "#5c5c5c"]
CMAP = LinearSegmentedColormap.from_list("papier_encre", ["#f6f3ec", "#cfdbd6", "#1f4e5f"])
CMAP_R = CMAP.reversed()
SEQ = [[0, "#f6f3ec"], [1, "#1f4e5f"]]
CMAP_DIV = LinearSegmentedColormap.from_list("rouille_encre", ["#9a3412", "#f6f3ec", "#1f4e5f"])

pio.templates["labo"] = go.layout.Template(layout=dict(
    font=dict(family="IBM Plex Sans, Helvetica, Arial, sans-serif", size=13, color=INK), colorway=PALETTE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=False, linecolor=INK, ticks="outside", zeroline=False), yaxis=dict(gridcolor="#ddd6c4", linecolor=INK, zeroline=False, ticks="outside"),
    title=dict(font=dict(family="Newsreader, Georgia, serif", size=17)), legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(l=10, r=10, t=50, b=10)))
pio.templates.default = "labo"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,600&display=swap');
html, body, [class*="css"], .stApp {font-family:'IBM Plex Sans',Helvetica,Arial,sans-serif;}
.stApp {background:#f6f3ec;}
header[data-testid="stHeader"] {background:transparent;}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDeployButton"] {display:none !important;}
.block-container {padding-top:2.2rem; max-width:1180px;}
h1, h2, h3, h4 {font-family:'Newsreader',Georgia,serif !important; font-weight:600 !important; letter-spacing:-0.01em; color:#1b1f2a;}
h1 {font-size:2.15rem !important; line-height:1.15; margin-bottom:.15rem;}
h2 {font-size:1.45rem !important; border-bottom:1px solid #b9b2a0; padding-bottom:.25rem; margin-top:2rem;}
h3, h4 {font-size:1.12rem !important;}
.kicker {font-family:'IBM Plex Mono',monospace; font-size:.74rem; letter-spacing:.14em; text-transform:uppercase; color:#9a3412; margin-bottom:.35rem;}
.lede {font-family:'Newsreader',Georgia,serif; font-size:1.15rem; color:#3a3f4b; max-width:46rem; line-height:1.45; margin:.2rem 0 .6rem 0;}
.refline {font-family:'IBM Plex Mono',monospace; font-size:.74rem; color:#6d6a5f; border-top:1px solid #1b1f2a; padding-top:.4rem; margin:.4rem 0 1.4rem 0;}
.tag {display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; padding:.12rem .5rem; border:1px solid #1b1f2a; margin:0 .5rem .4rem 0;}
.tag-cours {background:#1b1f2a; color:#f6f3ec;} .tag-cree {background:transparent; color:#1b1f2a;}
[data-testid="stSidebar"] {background:#ece6d6; border-right:1px solid #1b1f2a;}
[data-testid="stSidebar"] a {color:#1b1f2a !important;}
[data-testid="stSidebarNavSeparator"], [data-testid="stSidebarNav"] span[data-testid="stHeader"], [data-testid="stSidebarNav"] header {font-family:'IBM Plex Mono',monospace; text-transform:uppercase; letter-spacing:.1em; font-size:.68rem; color:#6d6a5f;}
[data-testid="stSidebarNavLink"][aria-current="page"] {background:#f6f3ec; border-left:3px solid #9a3412; border-radius:0; font-weight:600;}
[data-testid="stSidebarNavLink"] {border-radius:0;}
.stTabs [data-baseweb="tab-list"] {gap:0; border-bottom:1px solid #1b1f2a;}
.stTabs [data-baseweb="tab"] {font-family:'IBM Plex Mono',monospace; font-size:.78rem; letter-spacing:.05em; text-transform:uppercase; padding:.55rem 1rem; border-radius:0; color:#6d6a5f;}
.stTabs [aria-selected="true"] {color:#1b1f2a; font-weight:600;}
.stTabs [data-baseweb="tab-highlight"] {background:#9a3412; height:3px;}
[data-testid="stAlert"] {border-radius:0 !important; border:none; border-left:4px solid #1f4e5f; background:#ebe5d6 !important; color:#1b1f2a;}
[data-testid="stAlert"] svg {display:none;}
[data-testid="stAlertContentSuccess"] {color:#1b1f2a;}
div[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {border-left-color:#b08900;}
div[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {border-left-color:#9a3412;}
div[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {border-left-color:#6b7f3a;}
[data-testid="stExpander"] {border:1px solid #b9b2a0; border-radius:0; background:transparent;}
[data-testid="stExpander"] summary {font-family:'IBM Plex Mono',monospace; font-size:.78rem; letter-spacing:.05em; text-transform:uppercase;}
[data-testid="stMetric"] {border-top:2px solid #1b1f2a; padding-top:.3rem;}
[data-testid="stMetricLabel"] {font-family:'IBM Plex Mono',monospace; font-size:.7rem; text-transform:uppercase; letter-spacing:.08em; color:#6d6a5f;}
[data-testid="stMetricValue"] {font-family:'Newsreader',Georgia,serif; font-weight:600;}
.stButton button, .stDownloadButton button {border-radius:0; border:1px solid #1b1f2a; background:#1b1f2a; color:#f6f3ec; font-family:'IBM Plex Mono',monospace; letter-spacing:.06em; text-transform:uppercase; font-size:.78rem;}
.stButton button:hover, .stDownloadButton button:hover {background:#9a3412; border-color:#9a3412; color:#f6f3ec;}
[data-testid="stDataFrame"], [data-testid="stDataFrameResizable"] {border:1px solid #b9b2a0; border-radius:0;}
.stCaption, [data-testid="stCaptionContainer"] {color:#6d6a5f;}
code {font-family:'IBM Plex Mono',monospace; background:#ebe5d6; color:#1b1f2a; border-radius:0;}
"""


def inject_css():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
BEN = "Bénéfice (+)"
COST = "Coût (−)"


def header(title: str, subtitle: str, source: str | None = None, kicker: str | None = None):
    num, _, rest = title.partition(" · ")
    kick = kicker or (f"Chapitre {num}" if rest and num.isdigit() else "")
    if not rest or not num.isdigit(): rest = title
    if kick: st.markdown(f"<div class='kicker'>{kick}</div>", unsafe_allow_html=True)
    st.title(rest)
    st.markdown(f"<div class='lede'>{subtitle}</div>", unsafe_allow_html=True)
    if source: st.markdown(f"<div class='refline'>Référence : {source}</div>", unsafe_allow_html=True)


def course_box(md: str, title: str = "Rappel du cours"):
    with st.expander(title, expanded=False):
        st.markdown(md)


def note_source(ds_source: str, text: str = ""):
    tag = "<span class='tag tag-cours'>Données du cours</span>" if ds_source == "cours" else "<span class='tag tag-cree'>Données simulées</span>"
    st.markdown(tag + (f"<span style='font-size:.85rem;color:#6d6a5f'>{text}</span>" if text else ""), unsafe_allow_html=True)


def matrix_editor(ds: dict, key: str, with_weights: bool = True, default_w=None, fmt: str = "%.4g"):
    """Matrice de décision + types + poids éditables. Retourne X, benefit, w, alts, crit."""
    st.markdown("**Matrice de décision** (cliquez pour modifier une valeur)")
    base = pd.DataFrame(ds["X"], index=ds["alts"], columns=ds["crit"])
    Xdf = st.data_editor(base, key=f"{key}_X", width="stretch", num_rows="fixed")
    n = len(ds["crit"])
    w0 = ds["w"] if ds.get("w") is not None else (default_w if default_w is not None else np.full(n, 1 / n))
    meta = pd.DataFrame({"Critère": ds["crit"], "Type": [BEN if b else COST for b in ds["benefit"]], "Poids": np.asarray(w0, float)})
    cols = {"Type": st.column_config.SelectboxColumn(options=[BEN, COST], required=True), "Critère": st.column_config.TextColumn(disabled=True)}
    if not with_weights:
        meta = meta.drop(columns="Poids")
    M = st.data_editor(meta, key=f"{key}_meta", width="stretch", hide_index=True, column_config=cols)
    X = Xdf.to_numpy(float)
    benefit = (M["Type"] == BEN).to_numpy()
    w = None
    if with_weights:
        w = M["Poids"].to_numpy(float)
        if w.sum() <= 0:
            st.error("La somme des poids doit être > 0."); st.stop()
        if abs(w.sum() - 1) > 1e-6:
            st.caption(f"Les poids saisis somment à {w.sum():.3f} : ils sont normalisés automatiquement (Σw = 1).")
            w = w / w.sum()
    if np.isnan(X).any():
        st.error("La matrice contient des valeurs manquantes."); st.stop()
    return X, benefit, w, list(Xdf.index), list(Xdf.columns)


def steps(tables: dict[str, pd.DataFrame | np.ndarray], index=None, columns=None, fmt: int = 4, title="Détail du calcul"):
    with st.expander(title, expanded=False):
        for name, t in tables.items():
            st.markdown(f"**{name}**")
            if not isinstance(t, (pd.DataFrame, pd.Series)):
                t = pd.DataFrame(t, index=index, columns=columns)
            st.dataframe(t.round(fmt), width="stretch")


def bar(values, labels, title, ylabel="", sort=True, color=None, horizontal=False, fmt=".3f"):
    s = pd.Series(values, index=labels)
    if sort: s = s.sort_values(ascending=horizontal)
    fig = px.bar(x=s.values if horizontal else s.index, y=s.index if horizontal else s.values, title=title,
                 color_discrete_sequence=[color or PALETTE[0]], text=[f"{v:{fmt}}" for v in s.values])
    fig.update_layout(xaxis_title="" if not horizontal else ylabel, yaxis_title=ylabel if not horizontal else "",
                      margin=dict(l=10, r=10, t=50, b=10), height=360, showlegend=False)
    fig.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, width="stretch")


def weights_chart(w, crit, title="Poids des critères"):
    short = [c.split(" ")[0] if len(c) > 18 else c for c in crit]
    bar(w, short, title, "poids", sort=False, fmt=".3f")


def rank_df(alts, **cols) -> pd.DataFrame:
    d = pd.DataFrame(cols, index=alts)
    return d


def ranking_result(alts, scores, rank, score_name="Score", fmt=".4f", higher_better=True):
    d = pd.DataFrame({score_name: scores, "Rang": rank}, index=alts).sort_values("Rang")
    st.dataframe(d.style.format({score_name: "{:" + fmt + "}"}).background_gradient(subset=[score_name], cmap=CMAP), width="stretch")
    st.success(f"Meilleure alternative : {d.index[0]}")
    return d


def line_chart(df, x, ys, title, ytitle="", xtitle=""):
    fig = go.Figure()
    for i, y in enumerate(ys):
        fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="lines", name=y, line=dict(color=PALETTE[i % len(PALETTE)], width=2.5)))
    fig.update_layout(title=title, xaxis_title=xtitle or x, yaxis_title=ytitle, height=380, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    return fig


def heatmap(M: pd.DataFrame, title: str, zmin=None, zmax=None, scale=None, fmt=".2f"):
    scale = scale or [[0, "#9a3412"], [0.5, "#f6f3ec"], [1, "#1f4e5f"]]
    fig = go.Figure(go.Heatmap(z=M.values, x=list(M.columns), y=list(M.index), zmin=zmin, zmax=zmax, colorscale=scale,
                               text=np.vectorize(lambda v: f"{v:{fmt}}")(M.values), texttemplate="%{text}", hoverongaps=False))
    fig.update_layout(title=title, height=380, margin=dict(l=10, r=10, t=50, b=10), yaxis_autorange="reversed")
    st.plotly_chart(fig, width="stretch")


def df_download(d: pd.DataFrame, name: str, label="Télécharger en CSV"):
    st.download_button(label, d.to_csv().encode("utf-8-sig"), file_name=name, mime="text/csv")


def weights_source(X, benefit, w_manual, key: str, allow=("Manuel", "Entropie", "CRITIC")):
    """Choix de l'origine des poids : saisis, entropie ou CRITIC (méthodes objectives du cours)."""
    from . import weights as W
    choice = st.radio("Origine des poids", allow, horizontal=True, key=f"{key}_wsrc")
    if choice == "Entropie":
        try:
            w = W.entropy_weights(X, benefit, "sum" if (X > 0).all() else "minmax")["w"]
        except ValueError:
            w = W.entropy_weights(X, benefit, "minmax")["w"]
    elif choice == "CRITIC":
        w = W.critic_weights(X, benefit)["w"]
    else:
        w = w_manual
    return choice, np.asarray(w, float)


def pick_dataset(key: str, options: dict, default: str):
    """Sélecteur de dataset : options = {libellé: callable → ds}."""
    names = list(options)
    sel = st.selectbox("Dataset", names, index=names.index(default) if default in names else 0, key=f"{key}_ds")
    ds = options[sel]()
    note_source(ds["source"] if "source" in ds else "créé")
    if ds.get("desc"): st.caption(ds["desc"])
    return ds
