import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from mcdm_core import ranking as R, datasets as D
from mcdm_core.ui import *

header('3 · Classement additif', 'Un score global agrège les critères ; un bon score peut en compenser un mauvais.', 'cours, classement des alternatives, méthodes additives')
ALL = {"Électrification d'un village (14 x 9)": D.energie_village, "Portefeuille d'actions (16 x 10)": D.portefeuille, "Hôpital régional (12 x 8)": D.sites_hopital,
       "Modèles d'apprentissage (14 x 8)": D.ml_models, "Choix de machines (cours, WASPAS)": D.machines_waspas, "Choix d'une voiture (cours, TOPSIS)": D.voitures_topsis,
       "Achat d'une machine (cours, VIKOR)": D.machines_vikor}
t_w, t_t, t_v = st.tabs(["WSM · WPM · WASPAS", "TOPSIS", "VIKOR"])


def setup(key, default):
    ds = pick_dataset(key, ALL, default)
    X, ben, w, alts, crit = matrix_editor(ds, key)
    src, w = weights_source(X, ben, w, key)
    st.caption("Poids utilisés : " + ", ".join(f"{c.split(' ')[0]}={v:.3f}" for c, v in zip(crit, w)))
    return X, ben, w, alts, crit


with t_w:
    course_box(r"""
**Solutions idéales** $x_j^+$ (meilleure valeur), **normalisation** : positif $r_{ij}=x_{ij}/x_j^+$ ; négatif $r_{ij}=x_j^+/x_{ij}$.
- **WSM** : $Q_i^{(1)}=\sum_j w_j r_{ij}$  - **WPM** : $Q_i^{(2)}=\prod_j r_{ij}^{w_j}$
- **WASPAS** : $Q_i=\lambda Q_i^{(1)}+(1-\lambda)Q_i^{(2)}$, $\lambda\in[0,1]$ ($\lambda=0{,}5$ : moyenne des deux).""")
    X, ben, w, alts, crit = setup("w", "Électrification d'un village (14 x 9)")
    lam = st.slider("λ (WASPAS)", 0.0, 1.0, 0.5, 0.05)
    try:
        r = R.waspas(X, ben, w, lam)
    except ValueError as err:
        st.error(str(err)); st.stop()
    xp, xm = R.ideal_points(X, ben)
    steps({"Solutions idéales x⁺ / anti-idéales x⁻": pd.DataFrame([xp, xm], index=["x⁺", "x⁻"], columns=crit), "Matrice normalisée r_ij": pd.DataFrame(r["R"], index=alts, columns=crit)})
    out = pd.DataFrame({"Q(1) WSM": r["Q1"], "Q(2) WPM": r["Q2"], "Q WASPAS": r["Q"], "Rang WSM": r["rank_wsm"], "Rang WPM": r["rank_wpm"], "Rang WASPAS": r["rank"]}, index=alts).sort_values("Rang WASPAS")
    st.dataframe(out.style.format({"Q(1) WSM": "{:.4f}", "Q(2) WPM": "{:.4f}", "Q WASPAS": "{:.4f}"}), width="stretch")
    st.success(f"**Meilleure alternative (WASPAS, λ={lam:.2f}) : {out.index[0]}**")
    st.bar_chart(out[["Q(1) WSM", "Q(2) WPM", "Q WASPAS"]], height=320)

with t_t:
    course_box(r"""
1. $r_{ij}=x_{ij}/\sqrt{\sum_i x_{ij}^2}$  2. $v_{ij}=w_j r_{ij}$  3. $I^+$ / $I^-$ : meilleures / pires valeurs de $v_{\cdot j}$ (selon le type du critère)
4. $S_i^\pm=\sqrt{\sum_j (v_{ij}-I_j^\pm)^2}$  5. $RC_i^*=\dfrac{S_i^-}{S_i^++S_i^-}\in[0,1]$, la meilleure alternative a le $RC^*$ le plus élevé.""")
    X, ben, w, alts, crit = setup("t", "Portefeuille d'actions (16 x 10)")
    t = R.topsis(X, ben, w)
    steps({"Matrice normalisée r_ij": pd.DataFrame(t["R"], index=alts, columns=crit), "Matrice pondérée v_ij": pd.DataFrame(t["V"], index=alts, columns=crit),
           "Solutions idéales I⁺ / I⁻": pd.DataFrame([t["v_plus"], t["v_minus"]], index=["I⁺", "I⁻"], columns=crit)})
    d = pd.DataFrame({"S⁺": t["S_plus"], "S⁻": t["S_minus"], "RC*": t["RC"], "Rang": t["rank"]}, index=alts).sort_values("Rang")
    st.dataframe(d.style.format({"S⁺": "{:.4f}", "S⁻": "{:.4f}", "RC*": "{:.4f}"}).background_gradient(subset=["RC*"], cmap=CMAP), width="stretch")
    st.success(f"**Meilleure alternative : {d.index[0]}**")
    fig = go.Figure(go.Scatter(x=d["S⁺"], y=d["S⁻"], mode="markers+text", text=d.index, textposition="top center", marker=dict(size=14, color=PALETTE[0])))
    fig.update_layout(title="Distances aux solutions idéales (idéal = en bas à droite)", xaxis_title="S⁺ (distance à l'idéal positif)", yaxis_title="S⁻ (distance à l'idéal négatif)", height=380)
    st.plotly_chart(fig, width="stretch")

with t_v:
    course_box(r"""
$S_i=\sum_j w_j\frac{x_j^*-x_{ij}}{x_j^*-x_j^-}$ (utilité de groupe), $R_i=\max_j[\cdot]$ (regret individuel), $Q_i=v\frac{S_i-S^*}{S^--S^*}+(1-v)\frac{R_i-R^*}{R^--R^*}$.
Plus $S,R,Q$ sont **petits**, meilleure est l'alternative. Solution de compromis $A'$ (meilleur $Q$) si :
**C1 avantage acceptable** $Q(A'')-Q(A')\ge\frac1{m-1}$ et **C2 stabilité** : $A'$ aussi meilleure en $S$ ou en $R$. Sinon : $\{A',A''\}$ (C2 non vérifiée) ou $\{A',\dots,A^{(k)}\}$ (C1 non vérifiée).""")
    X, ben, w, alts, crit = setup("v", "Hôpital régional (12 x 8)")
    v = st.slider("v (stratégie d'utilité de groupe maximale)", 0.0, 1.0, 0.5, 0.05)
    k = R.vikor(X, ben, w, v)
    steps({"Termes w_j (x*−x_ij)/(x*−x⁻)": pd.DataFrame(k["T"], index=alts, columns=crit)})
    d = pd.DataFrame({"S": k["S"], "R": k["R"], "Q": k["Q"], "Rang S": k["rank_S"], "Rang R": k["rank_R"], "Rang Q": k["rank_Q"]}, index=alts).sort_values("Rang Q")
    st.dataframe(d.style.format({"S": "{:.4f}", "R": "{:.4f}", "Q": "{:.4f}"}), width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.metric("Seuil DQ = 1/(m−1)", f"{k['DQ']:.3f}"); c2.metric("C1 avantage acceptable", "oui" if k["C1"] else "non"); c3.metric("C2 stabilité acceptable", "oui" if k["C2"] else "non")
    st.success(f"**Solution de compromis :** {', '.join(alts[i] for i in k['compromise'])}, {k['compromise_kind']}")
    st.bar_chart(d[["S", "R", "Q"]], height=300)
