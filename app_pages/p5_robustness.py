import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from mcdm_core import robustness as RB, datasets as D, ranking as R
from mcdm_core.ui import *

header('5 · Robustesse et comparaison des méthodes', 'Deux méthodes peuvent classer différemment les mêmes alternatives. On mesure leur accord par la corrélation de Spearman, puis la stabilité du premier face aux variations des poids.', 'cours, robustesse et comparaison des méthodes')
t1, t2, t3 = st.tabs(["Corrélation de Spearman (exemple du cours)", "Comparer les méthodes sur un dataset", "Sensibilité aux poids"])

with t1:
    course_box(r"""$\rho=1-\dfrac{6\sum d_i^2}{n(n^2-1)}$, $d_i$ = différence des rangs de l'alternative $i$. $\rho=+1$ : classements identiques ; $0$ : aucun lien ; $-1$ : inversés.""")
    st.success("Table des rangs de la diapositive (A1 à A6, trois méthodes). Modifiez un rang pour voir l'effet.")
    ranks = st.data_editor(D.RANGS_COURS, key="rk_course", width="stretch")
    if any(sorted(ranks[c]) != list(range(1, len(ranks) + 1)) for c in ranks): st.warning("Chaque colonne devrait être une permutation de 1…n.")
    M = RB.spearman_matrix(ranks)
    c1, c2 = st.columns(2)
    with c1: heatmap(M, "Matrice de corrélation de Spearman", -1, 1)
    n = len(ranks); crit_v = RB.spearman_critical(n)
    with c2:
        st.metric("Valeur critique à 5 % (bilatéral)", f"{crit_v:.3f}", help=f"n = {n}. Au-delà de cette valeur, la corrélation est significative.")
        sig = [(a, b) for i, a in enumerate(M.columns) for b in M.columns[i + 1:] if abs(M.loc[a, b]) >= crit_v]
        st.write("Paires significatives : " + (", ".join(f"**{a} – {b}**" for a, b in sig) if sig else "aucune"))
        st.caption("Avec n = 6, seule ρ ≥ 0,886 est significative à 5 % : uniquement TOPSIS – PROMETHEE II (cours).")
    top2 = {c: set(ranks.index[ranks[c] <= 2]) for c in ranks}
    if len({frozenset(s) for s in top2.values()}) == 1: st.info(f"Le choix de {' et '.join(sorted(list(top2.values())[0]))} est **robuste** aux trois méthodes ; le reste du classement dépend de la méthode.")
    diff = ranks.ne(ranks.iloc[:, 0], axis=0)
    st.dataframe(ranks.style.apply(lambda col: ["background-color:#ffe3e3" if x else "" for x in diff[col.name]], axis=0), width="stretch")
    st.caption("Cellules colorées : rang différent de celui de la première méthode.")

OPT = {"Électrification d'un village (14 x 9)": D.energie_village, "Portefeuille d'actions (16 x 10)": D.portefeuille, "Hôpital régional (12 x 8)": D.sites_hopital,
       "Modèles d'apprentissage (14 x 8)": D.ml_models, "Choix de machines (cours)": D.machines_waspas, "Choix d'une voiture (cours)": D.voitures_topsis}
with t2:
    ds = pick_dataset("rb", OPT, "Électrification d'un village (14 x 9)")
    X, ben, w, alts, crit = matrix_editor(ds, "rb")
    src, w = weights_source(X, ben, w, "rb")
    reg, dropped = RB.usable_methods(X, ben, w)
    for k, m in dropped.items(): st.warning(f"**{k}** écartée : {m}")
    names = st.multiselect("Méthodes à comparer", list(reg), default=list(reg))
    if len(names) < 2: st.warning("Choisissez au moins deux méthodes."); st.stop()
    cmp = RB.compare_methods(X, ben, w, names, reg); cmp.index = alts
    st.dataframe(cmp.style.background_gradient(cmap=CMAP_R, axis=None), width="stretch")
    M = RB.spearman_matrix(cmp)
    heatmap(M, "Accord entre méthodes (Spearman)", -1, 1)
    st.caption(f"ρ moyen entre méthodes : **{(M.values[np.triu_indices(len(M), 1)]).mean():.2f}** · n = {len(alts)} (valeur critique 5 % : {RB.spearman_critical(len(alts)):.3f})")
    st.info("**Meilleure alternative par méthode :** " + " · ".join(f"{m} → {alts[int(cmp[m].values.argmin())]}" for m in names))
    st.caption("Paramètres par défaut : PROMETHEE II à fonction linéaire (q = 5 %, p = 50 % de l'étendue) ; ELECTRE III q = 5 %, p = 20 %, v = 80 % de l'étendue.")

with t3:
    st.markdown("Faire varier les poids et vérifier la **stabilité du classement** (bonne pratique du cours).")
    ds = pick_dataset("rs", OPT, "Électrification d'un village (14 x 9)")
    X, ben, w, alts, crit = matrix_editor(ds, "rs")
    src, w = weights_source(X, ben, w, "rs")
    reg, dropped = RB.usable_methods(X, ben, w)
    for k, m in dropped.items(): st.warning(f"**{k}** écartée : {m}")
    mname = st.selectbox("Méthode", list(reg), index=list(reg).index("TOPSIS") if "TOPSIS" in reg else 0, key="rs_m"); fn = reg[mname]
    st.markdown("##### a) Variation d'un poids (les autres sont rééchelonnés)")
    j = st.selectbox("Critère", range(len(crit)), format_func=lambda i: crit[i], key="rs_j")
    sw = RB.sweep_weight(fn, X, ben, w, j)
    fig = go.Figure()
    for i, a in enumerate(alts):
        fig.add_trace(go.Scatter(x=sw["w"], y=sw[f"r{i}"], mode="lines", name=a, line=dict(color=PALETTE[i % len(PALETTE)], shape="hv")))
    fig.add_vline(x=float(w[j]), line_dash="dot", annotation_text="poids actuel")
    fig.update_layout(title=f"Rang de chaque alternative selon le poids de « {crit[j]} »", xaxis_title="poids", yaxis_title="rang (1 = meilleur)", yaxis_autorange="reversed", height=420)
    st.plotly_chart(fig, width="stretch")
    first = sw[[f"r{i}" for i in range(len(alts))]].values.argmin(1)
    changes = np.where(np.diff(first) != 0)[0]
    st.caption("Le n°1 change aux poids : " + (", ".join(f"≈ {sw['w'].iloc[c+1]:.2f} ({alts[first[c]]} → {alts[first[c+1]]})" for c in changes) if len(changes) else "jamais sur [0 ; 1] → classement de tête très stable pour ce critère."))
    st.markdown("##### b) Simulation de Monte-Carlo (poids tirés d'une loi de Dirichlet autour des poids actuels)")
    c1, c2 = st.columns(2)
    conc = c1.slider("Concentration (grande = perturbation faible)", 5, 200, 40); N = c2.select_slider("Nombre de tirages", [100, 300, 500, 1000], 300)
    with st.spinner("Simulation…"):
        mc = RB.monte_carlo(fn, X, ben, w, N, conc)
    A = pd.DataFrame(mc["acceptability"], index=alts, columns=[f"rang {k}" for k in range(1, len(alts) + 1)])
    heatmap(A, "Indice d'acceptabilité des rangs (fréquence)", 0, 1, SEQ)
    m1, m2 = st.columns(2); m1.metric("ρ de Spearman moyen vs classement de base", f"{mc['rho_mean']:.3f}"); m2.metric("ρ minimum observé", f"{mc['rho_min']:.3f}")
    best_i = int(np.argmin(mc["base_rank"])); st.info(f"**{alts[best_i]}** (n°1 de base) reste n°1 dans **{mc['top1'][best_i]:.0%}** des tirages.")
