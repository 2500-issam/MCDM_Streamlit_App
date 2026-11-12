import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from mcdm_core import fuzzy as F, datasets as D
from mcdm_core.ui import *

header('8 · MCDM flou', 'Des jugements verbaux (bon, passable) sont convertis en nombres flous, agrégés, puis classés.', 'cours, analyse multicritère floue, TOPSIS flou, TOPSIS flou intuitionniste')
t1, t2, t3 = st.tabs(["Une alternative (TFN)", "TOPSIS flou, recrutement (cours)", "TOPSIS flou intuitionniste, traitement"])
fmt3 = lambda t: "(" + ", ".join(f"{v:.2f}" for v in t) + ")"

with t1:
    course_box(r"""Chaque évaluation est un **nombre triangulaire flou** $\tilde x=(a,b,c)$ : $a$ pessimiste, $b$ la plus probable, $c$ optimiste.
1. **Normalisation** $(a,b,c)\div c_{max}$ · 2. **Agrégation** $\tilde S=\sum w_i\tilde x_i$ · 3. **Défuzzification** $S=\frac{a+b+c}3$.
La largeur $c-a$ de $\tilde S$ mesure l'incertitude, à surveiller **en plus** du score.""")
    fcase = st.selectbox("Cas", ["ROA et liquidité (cours)", "Projet d'investissement (8 critères)"], key="fs_case")
    if fcase.startswith("ROA"):
        ex = D.FUZZY_SINGLE; note_source("cours", "Poids supposés 0,6 et 0,4.")
    else:
        ex = dict(criteria=["Rentabilité", "Risque maîtrisé", "Faisabilité technique", "Délai de réalisation", "Impact environnemental", "Impact social", "Alignement stratégique", "Flexibilité"],
                  tfn=[(6, 7.5, 9), (4, 6, 8), (7, 8, 9), (3, 5, 7), (5, 6.5, 8), (4, 5.5, 7), (7, 8.5, 10), (2, 4, 6)], w=[.20, .15, .15, .10, .10, .10, .12, .08]); note_source("créé", "Notes sur 10 sous forme de nombres triangulaires.")
    base = pd.DataFrame({"Critère": ex["criteria"], "a": [t[0] for t in ex["tfn"]], "b": [t[1] for t in ex["tfn"]], "c": [t[2] for t in ex["tfn"]], "Poids": ex["w"]})
    ed = st.data_editor(base, key=f"fs_{fcase[:3]}", hide_index=True, width="stretch", disabled=["Critère"])
    if not ((ed["a"] <= ed["b"]) & (ed["b"] <= ed["c"])).all(): st.error("a ≤ b ≤ c requis."); st.stop()
    w = ed["Poids"].to_numpy(float); w = w / w.sum()
    r = F.fuzzy_single(ed[["a", "b", "c"]].to_numpy(float), w)
    steps({"Matrice normalisée (÷ c_max)": pd.DataFrame(r["R"], index=ed["Critère"], columns=["a", "b", "c"])})
    c1, c2, c3 = st.columns(3); c1.metric("S̃ = Σ wᵢ·x̃ᵢ", fmt3(r["S"])); c2.metric("Score défuzzifié S", f"{r['score']:.3f}"); c3.metric("Largeur (incertitude)", f"{r['width']:.3f}")
    xs = np.linspace(0, 1.05, 400); fig = go.Figure()
    for i in range(len(ed)): fig.add_trace(go.Scatter(x=xs, y=F.mf_tri(xs, *r["R"][i]), name=ed["Critère"][i], line=dict(dash="dot", color=PALETTE[(i + 1) % len(PALETTE)], width=1.2)))
    fig.add_trace(go.Scatter(x=xs, y=F.mf_tri(xs, *r["S"]), name="S̃ agrégé", line=dict(width=4, color=PALETTE[0])))
    fig.update_layout(height=340, xaxis_title="valeur normalisée", yaxis_title="μ", margin=dict(t=20)); st.plotly_chart(fig, width="stretch")

with t2:
    course_box(r"""**Chen (2000)**, 9 étapes : (1) chaque décideur évalue chaque alternative par une variable linguistique (TFN, Tableau 1) ; (2) évalue l'importance de chaque critère ;
(3) **évaluations collectives** $\tilde x_{ij}=\frac1K(\tilde x^1_{ij}\oplus\dots\oplus\tilde x^K_{ij})$ ; (4) **normalisation** (positif : $\div c_j^*=\max c_{ij}$ ; négatif : $a_j^-/(c,b,a)$) ; (5) **pondération** $\tilde v_{ij}=\tilde r_{ij}\otimes\tilde w_j$ ;
(6) idéaux $A^*=(1,1,1)$, $A^-=(0,0,0)$ ; (7) distances (méthode des sommets) $d(\tilde v,\tilde v^*)=\sqrt{\frac13\sum(v^l-1)^2}$ ; (8) $CC_i=\frac{D_i^-}{D_i^*+D_i^-}$ ; (9) classement décroissant.""")
    rcase = st.selectbox("Cas", ["Recrutement de l'exercice (3 candidats, 5 critères, 3 décideurs)", "Recrutement élargi (8 candidats, 7 critères, 4 décideurs)"], key="ft_case")
    rk = "c" if rcase.startswith("Recrutement de") else "l"
    ex = D.RECRUTEMENT if rk == "c" else D.recrutement_large(); K = len(ex["dms"])
    note_source("cours" if rk == "c" else "créé", "" if rk == "c" else "Niveaux linguistiques tirés au hasard autour d'une valeur latente (graine 31).")
    with st.expander("Tableau 1, échelles linguistiques (Chen 2000)"):
        c1, c2 = st.columns(2)
        c1.dataframe(pd.DataFrame(F.ALT_SCALE, index=["a", "b", "c"]).T, width="stretch"); c2.dataframe(pd.DataFrame(F.W_SCALE, index=["a", "b", "c"]).T, width="stretch")
    cfg = {c: st.column_config.SelectboxColumn(options=list(F.ALT_SCALE), required=True) for c in ex["crit"]}
    rows = [f"{a} · {dm}" for a in ex["alts"] for dm in ex["dms"]]
    t2d = pd.DataFrame([ex["ratings"][a][k] for a in ex["alts"] for k in range(K)], columns=ex["crit"], index=rows)
    st.markdown("Tableau 2 : évaluations linguistiques")
    t2e = st.data_editor(t2d, key=f"ft2_{rk}", width="stretch", column_config=cfg)
    wcfg = {c: st.column_config.SelectboxColumn(options=list(F.W_SCALE), required=True) for c in ex["crit"]}
    st.markdown("Tableau 3 : importance des critères")
    t3e = st.data_editor(pd.DataFrame(ex["weights"], columns=ex["crit"], index=ex["dms"]), key=f"ft3_{rk}", width="stretch", column_config=wcfg)
    cost = st.multiselect("Critères de coût (par défaut : tous de type bénéfice)", ex["crit"], key=f"ft_cost_{rk}")
    ben = np.array([c not in cost for c in ex["crit"]])
    ratings = np.array([[[F.ALT_SCALE[t2e.loc[f"{a} · {dm}", c]] for c in ex["crit"]] for a in ex["alts"]] for dm in ex["dms"]], float)
    weights = np.array([[F.W_SCALE[t3e.loc[dm, c]] for c in ex["crit"]] for dm in ex["dms"]], float)
    r = F.fuzzy_topsis(ratings, weights, ben)
    mk = lambda M, idx, cols: pd.DataFrame([[fmt3(M[i, j]) for j in range(M.shape[1])] for i in range(M.shape[0])], index=idx, columns=cols)
    st.markdown("#### Réponses aux questions de l'exercice")
    with st.expander("1) Matrice de décision collective", expanded=True): st.dataframe(mk(r["X"], ex["alts"], ex["crit"]), width="stretch")
    with st.expander("2) Matrice de décision collective normalisée"): st.dataframe(mk(r["R"], ex["alts"], ex["crit"]), width="stretch")
    with st.expander("3) Poids collectifs des critères"): st.dataframe(pd.DataFrame([[fmt3(x) for x in r["W"]]], columns=ex["crit"], index=["w̃"]), width="stretch")
    with st.expander("4) Matrice normalisée pondérée"): st.dataframe(mk(r["V"], ex["alts"], ex["crit"]), width="stretch")
    st.markdown("**5) Résolution**")
    d = pd.DataFrame({"D*ᵢ (idéal +)": r["D_plus"], "D⁻ᵢ (idéal −)": r["D_minus"], "CCᵢ": r["CC"], "Rang": r["rank"]}, index=ex["alts"]).sort_values("Rang")
    st.dataframe(d.style.format({"D*ᵢ (idéal +)": "{:.4f}", "D⁻ᵢ (idéal −)": "{:.4f}", "CCᵢ": "{:.4f}"}).background_gradient(subset=["CCᵢ"], cmap=CMAP), width="stretch")
    st.success(f"**Candidat retenu : {d.index[0]}**"); bar(r["CC"], ex["alts"], "Coefficient de proximité CC", "CC")

with t3:
    course_box(r"""Les décideurs donnent des **IFN** $\alpha^k_{ij}=(\mu,\nu)$. (1-2) matrices par décideur ; (3) agrégation **IFWA** pondérée par $\sigma_k$ ; (4) poids des critères agrégés aussi par IFWA ;
(5) matrice pondérée $a_{ij}\otimes w_j=(\mu_a\mu_w,\ \nu_a+\nu_w-\nu_a\nu_w)$ ; (6) $I^+=(\max\mu,\min\nu)$ pour un critère positif (inverse pour un négatif), $I^-$ symétrique ;
(7) $S_i^\pm=\frac1{2m}\sum_j[(\Delta\mu)^2+(\Delta\nu)^2+(\Delta\pi)^2]$ ; (8) $C_i=\frac{S_i^-}{S_i^-+S_i^+}$, la meilleure alternative a le $C_i$ le plus élevé.""")
    icase = st.selectbox("Cas", ["Traitement médical (4 traitements, 5 critères, 3 décideurs)", "Traitement médical élargi (10 traitements, 8 critères, 4 décideurs)"], key="if_case")
    ik = "c" if icase.startswith("Traitement médical (") else "l"
    ex = D.traitement_medical() if ik == "c" else D.traitement_large(); K = len(ex["dms"]); note_source("créé"); st.caption(ex.get("desc", ""))
    with st.expander("Échelles linguistiques utilisées (μ, ν)"):
        c1, c2 = st.columns(2)
        c1.dataframe(pd.DataFrame({"Niveau": D.IF_LABELS_ALT, "(μ, ν)": {k: str(v) for k, v in D.IF_SCALE_ALT.items()}}), width="stretch")
        c2.dataframe(pd.DataFrame({"Importance": D.IF_LABELS_W, "(μ, ν)": {k: str(v) for k, v in D.IF_SCALE_W.items()}}), width="stretch")
    rows = [f"{a} · {dm}" for a in ex["alts"] for dm in ex["dms"]]
    cfg = {c: st.column_config.SelectboxColumn(options=list(D.IF_SCALE_ALT), required=True) for c in ex["crit"]}
    t2d = pd.DataFrame([ex["ratings"][a][k] for a in ex["alts"] for k in range(K)], columns=ex["crit"], index=rows)
    t2e = st.data_editor(t2d, key=f"if2_{ik}", width="stretch", column_config=cfg)
    wcfg = {c: st.column_config.SelectboxColumn(options=list(D.IF_SCALE_W), required=True) for c in ex["crit"]}
    t3e = st.data_editor(pd.DataFrame(ex["weights"], columns=ex["crit"], index=ex["dms"]), key=f"if3_{ik}", width="stretch", column_config=wcfg)
    st.markdown("Poids des décideurs σₖ (somme égale à 1)")
    cols = st.columns(K); sg = np.array([cols[i].number_input(ex["dms"][i], 0.0, 1.0, ex["sigma"][i], 0.05, key=f"sg_{ik}_{i}") for i in range(K)])
    if sg.sum() <= 0: st.error("Σσ > 0 requis."); st.stop()
    sg = sg / sg.sum(); st.caption(f"σ normalisés : {np.round(sg, 3)}")
    ratings = np.array([[[D.IF_SCALE_ALT[t2e.loc[f"{a} · {dm}", c]] for c in ex["crit"]] for a in ex["alts"]] for dm in ex["dms"]], float)
    weights = np.array([[D.IF_SCALE_W[t3e.loc[dm, c]] for c in ex["crit"]] for dm in ex["dms"]], float)
    r = F.if_topsis(ratings, weights, sg, np.array(ex["benefit"], bool))
    mk = lambda M, idx, cols: pd.DataFrame([[f"({M[i,j,0]:.3f}; {M[i,j,1]:.3f})" for j in range(M.shape[1])] for i in range(M.shape[0])], index=idx, columns=cols)
    steps({"Matrice agrégée D (IFWA)": mk(r["A"], ex["alts"], ex["crit"]), "Poids agrégés W (IFWA)": mk(r["W"][None], ["w"], ex["crit"]),
           "Matrice pondérée D' = D ⊗ W": mk(r["Aw"], ex["alts"], ex["crit"]),
           "Solutions idéales I⁺ / I⁻": mk(np.stack([r["I_plus"], r["I_minus"]]), ["I⁺", "I⁻"], ex["crit"])}, title="Étapes 3 à 6")
    d = pd.DataFrame({"S⁺ᵢ": r["S_plus"], "S⁻ᵢ": r["S_minus"], "Cᵢ": r["C"], "Rang": r["rank"]}, index=ex["alts"]).sort_values("Rang")
    st.dataframe(d.style.format({"S⁺ᵢ": "{:.4f}", "S⁻ᵢ": "{:.4f}", "Cᵢ": "{:.4f}"}).background_gradient(subset=["Cᵢ"], cmap=CMAP), width="stretch")
    st.success(f"**Traitement recommandé : {d.index[0]}**"); bar(r["C"], ex["alts"], "Coefficient de proximité Cᵢ", "C")
