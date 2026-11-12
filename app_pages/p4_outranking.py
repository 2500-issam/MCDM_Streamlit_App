import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from mcdm_core import outranking as O, datasets as D
from mcdm_core.ui import *

header('4 · Surclassement', 'Les alternatives sont comparées deux à deux, sans compensation complète entre critères.', 'cours, classement des alternatives, méthodes de surclassement')
st.warning("Dans le support de cours, PROMETHEE et ELECTRE III n'ont qu'une page de titre. Les calculs ci-dessous suivent les formulations standard de la littérature "
           "(Brans & Vincke ; Roy), pas des formules propres au cours.")
OPT = {"Électrification d'un village (14 x 9)": D.energie_village, "Hôpital régional (12 x 8)": D.sites_hopital, "Portefeuille d'actions (16 x 10)": D.portefeuille,
       "Choix de machines (cours)": D.machines_waspas}
ds = pick_dataset("o", OPT, "Électrification d'un village (14 x 9)")
X, ben, w, alts, crit = matrix_editor(ds, "o")
n = len(crit); rng = np.ptp(X, axis=0); rng = np.where(rng == 0, 1, rng)
short = [c.split(" ")[0] if len(c) > 14 else c for c in crit]
tp, te = st.tabs(["PROMETHEE I / II", "ELECTRE III"])

with tp:
    course_box(r"""
Pour chaque critère, une **fonction de préférence** $P_j(d)$ transforme l'écart $d=g_j(a)-g_j(b)$ en degré de préférence ∈ [0,1] (usuelle, U, V, paliers, linéaire, gaussienne).
Indice agrégé $\pi(a,b)=\sum_j w_jP_j(d_j(a,b))$ ; flux sortant $\phi^+(a)=\frac1{m-1}\sum_b\pi(a,b)$, entrant $\phi^-(a)=\frac1{m-1}\sum_b\pi(b,a)$, net $\phi=\phi^+-\phi^-$.
**PROMETHEE II** : classement complet par $\phi$. **PROMETHEE I** : ordre partiel (intersection de $\phi^+$ et $\phi^-$ → préférence P, indifférence I, incomparabilité R).""")
    base = pd.DataFrame({"Critère": crit, "Fonction": ["linear"] * n, "q (indiff.)": 0.05 * rng, "p (préf.)": 0.5 * rng, "σ (gauss)": 0.3 * rng})
    pf = st.data_editor(base, key="pf", hide_index=True, width="stretch", disabled=["Critère"],
                        column_config={"Fonction": st.column_config.SelectboxColumn(options=list(O.PREF_TYPES), required=True)})
    with st.expander("Légende des fonctions"): st.write(O.PREF_TYPES)
    pr = O.promethee(X, ben, w, list(pf["Fonction"]), q=pf["q (indiff.)"].to_numpy(float), p=pf["p (préf.)"].to_numpy(float), s=pf["σ (gauss)"].to_numpy(float))
    steps({"Indices de préférence agrégés π(a,b)": pd.DataFrame(pr["pi"], index=alts, columns=alts)}, title="Matrice π")
    d = pd.DataFrame({"φ⁺": pr["phi_plus"], "φ⁻": pr["phi_minus"], "φ (net)": pr["phi"], "Rang PROMETHEE II": pr["rank"]}, index=alts).sort_values("Rang PROMETHEE II")
    st.dataframe(d.style.format({"φ⁺": "{:.4f}", "φ⁻": "{:.4f}", "φ (net)": "{:.4f}"}).background_gradient(subset=["φ (net)"], cmap=CMAP_DIV), width="stretch")
    st.success(f"**PROMETHEE II, meilleure alternative : {d.index[0]}**")
    rel = pd.DataFrame(pr["relation"], index=alts, columns=alts).replace({"P-": "◄", "P": "►", "=": "·", "I": "I", "R": "R"})
    st.markdown("**PROMETHEE I**, ligne *a* vs colonne *b* : ► *a* surclasse *b* · ◄ *b* surclasse *a* · I indifférents · R incomparables")
    st.dataframe(rel, width="stretch")
    fig = go.Figure(go.Scatter(x=pr["phi_minus"], y=pr["phi_plus"], mode="markers+text", text=alts, textposition="top center", marker=dict(size=13, color=PALETTE[1])))
    fig.update_layout(title="Plan (φ⁻, φ⁺) : le meilleur est en haut à gauche", xaxis_title="φ⁻ (flux entrant)", yaxis_title="φ⁺ (flux sortant)", height=400)
    st.plotly_chart(fig, width="stretch")

with te:
    course_box(r"""
Seuils par critère : **indifférence** $q_j$, **préférence** $p_j$, **veto** $v_j$. Concordance partielle $c_j(a,b)=1$ si $g_j(b)-g_j(a)\le q_j$, $0$ si $\ge p_j$, linéaire entre les deux ;
$C(a,b)=\sum_jw_jc_j$. Discordance $d_j(a,b)=0$ si $\le p_j$, $1$ si $\ge v_j$ (**veto**), linéaire entre les deux.
**Crédibilité** $\sigma(a,b)=C(a,b)\prod_{j\in F}\frac{1-d_j}{1-C}$ avec $F=\{j: d_j>C\}$. Deux **distillations** (descendante / ascendante) donnent deux préordres dont l'intersection est le préordre partiel final.""")
    use_veto = st.checkbox("Activer le veto (non-compensation)", True)
    base = pd.DataFrame({"Critère": crit, "q": 0.05 * rng, "p": 0.2 * rng, "v": 0.8 * rng})
    th = st.data_editor(base, key="th", hide_index=True, width="stretch", disabled=["Critère"])
    q, p, v = th["q"].to_numpy(float), th["p"].to_numpy(float), th["v"].to_numpy(float)
    if (p < q).any() or (v < p).any(): st.error("Il faut q ≤ p ≤ v pour chaque critère."); st.stop()
    v_eff = v if use_veto else np.full(n, 1e12)
    e = O.electre3(X, ben, w, q, p, v_eff)
    steps({"Concordance globale C(a,b)": pd.DataFrame(e["C"], index=alts, columns=alts), "Crédibilité σ(a,b)": pd.DataFrame(e["sigma"], index=alts, columns=alts)}, title="Matrices C et σ")
    d = pd.DataFrame({"Rang distillation ↓": e["rank_desc"], "Rang distillation ↑": e["rank_asc"], "Rang moyen": e["mean_rank"], "Classement final": e["rank"]}, index=alts).sort_values(["Classement final", "Rang moyen"])
    st.dataframe(d, width="stretch")
    st.success(f"**ELECTRE III, meilleure alternative : {d.index[0]}**")
    rel = pd.DataFrame(e["relation"], index=alts, columns=alts).replace({"P-": "◄", "P": "►", "=": "·"})
    st.markdown("**Préordre partiel final**, ► ligne surclasse colonne · ◄ colonne surclasse ligne · I indifférents · R incomparables")
    st.dataframe(rel, width="stretch")
    if use_veto:
        e0 = O.electre3(X, ben, w, q, p, np.full(n, 1e12))
        diff = [alts[i] for i in range(len(alts)) if e0["rank"][i] != e["rank"][i]]
        st.info("**Effet du veto** (comparaison avec veto désactivé) : " + (f"le rang change pour {', '.join(diff)}." if diff else "aucun changement de rang sur ces données, essayez de rapprocher les seuils v des seuils p."))
