import numpy as np, pandas as pd, streamlit as st
from mcdm_core import weights as W, datasets as D
from mcdm_core.ui import *

header('2 · Pondération objective', 'Les poids sont calculés sur la matrice de décision : plus un critère sépare les alternatives, plus son poids est élevé.', 'cours, méthodes objectives (entropie, CRITIC)')
OPTS = {"Modèles d'apprentissage (14 x 8)": D.ml_models, "Électrification d'un village (14 x 9)": D.energie_village, "Portefeuille d'actions (16 x 10)": D.portefeuille,
        "Hôpital régional (12 x 8)": D.sites_hopital, "Machines de découpe (cours, exercice CRITIC)": D.machines_critic}
ds = pick_dataset("obj", OPTS, "Modèles d'apprentissage (14 x 8)")
X, benefit, w_subj, alts, crit = matrix_editor(ds, "obj", with_weights=True, default_w=None)
short = [c.split(" ")[0] if len(c) > 14 else c for c in crit]
t1, t2, t3, t4 = st.tabs(["Entropie", "CRITIC", "Comparaison", "Poids hybrides"])

with t1:
    course_box(r"""
$p_{ij}=\dfrac{d_{ij}}{\sum_i d_{ij}}$, $\;E_j=-k\sum_i p_{ij}\ln p_{ij}$, $k=\dfrac1{\ln m}$, $\;w_j=\dfrac{1-E_j}{n-\sum_k E_k}$.
Entropie élevée (≈1) ⇒ valeurs uniformes ⇒ critère peu discriminant ⇒ poids faible.""")
    norm = st.radio("Normalisation préalable", ["sum", "minmax"], horizontal=True, key="ent_n",
                    format_func=lambda s: "x/Σx (coût : 1/x)" if s == "sum" else "min-max du cours (coût inversé)")
    try:
        e = W.entropy_weights(X, benefit, norm)
    except ValueError as err:
        st.error(str(err)); st.stop()
    steps({"Matrice normalisée d_ij": pd.DataFrame(e["D"], index=alts, columns=crit), "Proportions p_ij": pd.DataFrame(e["P"], index=alts, columns=crit),
           "Entropie Eⱼ, diversification 1−Eⱼ, poids": pd.DataFrame({"Eⱼ": e["E"], "1−Eⱼ": e["diversification"], "wⱼ": e["w"]}, index=crit).T})
    weights_chart(e["w"], crit, "Poids par entropie")
    st.caption("Le résultat dépend de la normalisation choisie (le cours ne la précise pas), comparez les deux options.")

with t2:
    course_box(r"""
1. Normalisation min-max : positif $\frac{x_{ij}-\min}{\max-\min}$ ; négatif $\frac{\max-x_{ij}}{\max-\min}$.
2. Corrélation de Pearson $\rho_{jk}$ entre colonnes. 3. $\sigma_j$ (écart-type, dénominateur $m-1$) puis $C_j=\sigma_j\sum_k(1-|\rho_{jk}|)$.
4. $w_j=C_j/\sum C_j$. Un critère **dispersé** et **peu corrélé** aux autres est important.""")
    c = W.critic_weights(X, benefit)
    steps({"Matrice normalisée r_ij": pd.DataFrame(c["R"], index=alts, columns=crit), "Corrélations ρ_jk": pd.DataFrame(c["rho"], index=crit, columns=crit),
           "σⱼ, Σ(1−|ρ|), Cⱼ, wⱼ": pd.DataFrame({"σⱼ": c["sigma"], "Σ(1−|ρ_jk|)": c["independence"], "Cⱼ": c["C"], "wⱼ": c["w"]}, index=crit).T})
    heatmap(pd.DataFrame(c["rho"], index=short, columns=short), "Corrélation des critères (normalisés)", -1, 1)
    weights_chart(c["w"], crit, "Poids CRITIC")

with t3:
    ent = W.entropy_weights(X, benefit, "minmax" if (X <= 0).any() else "sum")["w"]; cri = W.critic_weights(X, benefit)["w"]
    cmp = pd.DataFrame({"Entropie": ent, "CRITIC": cri, "Subjectifs (tableau)": w_subj}, index=short)
    st.bar_chart(cmp, height=360)
    st.dataframe(cmp.style.format("{:.3f}"), width="stretch")
    st.info("**Limites (cours)**, objectives : ignorent l'importance stratégique réelle ; subjectives : dépendent de l'expert. "
            "CRITIC tient compte de la **corrélation** entre critères, l'entropie seulement de la dispersion.")

with t4:
    st.markdown("**N.B. du cours :** en pratique on combine souvent subjectif + objectif pour équilibrer expertise et données.")
    a = st.slider("Part du subjectif α", 0.0, 1.0, 0.5, 0.05)
    obj = st.radio("Poids objectif", ["CRITIC", "Entropie"], horizontal=True)
    wo = cri if obj == "CRITIC" else ent
    wh = W.combine_weights([w_subj, wo], alpha=a)
    st.dataframe(pd.DataFrame({"Subjectif": w_subj, obj: wo, "Hybride": wh}, index=short).style.format("{:.3f}"), width="stretch")
    st.bar_chart(pd.DataFrame({"Subjectif": w_subj, obj: wo, "Hybride": wh}, index=short), height=320)
