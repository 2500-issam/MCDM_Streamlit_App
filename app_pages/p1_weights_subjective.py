import numpy as np, pandas as pd, streamlit as st
from mcdm_core import weights as W, datasets as D
from mcdm_core.ui import *

header("1 · Pondération subjective : AHP et BWM", "Les poids traduisent le jugement d'un expert : comparaisons par paires pour l'AHP, deux séries de notes pour le BWM.",
       "méthodes subjectives (AHP, BWM)")
tab_ahp, tab_bwm = st.tabs(["AHP", "BWM"])


def pair_editor(names, upper, key):
    """Édite le triangle supérieur ; le triangle inférieur est l'inverse (réciprocité)."""
    df = pd.DataFrame(np.array(upper, float), index=names, columns=names)
    ed = st.data_editor(df, key=key, width="stretch")
    A = W.ahp_reciprocal(ed.to_numpy(float))
    if (A <= 0).any() or np.isnan(A).any():
        st.error("Toutes les valeurs du triangle supérieur doivent être strictement positives."); st.stop()
    return A


with tab_ahp:
    course_box(r"""
1. Hiérarchie : objectif, critères, alternatives.
2. Poids des critères : matrice de comparaison par paires $A=(\alpha_{ij})$, $\alpha_{ji}=1/\alpha_{ij}$, échelle de Saaty de 1 à 9.
   Approximation du vecteur propre : somme de chaque colonne $s_j$, division de chaque terme par $s_j$, moyenne de chaque ligne, d'où $w_i$.
3. Consistance : $\lambda_{max}=\sum s_i w_i$, $CI=\frac{\lambda_{max}-n}{n-1}$, $CR=\frac{CI}{RI}$. Les jugements sont consistants si $CR<0{,}1$.
4. Priorités locales (une matrice de comparaison des alternatives par critère), puis priorités globales $\sum_j w_j w_{ji}$.
5. Analyse de sensibilité sur les poids.""")
    case = st.selectbox("Cas", ["Achat d'une voiture (cours)", "Implantation d'un site industriel (8 critères, 6 villes)"], key="ahp_case")
    is_course = case.startswith("Achat"); ck = "c" if is_course else "s"
    ex = D.AHP_VOITURE if is_course else D.ahp_site()
    note_source("cours" if is_course else "créé", "" if is_course else "Matrices obtenues en arrondissant des rapports de scores latents à l'échelle de Saaty (graine 3).")
    method = st.radio("Calcul des poids", ["approx", "eigen"], format_func=lambda m: "Approximation du cours (somme des colonnes)" if m == "approx" else "Vecteur propre exact", horizontal=True, key="ahp_m")

    st.subheader("Étape 1. Poids des critères")
    st.caption("Saisir uniquement le triangle supérieur (valeurs de Saaty entre 1/9 et 9) ; le reste est complété par réciprocité.")
    A = pair_editor(ex["criteria"], ex["crit_upper"], f"ahp_crit_{ck}")
    top = W.ahp(A, method)
    steps({"Matrice complète (réciprocité)": pd.DataFrame(A, index=ex["criteria"], columns=ex["criteria"]),
           "Somme des colonnes": pd.DataFrame([top["col_sums"]], columns=ex["criteria"], index=["s"]),
           "Matrice normalisée (a_ij / s_j)": pd.DataFrame(top["normalized"], index=ex["criteria"], columns=ex["criteria"])})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("λmax", f"{top['lambda_max']:.4f}"); c2.metric("CI", f"{top['CI']:.4f}"); c3.metric("RI", f"{top['RI']:.2f}"); c4.metric("CR", f"{top['CR']:.4f}")
    (st.success if top["consistent"] else st.error)("CR < 0,1 : jugements consistants, les poids sont conservés." if top["consistent"] else "CR ≥ 0,1 : jugements inconsistants, l'expert doit réviser ses comparaisons.")
    weights_chart(top["w"], ex["criteria"], "Poids AHP des critères")

    st.subheader("Étape 2. Priorités locales des alternatives")
    alt_As = []
    if len(ex["criteria"]) <= 3:
        for c, col in zip(ex["criteria"], st.columns(len(ex["criteria"]))):
            with col:
                st.markdown(f"Selon « {c} »"); alt_As.append(pair_editor(ex["alts"], ex["alt_matrices"][c], f"ahp_alt_{ck}_{c}"))
    else:
        for c in ex["criteria"]:
            with st.expander(f"Comparaison des alternatives selon « {c} »"):
                alt_As.append(pair_editor(ex["alts"], ex["alt_matrices"][c], f"ahp_alt_{ck}_{c}"))
    res = W.ahp_rank(A, alt_As, method)
    loc = pd.DataFrame(res["local_matrix"], index=ex["alts"], columns=ex["criteria"])
    loc.loc["CR"] = [r["CR"] for r in res["local"]]
    st.dataframe(loc.round(4), width="stretch")
    bad = [c for c, r in zip(ex["criteria"], res["local"]) if not r["consistent"]]
    if bad: st.warning("Matrices locales inconsistantes (CR ≥ 0,1) : " + ", ".join(bad))

    st.subheader("Étape 3. Priorités globales")
    G = res["global_priorities"]; rank = (-G).argsort().argsort() + 1
    ranking_result(ex["alts"], G, rank, "Priorité globale")
    st.markdown("Décomposition (poids du critère multiplié par la priorité locale)")
    st.bar_chart((loc.drop(index="CR") * top["w"]), horizontal=False)

    st.subheader("Étape 4. Sensibilité")
    j = st.selectbox("Critère dont le poids varie", range(len(ex["criteria"])), format_func=lambda i: ex["criteria"][i], key=f"ahp_sj_{ck}")
    grid = np.linspace(0, 1, 101); w0 = top["w"]; oth = np.delete(w0, j); oth = oth / oth.sum(); rows = []
    for g in grid:
        wn = np.empty_like(w0); wn[j] = g; wn[np.arange(len(w0)) != j] = (1 - g) * oth
        rows.append(dict(poids=g, **{ex["alts"][i]: v for i, v in enumerate(res["local_matrix"] @ wn)}))
    sens = pd.DataFrame(rows)
    line_chart(sens, "poids", ex["alts"], f"Priorités globales selon le poids de « {ex['criteria'][j]} »", "priorité", "poids du critère")
    lead = sens[ex["alts"]].values.argmax(1); chg = np.where(np.diff(lead) != 0)[0]
    st.caption(f"Poids actuel : {w0[j]:.3f}. " + ("Le premier change pour w ≈ " + ", ".join(f"{sens['poids'].iloc[c+1]:.2f} ({ex['alts'][lead[c]]} puis {ex['alts'][lead[c+1]]})" for c in chg) + "." if len(chg) else "Le premier ne change jamais sur [0 ; 1]."))

with tab_bwm:
    course_box(r"""
L'expert désigne le meilleur critère (B) et le pire (W), puis note de 1 à 9 le vecteur Best-to-Others $a_{Bj}$ et le vecteur Others-to-Worst $a_{jW}$.
Le nombre de comparaisons est $2n-3$, contre $n(n-1)/2$ pour l'AHP. Les poids minimisent le plus grand écart $\xi$ :

$$\min \xi\;\;\text{s.c.}\;\;|w_B-a_{Bj}w_j|\le\xi,\;|w_j-a_{jW}w_W|\le\xi,\;\sum w_j=1,\;w_j\ge0$$

C'est un programme linéaire, résolu ici avec `scipy.optimize.linprog`. Un $\xi^*$ proche de 0 indique des jugements cohérents.""")
    bcase = st.selectbox("Cas", ["Choix d'un fournisseur (cours)", "Fournisseur stratégique (8 critères)"], key="bwm_case")
    is_c = bcase.startswith("Choix"); bk = "c" if is_c else "l"
    ex = D.BWM_FOURNISSEUR if is_c else D.bwm_large()
    note_source("cours" if is_c else "créé")
    crit = ex["criteria"]
    c1, c2 = st.columns(2)
    best = c1.selectbox("Meilleur critère (B)", range(len(crit)), index=ex["best"], format_func=lambda i: crit[i], key=f"bwm_b_{bk}")
    worst = c2.selectbox("Pire critère (W)", range(len(crit)), index=ex["worst"], format_func=lambda i: crit[i], key=f"bwm_w_{bk}")
    if best == worst: st.error("B et W doivent être différents."); st.stop()
    inp = st.data_editor(pd.DataFrame({"Critère": crit, "Best-to-Others (a_Bj)": ex["bo"], "Others-to-Worst (a_jW)": ex["ow"]}),
                         key=f"bwm_in_{bk}", hide_index=True, width="stretch", disabled=["Critère"])
    bo, ow = inp.iloc[:, 1].to_numpy(float), inp.iloc[:, 2].to_numpy(float)
    for msg in W.bwm_checks(best, worst, bo, ow): st.warning(msg)
    r = W.bwm(best, worst, bo, ow)
    steps({"Contraintes du programme linéaire (A·x ≤ 0, avec x = (w1 … wn, ξ))": pd.DataFrame(r["A_ub"], index=r["labels"], columns=[f"w{i+1}" for i in range(len(crit))] + ["ξ"])},
          fmt=2, title="Programme linéaire (matrice des contraintes)")
    m1, m2 = st.columns(2); m1.metric("ξ*", f"{r['xi']:.4f}")
    m2.metric("Comparaisons demandées", f"{2*len(crit)-3} (AHP : {len(crit)*(len(crit)-1)//2})")
    st.dataframe(pd.DataFrame({"Critère": crit, "Poids": r["w"]}).set_index("Critère").style.format("{:.4f}"), width="stretch")
    weights_chart(r["w"], crit, "Poids BWM")
    if is_c and (best, worst) == (ex["best"], ex["worst"]) and list(bo) == ex["bo"] and list(ow) == ex["ow"]:
        st.warning("Écart avec la diapositive. Le cours indique w = (0,552 ; 0,207 ; 0,155 ; 0,086) et ξ* = 0,08. L'optimum du programme "
                   f"linéaire du cours est unique : w = ({', '.join(f'{v:.3f}' for v in r['w'])}), ξ* = {r['xi']:.3f}, ce qui correspond bien au 0,08 annoncé. "
                   "En revanche, les poids de la diapositive violent ses propres contraintes (écart maximal d'environ 0,22, supérieur à ξ*) : ils proviennent probablement d'un arrondi ou d'une autre formulation.")
