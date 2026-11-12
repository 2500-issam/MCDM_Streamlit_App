import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from mcdm_core import hybrid as H, datasets as D, ranking as R, weights as W, robustness as RB
from mcdm_core.ui import *

header('10 · MCDM et apprentissage automatique', "Quatre manières de relier les méthodes multicritères à l'apprentissage : classifieur, pondération par valeurs de Shapley, données dynamiques, extraction de scores depuis du texte.", 'cours, MCDM et IA, MCDM dynamique, MCDM et IA générative')
t1, t2, t3, t4 = st.tabs(["Scénario 1 · MCDM → classifieur", "Scénario 2 · XAI / SHAP", "MCDM dynamique", "LLM extracteur de scores"])
crit = list(D.fournisseurs_ml().columns); ben = np.array(D.FOURNISSEURS_BENEFIT, bool)


@st.cache_data
def data(n): return D.fournisseurs_ml(n)


def score_fn(name):
    return {"TOPSIS": lambda X, w: R.topsis(X, ben, w)["RC"], "WSM": lambda X, w: R.wsm(R.ratio_norm(X, ben), w),
            "WASPAS": lambda X, w: R.waspas(X, ben, w)["Q"]}[name]


with st.sidebar:
    st.markdown("### Jeu de données (créé)")
    n = st.select_slider("Nombre de fournisseurs simulés", [100, 200, 300, 500], 300)
df = data(n)

with t1:
    course_box(r"""**Scénario 1.** (1) Pondérer par MCDM (AHP, CRITIC, entropie…) ; (2) calculer le score $S_i$ de chaque alternative par une méthode additive ; (3) **étiqueter** (A, B, C, D) selon $S_i$ ; (4) entraîner un modèle de ML critères → classe ; (5) prédire la classe d'une **nouvelle** alternative.""")
    note_source("créé", f"{n} fournisseurs, 8 critères (graine 42).")
    c1, c2 = st.columns(2)
    wsrc = c1.radio("Poids", ["Subjectifs (AHP/BWM supposés)", "CRITIC", "Entropie"], key="h1w"); meth = c2.radio("Méthode de score", ["TOPSIS", "WSM", "WASPAS"], horizontal=True, key="h1m")
    Xv = df.to_numpy(float)
    w = {"Subjectifs (AHP/BWM supposés)": np.array(D.FOURNISSEURS_W_SUBJ), "CRITIC": W.critic_weights(Xv, ben)["w"], "Entropie": W.entropy_weights(Xv, ben, "minmax")["w"]}[wsrc]
    st.caption("Poids : " + " · ".join(f"{c.split(' (')[0]} {v:.2f}" for c, v in zip(crit, w)))
    S = score_fn(meth)(Xv, w); y = H.label_by_quantiles(S)
    res = H.train_classifier(df, y)
    c1, c2, c3 = st.columns(3); c1.metric("Exactitude (validation croisée 5 plis)", f"{res['cv_mean']:.1%}", f"± {res['cv_std']:.1%}", delta_color="off")
    c2.metric("Classes", "A (top 25 %) … D"); c3.metric("Observations", n)
    imp = res["importances"].sort_values(); fig = go.Figure(go.Bar(x=imp.values, y=[i.split(" (")[0] for i in imp.index], orientation="h", marker_color=PALETTE[0]))
    fig.update_layout(title="Importance des critères pour le classifieur (Random Forest)", height=320, margin=dict(l=10, t=50)); st.plotly_chart(fig, width="stretch")
    st.markdown("#### Prédire la classe d'un nouveau fournisseur")
    cols = st.columns(4); new = {}
    for i, c in enumerate(crit):
        lo, hi = float(df[c].min()), float(df[c].max()); new[c] = cols[i % 4].slider(c, lo, hi, float(df[c].median()), key=f"nw{i}")
    xn = pd.DataFrame([new]); pred = res["model"].predict(xn)[0]; proba = pd.Series(res["model"].predict_proba(xn)[0], index=res["model"].classes_)
    direct = score_fn(meth)(np.vstack([Xv, xn.to_numpy(float)]), w)[-1]; pct = (S < direct).mean()
    c1, c2 = st.columns(2); c1.metric("Classe prédite par le ML", pred); c2.metric("Score MCDM direct (percentile)", f"{pct:.0%}", help="Part des fournisseurs existants ayant un score inférieur.")
    st.bar_chart(proba, height=200)
    st.warning("**Limite :** le modèle apprend à *imiter* la méthode MCDM choisie ; il n'apporte pas de nouvelle information, mais généralise vite à de nombreuses alternatives sans refaire les calculs (et sans comparer à tout le lot).")

with t2:
    course_box(r"""**Scénario 2, XAI comme approche de pondération.** (1-2) poids MCDM et score $S_i=\sum w_jx_{ij}$ ; (3) apprendre $\hat S=f(x)$ (XGBoost, Random Forest, réseau…) ; (4) **SHAP** : $\hat S=\phi_0+\sum_j\phi_j$ (importance locale et globale) ;
(5) **mise à jour** : $w_j^{new}=\alpha w_j^{AHP}+(1-\alpha)\dfrac{|\phi_j|}{\sum|\phi_j|}$.""")
    st.caption("Les valeurs de Shapley sont calculées **exactement** (espérance marginale sur un échantillon de référence, 2⁸ coalitions), même grandeur que `shap`, sans dépendance lourde.")
    alpha = st.slider("α : confiance dans les poids experts", 0.0, 1.0, 0.5, 0.05, key="h2a")
    w0 = np.array(D.FOURNISSEURS_W_SUBJ); Xv = df.to_numpy(float); S = R.topsis(Xv, ben, w0)["RC"]
    @st.cache_data(show_spinner="Apprentissage du modèle et calcul des valeurs de Shapley…")
    def _shap(n, alpha): return H.shap_weight_update(data(n), S, w0, alpha, n_explain=30)
    r = _shap(n, alpha)
    st.metric("R² du modèle de substitution Ŝ = f(x)", f"{r['r2']:.3f}")
    tab = pd.DataFrame({"w AHP (expert)": w0, "Importance SHAP |φ| normalisée": r["shap_importance"], "w nouveau": r["w_new"]}, index=[c.split(" (")[0] for c in crit])
    st.dataframe(tab.style.format("{:.3f}"), width="stretch"); st.bar_chart(tab, height=320)
    ra, rb = RB.rank_of(R.topsis(Xv, ben, w0)["RC"]), RB.rank_of(R.topsis(Xv, ben, r["w_new"])["RC"])
    top = np.argsort(rb)[:10]
    st.markdown("**Effet sur le classement TOPSIS** (10 meilleurs avec les nouveaux poids)")
    st.dataframe(pd.DataFrame({"Fournisseur n°": top, "Rang (poids experts)": ra[top], "Rang (nouveaux poids)": rb[top]}).set_index("Fournisseur n°"), width="stretch")
    st.caption(f"ρ de Spearman entre les deux classements complets : **{RB.spearman(ra, rb):.3f}**")
    st.markdown("**Explication locale d'un fournisseur**")
    i = st.selectbox("Fournisseur", list(range(len(df))), index=0, key="h2i")
    bg = Xv[np.random.default_rng(0).choice(len(Xv), 25, replace=False)]
    phi = H.exact_shapley(r["model"].predict, Xv[i], bg); base_v = float(r["model"].predict(bg).mean())
    fig = go.Figure(go.Bar(x=[c.split(" (")[0] for c in crit], y=phi, marker_color=[PALETTE[2] if v >= 0 else PALETTE[4] for v in phi]))
    fig.update_layout(title=f"Contributions φⱼ au score prédit (base φ₀ = {base_v:.3f}, prédiction = {base_v+phi.sum():.3f})", height=330); st.plotly_chart(fig, width="stretch")

with t3:
    course_box(r"""Les critères évoluent dans le temps (rendement, volatilité, liquidité, bêta, ratio de Sharpe…). Pour chaque couple (alternative, critère), un modèle de séries temporelles (ARIMA, LSTM, GARCH…) prévoit la valeur à $t+1$ ; la matrice prévisionnelle est ensuite traitée par une méthode MCDM.
Limites : propagation des erreurs (une erreur de prévision biaise le classement) et dépendance aux modèles utilisés.""")
    mode = st.radio("Matrice de décision", ["Tableau de prévisions du cours (3 x 3)", "Prévisions à partir de séries simulées (10 actions x 7 critères)"], horizontal=True)
    if mode.startswith("Tableau"):
        note_source("cours")
        wd = np.array(st.text_input("Poids (rendement, volatilité, liquidité)", "0.4,0.35,0.25").split(","), float); wd = wd / wd.sum(); bd = np.array([True, False, True])
        M = D.ACTIONS_DYN_COURS; alts = list(M.index); X = M.to_numpy(float); st.dataframe(M, width="stretch")
    else:
        note_source("créé", "Séries mensuelles sur 60 mois, graine 7.")
        labels = [H.DYN_LABELS[c] for c in H.DYN_CRITERIA]
        wd = np.array(st.text_input("Poids (7 critères, dans l'ordre du tableau)", "0.20,0.15,0.10,0.12,0.18,0.10,0.15").split(","), float); wd = wd / wd.sum(); bd = np.array(H.DYN_BENEFIT, bool)
        ser = H.simulate_series(); alts = list(ser["action"].unique())
        var = st.selectbox("Série affichée", H.DYN_CRITERIA, format_func=lambda c: H.DYN_LABELS[c]); piv = ser.pivot(index="t", columns="action", values=var).reset_index()
        line_chart(piv, "t", alts, f"{H.DYN_LABELS[var]}, séries simulées sur 60 mois", H.DYN_LABELS[var], "mois")
        c1, c2 = st.columns(2); meth = c1.selectbox("Modèle de prévision", ["holt", "ses", "ar1", "moyenne_mobile", "naive"], key="fm")
        a_ = c2.slider("Paramètre de lissage α", 0.05, 0.95, 0.4, 0.05)
        X = np.array([[H.forecast_next(ser[ser.action == a][v].to_numpy(), meth, alpha=a_) for v in H.DYN_CRITERIA] for a in alts])
        st.markdown("Matrice prévisionnelle à t+1"); st.dataframe(pd.DataFrame(X, index=alts, columns=labels).round(3), width="stretch")
    if len(wd) != len(bd): st.error(f"{len(bd)} poids sont attendus."); st.stop()
    t = R.topsis(np.clip(X, 1e-9, None), bd, wd); ranking_result(alts, t["RC"], t["rank"], "RC* (TOPSIS)")
    st.caption("Changer de modèle de prévision ou de lissage : si le classement bascule, la décision est sensible aux erreurs de prévision.")

with t4:
    course_box(r"""Un modèle de langage ajusté $f_{LLM}(T,C)\to s\in[0,10]^n$ extrait de textes non structurés (CV, rapports, avis) un vecteur de scores par critère et une justification. Ces scores forment une ligne de la matrice de décision passée à la méthode MCDM.
Les données d'ajustement sont des triplets $(T,C,s^*)$ annotés par des experts. Limites : qualité des annotations, subjectivité entre annotateurs, rubrique de notation rigoureuse indispensable.""")
    st.warning("Simulateur hors ligne : l'extracteur est un jeu de règles déterministes qui applique la rubrique ci-dessous, sans appeler de modèle de langage. Il illustre la chaîne texte, scores, MCDM. Remplacer `extract_scores` par un appel à un modèle respectant la même rubrique est l'extension naturelle.")
    st.dataframe(pd.DataFrame(H.CV_RUBRIC, columns=["Critère", "Sens", "Rubrique de notation"]), width="stretch", hide_index=True)
    cvs = {"Candidat 1 (CV du cours)": "Ingénieur logiciel avec 7 ans d'expérience en développement Python et Java, management d'une équipe de 5 personnes, certifié PMP. Prétentions : 85 000 €/an.",
           "Candidat 2": "Développeuse data, master en statistique, 3 ans d'expérience en Python, SQL et AWS. Aucune expérience de management. Anglais et français. Préavis de 2 mois. Prétentions : 55 000 €/an.",
           "Candidat 3": "Architecte, ingénieur, 12 ans d'expérience (Java, Kubernetes, Docker), responsable d'une équipe de 9 personnes, certifié TOGAF et Scrum. Anglais, français, allemand. Auteur d'une publication et d'un brevet. Préavis de 3 mois. Prétentions : 98 000 €/an.",
           "Candidat 4": "Docteur en informatique, 5 ans d'expérience en recherche (Python, Scala), contributeur open source et projet majeur de recherche. Anglais, français, espagnol. Disponible immédiatement. Prétentions : 62 000 €/an.",
           "Candidat 5": "Chef de projet, licence, 9 ans d'expérience, management d'une équipe de 12 personnes, certifié PMP, ITIL et Scrum. Français et arabe. Préavis de 1 mois. Prétentions : 90 000 €/an.",
           "Candidat 6": "Développeur junior, 1 an d'expérience en JavaScript et React. Master. Anglais, français, arabe. Disponible immédiatement. Prétentions : 38 000 €/an."}
    names = list(cvs); mats = {}
    for k in names:
        tx = st.text_area(k, cvs[k], height=80, key=f"cv_{k}"); e = H.extract_scores(tx); mats[k] = e["Score /10"].to_list()
        with st.expander(f"Scores et justifications, {k}"): st.dataframe(e, width="stretch", hide_index=True)
    Xc = np.array(list(mats.values()), float)
    cw = np.array([float(x) for x in st.text_input("Poids des 7 critères", "0.25,0.12,0.12,0.12,0.08,0.13,0.18").split(",")]); cw = cw / cw.sum()
    short_c = [c[0].split(" /")[0] for c in H.CV_RUBRIC]
    st.markdown("Matrice de décision extraite (le score salarial est déjà inversé par la rubrique, donc traité comme un critère de bénéfice)")
    st.dataframe(pd.DataFrame(Xc, index=names, columns=short_c), width="stretch")
    t = R.topsis(Xc + 1e-9, np.ones(len(cw), bool), cw); ranking_result(names, t["RC"], t["rank"], "RC* (TOPSIS)")
    with st.expander("Gabarit de prompt pour un vrai modèle de langage"):
        st.code("Tu es un évaluateur RH. Pour le texte T et chaque critère de la liste C (avec sa définition opérationnelle et sa rubrique),\n"
                "attribue un score entier de 0 à 10 et une justification d'une phrase fondée uniquement sur T.\n"
                "Pour un critère de coût, inverse le score (10 = très favorable). Réponds en JSON : {critère: {score, justification}}.\n\nT = <CV>\nC = <critères et rubriques>", language="text")
