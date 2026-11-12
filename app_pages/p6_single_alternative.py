import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from mcdm_core import single as S, datasets as D
from mcdm_core.ui import *

header('6 · Alternative isolée', 'Un objet unique est évalué par rapport à des seuils ou à une référence : score, décision, classe.', 'cours, techniques MCDM à une seule alternative')
t1, t2, t3 = st.tabs(["MAUT + seuil fixé", "Choisir le seuil à partir des données", "Distance à l'idéal (TOPSIS-like)"])

with t1:
    course_box(r"""$S(A)=\sum_j w_ju_j(x_j)$ avec $u_j=\frac{x-x_j^{min}}{x_j^{max}-x_j^{min}}$ (bénéfice) ou $\frac{x-x_j^{max}}{x_j^{min}-x_j^{max}}$ (coût). **Décision :** accepter si $S(A)\ge\tau$, rejeter sinon.
Cas d'usage : scoring de crédit, évaluation de performance, agrégation de KPI.""")
    note_source("créé", "Scoring de crédit d'un demandeur, 8 critères (valeurs inventées).")
    base = pd.DataFrame({"Critère": ["Revenu mensuel (DH)", "Taux d'endettement (%)", "Ancienneté dans l'emploi (ans)", "Incidents de paiement (nb)", "Épargne (DH)",
                                     "Valeur de la garantie (DH)", "Durée du prêt (mois)", "Apport personnel (%)"],
                         "Valeur x": [14000, 32, 6, 1, 45000, 180000, 84, 15], "min": [2000, 0, 0, 0, 0, 0, 12, 0], "max": [40000, 60, 20, 5, 200000, 800000, 240, 50],
                         "Type": [BEN, COST, BEN, COST, BEN, BEN, COST, BEN], "Poids": [.22, .18, .08, .14, .08, .10, .06, .14]})
    ed = st.data_editor(base, key="maut", hide_index=True, width="stretch", disabled=["Critère"],
                        column_config={"Type": st.column_config.SelectboxColumn(options=[BEN, COST], required=True)})
    ok = (ed["max"] > ed["min"]).all()
    if not ok: st.error("Il faut min < max pour chaque critère."); st.stop()
    w = ed["Poids"].to_numpy(float); w = w / w.sum()
    r = S.maut(ed["Valeur x"], ed["min"], ed["max"], (ed["Type"] == BEN).to_numpy(), w)
    tau = st.slider("Seuil de décision τ (fixé par politique / experts)", 0.0, 1.0, 0.7, 0.01)
    steps({"Utilités uⱼ et contributions wⱼ·uⱼ": pd.DataFrame({"uⱼ": r["u"], "wⱼ": w, "wⱼ·uⱼ": r["contrib"]}, index=ed["Critère"]).T})
    c1, c2 = st.columns(2); c1.metric("Score S(A)", f"{r['S']:.3f}"); c2.metric("Décision", "Accepter" if r["S"] >= tau else "Rejeter", delta=f"{r['S']-tau:+.3f} vs τ")
    bar(r["contrib"], list(ed["Critère"]), "Contribution de chaque critère au score", "wⱼ·uⱼ", sort=False)
    cls = "A" if r["S"] >= .8 else "B" if r["S"] >= .6 else "C" if r["S"] >= .4 else "D"
    st.caption(f"Classe indicative (A ≥ 0,8 · B ≥ 0,6 · C ≥ 0,4 · D sinon) : **{cls}**.")

with t2:
    course_box(r"""Si on dispose de décisions passées $y_i$, on choisit $\tau$ qui sépare le mieux les classes ($\hat y=1$ si $S\ge\tau$). Courbe **ROC** : $TPR=\frac{TP}{TP+FN}$ en fonction de $FPR=\frac{FP}{FP+TN}$.
**Méthode 1 :** minimiser la distance au point idéal $d=\sqrt{FPR^2+(1-TPR)^2}$. **Méthode 2 :** maximiser l'indice de Youden $J=TPR-FPR$. En cas d'égalité : départager par le coût d'un faux positif / faux négatif.""")
    which = st.radio("Historique", ["Table du cours (6 décisions)", "Historique simulé (60 dossiers)"], horizontal=True, key="roc_which")
    if which.startswith("Table"):
        base = D.SEUIL_COURS.copy(); note_source("cours")
    else:
        rg = np.random.default_rng(8); sc0 = rg.beta(5, 3, 60).round(3); yy = rg.random(60) < 1 / (1 + np.exp(-11 * (sc0 - .62)))
        base = pd.DataFrame({"Alternative": [f"D{i:02d}" for i in range(1, 61)], "Score S": sc0, "Décision": np.where(yy, "Accepter", "Rejeter")}); note_source("créé", "Graine 8.")
    base["Décision"] = base["Décision"].astype(str)
    ed = st.data_editor(base, key=f"roc_in_{which[:5]}", hide_index=True, width="stretch", num_rows="dynamic",
                        column_config={"Décision": st.column_config.SelectboxColumn(options=["Accepter", "Rejeter"])})
    ed = ed.dropna()
    sc, y = ed["Score S"].to_numpy(float), (ed["Décision"] == "Accepter").astype(int).to_numpy()
    if y.sum() == 0 or y.sum() == len(y): st.error("Il faut au moins une décision « Accepter » et une « Rejeter »."); st.stop()
    cand = S.candidate_thresholds(sc)
    full = S.roc_table(sc, y, cand)
    st.markdown("**Seuils testés par le cours** (0,75 · 0,65 · 0,60) :")
    st.dataframe(S.roc_table(sc, y, [.75, .65, .60]).style.format({"TPR": "{:.2f}", "FPR": "{:.2f}", "d": "{:.2f}", "J": "{:.2f}", "exactitude": "{:.2f}"}), width="stretch", hide_index=True)
    st.markdown("**Tous les seuils candidats** (= chaque score observé) :")
    st.dataframe(full.style.format({"TPR": "{:.2f}", "FPR": "{:.2f}", "d": "{:.3f}", "J": "{:.3f}", "exactitude": "{:.2f}"}), width="stretch", hide_index=True)
    b_d, b_j = full.loc[full["d"].idxmin()], full.loc[full["J"].idxmax()]
    c1, c2 = st.columns(2)
    c1.metric("Seuil optimal, distance à l'idéal", f"{b_d['seuil']:.2f}", f"d = {b_d['d']:.3f}", delta_color="off")
    c2.metric("Seuil optimal, Youden", f"{b_j['seuil']:.2f}", f"J = {b_j['J']:.3f}", delta_color="off")
    ties = full[np.isclose(full["d"], b_d["d"])]
    if len(ties) > 1: st.warning(f"Égalité entre les seuils {', '.join(f'{t:.2f}' for t in ties['seuil'])} : départager selon le coût d'un faux positif vs faux négatif (cours).")
    roc = pd.concat([pd.DataFrame({"FPR": [0.0], "TPR": [0.0]}), full[["FPR", "TPR"]].sort_values(["FPR", "TPR"]), pd.DataFrame({"FPR": [1.0], "TPR": [1.0]})])
    fig = go.Figure(); fig.add_trace(go.Scatter(x=roc["FPR"], y=roc["TPR"], mode="lines+markers", name="ROC", line=dict(color=PALETTE[0])))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="hasard", line=dict(dash="dot", color="#868e96")))
    fig.add_trace(go.Scatter(x=[b_d["FPR"]], y=[b_d["TPR"]], mode="markers", name="min d", marker=dict(size=14, symbol="star", color=PALETTE[1])))
    fig.update_layout(title="Courbe ROC", xaxis_title="FPR", yaxis_title="TPR", height=400); st.plotly_chart(fig, width="stretch")

with t3:
    course_box(r"""Même avec une seule alternative, on la compare à un **point idéal** $A^+$ et à un **anti-idéal** $A^-$ : $D^+=\sqrt{\sum_j(x_j-x_j^+)^2}$, $D^-=\sqrt{\sum_j(x_j-x_j^-)^2}$, $C=\dfrac{D^-}{D^++D^-}$.
Ici chaque écart est divisé par $|x_j^+-x_j^-|$ pour que les unités n'influencent pas la distance (précaution ajoutée, absente de la diapo).""")
    base = pd.DataFrame({"Critère": ["Disponibilité (%)", "Taux de défaut (%)", "Satisfaction client (/10)", "Délai de réponse (h)", "Rotation du personnel (%)", "Marge opérationnelle (%)", "Conformité réglementaire (%)"],
                         "Valeur x": [96.0, 2.1, 7.4, 9.0, 14.0, 11.5, 93.0], "Idéal x⁺": [100.0, 0.0, 10.0, 0.0, 0.0, 25.0, 100.0], "Anti-idéal x⁻": [90.0, 5.0, 0.0, 48.0, 40.0, 0.0, 70.0]})
    ed = st.data_editor(base, key="ts", hide_index=True, width="stretch", disabled=["Critère"])
    use_w = st.checkbox("Pondérer les critères (poids égaux sinon)", False)
    w = None
    if use_w:
        w = np.array(st.text_input("Poids (séparés par des virgules)", "0.2,0.15,0.2,0.1,0.1,0.15,0.1").split(","), float)
        if len(w) != len(ed): st.error("Un poids par critère."); st.stop()
        w = w / w.sum()
    r = S.topsis_single(ed["Valeur x"], ed["Idéal x⁺"], ed["Anti-idéal x⁻"], w)
    c1, c2, c3 = st.columns(3); c1.metric("D⁺", f"{r['D_plus']:.3f}"); c2.metric("D⁻", f"{r['D_minus']:.3f}"); c3.metric("Indice C", f"{r['C']:.3f}")
    st.progress(float(np.clip(r["C"], 0, 1)), text="0 = proche de l'anti-idéal · 1 = proche de l'idéal")
