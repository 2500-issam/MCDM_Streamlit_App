import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from mcdm_core import robustness as RB, datasets as D, report as RP
from mcdm_core.ui import *

header("9 · Dossier de décision",
       "Un même tableau passé à toutes les méthodes. On regarde où elles s'accordent, jusqu'où les poids peuvent bouger avant que le premier change, et on exporte le dossier.",
       "cours, bonnes pratiques : plusieurs méthodes, analyse de sensibilité, justification du choix")

OPT = {"Électrification d'un village (14 x 9)": D.energie_village, "Portefeuille d'actions (16 x 10)": D.portefeuille, "Hôpital régional (12 x 8)": D.sites_hopital,
       "Modèles d'apprentissage (14 x 8)": D.ml_models, "Choix de machines (cours)": D.machines_waspas, "Choix d'une voiture (cours)": D.voitures_topsis}
ds = pick_dataset("dos", OPT, "Électrification d'un village (14 x 9)")
X, ben, w, alts, crit = matrix_editor(ds, "dos")
src, w = weights_source(X, ben, w, "dos")
reg, dropped = RB.usable_methods(X, ben, w)
for k, m in dropped.items(): st.warning(f"{k} écartée : {m}")
names = st.multiselect("Méthodes retenues", list(reg), default=list(reg), key="dos_m")
if len(names) < 2: st.warning("Retenir au moins deux méthodes."); st.stop()
ranks = RB.compare_methods(X, ben, w, names, reg); ranks.index = alts
cons = RB.consensus(ranks); m_alt = len(alts)
winner = cons.index[0]

# ------------------------------------------------------------------ 1. frise des rangs
st.header("1. Frise des rangs")
st.caption("Chaque ligne est une alternative, chaque colonne une méthode. Les trois premières du consensus sont mises en couleur, les autres restent en gris.")
top3 = list(cons.index[:3]); fig = go.Figure()
for a in alts:
    if a in top3: continue
    fig.add_trace(go.Scatter(x=names, y=ranks.loc[a].values, mode="lines+markers", line=dict(color="#bdb6a4", width=1.2), marker=dict(size=5, color="#bdb6a4"), name=a, showlegend=False,
                             hovertemplate=a + " : rang %{y}<extra>%{x}</extra>"))
for i, a in enumerate(top3):
    c = PALETTE[i]
    fig.add_trace(go.Scatter(x=names, y=ranks.loc[a].values, mode="lines+markers+text", line=dict(color=c, width=3.5), marker=dict(size=11, color=c), name=a,
                             text=[str(v) for v in ranks.loc[a].values], textposition="middle center", textfont=dict(color="white", size=9),
                             hovertemplate=a + " : rang %{y}<extra>%{x}</extra>"))
fig.update_yaxes(autorange="reversed", dtick=1, title="rang", gridcolor="#e4ddcb"); fig.update_xaxes(side="top")
fig.update_layout(height=max(380, 26 * m_alt + 120), legend=dict(orientation="h", y=-0.06), margin=dict(t=60))
st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ 2. consensus
st.header("2. Consensus")
st.dataframe(cons.style.format({"Rang moyen": "{:.2f}", "Rang médian": "{:.1f}"}).background_gradient(subset=["Points de Borda"], cmap=CMAP), width="stretch")
firsts = {m: ranks[m].idxmin() for m in names}; agree = [m for m in names if firsts[m] == winner]
if len(agree) == len(names):
    avis = f"{winner} est classée première par les {len(names)} méthodes."
else:
    other = "; ".join(f"{m} retient {firsts[m]}" for m in names if firsts[m] != winner)
    avis = f"{winner} arrive en tête du consensus (somme de Borda) et première pour {len(agree)} méthode(s) sur {len(names)}. Désaccords : {other}."
st.success(avis)
spread = cons["Étendue"].sort_values(ascending=False)
st.caption(f"Alternative dont le rang dépend le plus de la méthode : {spread.index[0]} (de {int(cons.loc[spread.index[0], 'Meilleur rang'])} à {int(cons.loc[spread.index[0], 'Pire rang'])}).")

# ------------------------------------------------------------------ 3. règle de fragilité
st.header("3. Règle de fragilité des poids")
st.caption("Pour chaque critère, on déplace son poids de 0 à 1 (les autres restent proportionnels) et on cherche où le premier change. "
           "Le trait épais est l'intervalle dans lequel le premier actuel est stable, le losange est le poids actuel.")
tm = st.selectbox("Méthode analysée", names, index=names.index("TOPSIS") if "TOPSIS" in names else 0, key="dos_tm")
tips = RB.tipping_points(reg[tm], X, ben, w, steps=120)
base_first = alts[int(np.argmax(reg[tm](X, ben, w)))]
order = tips.assign(m=tips["marge_relative"].fillna(1e9)).sort_values("m", ascending=False)       # le plus fragile en haut
fig = go.Figure()
for pos, (_, r) in enumerate(order.iterrows()):
    lo = 0.0 if pd.isna(r["seuil_bas"]) else r["seuil_bas"]; hi = 1.0 if pd.isna(r["seuil_haut"]) else r["seuil_haut"]; lab = crit[int(r["j"])]
    fig.add_trace(go.Scatter(x=[0, 1], y=[lab, lab], mode="lines", line=dict(color="#d6d0c0", width=1), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lab, lab], mode="lines", line=dict(color=TEAL, width=7), showlegend=False,
                             hovertemplate=f"stable de {lo:.3f} à {hi:.3f}<extra>{lab}</extra>"))
    fig.add_trace(go.Scatter(x=[r["poids"]], y=[lab], mode="markers", marker=dict(symbol="diamond", size=11, color=RUST, line=dict(color=PAPER, width=1)), showlegend=False,
                             hovertemplate=f"poids actuel {r['poids']:.3f}<extra>{lab}</extra>"))
fig.update_xaxes(range=[0, 1], title="poids du critère", gridcolor="#e4ddcb", showgrid=True); fig.update_yaxes(title="")
fig.update_layout(height=max(320, 38 * len(crit) + 100), margin=dict(l=10, t=20)); st.plotly_chart(fig, width="stretch")
t = tips.copy(); t.insert(0, "Critère", [crit[j] for j in t["j"]]); t = t.drop(columns="j")
for c_, n_ in (("nouveau_si_bas", "Premier si poids bas"), ("nouveau_si_haut", "Premier si poids haut")):
    t[n_] = [alts[int(v)] if pd.notna(v) else "" for v in t[c_]]
t = t.drop(columns=["nouveau_si_bas", "nouveau_si_haut"]).rename(columns={"poids": "Poids actuel", "seuil_bas": "Seuil bas", "seuil_haut": "Seuil haut", "marge": "Marge", "marge_relative": "Marge relative"})
st.dataframe(t.set_index("Critère").style.format({c: "{:.3f}" for c in ["Poids actuel", "Seuil bas", "Seuil haut", "Marge", "Marge relative"]}, na_rep="jamais"), width="stretch")
fr = tips.dropna(subset=["marge_relative"]).sort_values("marge_relative")
if len(fr):
    f0 = fr.iloc[0]; cf = crit[int(f0["j"])]
    verdict_fr = (f"Avec {tm}, {base_first} reste première tant que le poids de « {cf} » reste dans [{(0 if pd.isna(f0['seuil_bas']) else f0['seuil_bas']):.3f} ; "
                  f"{(1 if pd.isna(f0['seuil_haut']) else f0['seuil_haut']):.3f}]. C'est le critère le plus fragile : un déplacement relatif de {f0['marge_relative']:.0%} suffit à changer le premier.")
    st.warning(verdict_fr)
else:
    verdict_fr = f"Avec {tm}, {base_first} reste première quel que soit le poids d'un critère pris isolément."; st.success(verdict_fr)

# ------------------------------------------------------------------ 4. Monte-Carlo
st.header("4. Tirages aléatoires des poids")
c1, c2 = st.columns(2); conc = c1.slider("Concentration autour des poids actuels", 5, 200, 40, key="dos_conc"); N = c2.select_slider("Nombre de tirages", [100, 300, 500, 1000], 300, key="dos_N")
mc = RB.monte_carlo(reg[tm], X, ben, w, N, conc)
share = pd.Series(mc["top1"], index=alts).sort_values(ascending=False); share = share[share > 0]
bar(share.values, list(share.index), f"Part des tirages où l'alternative est première ({tm})", "fréquence", sort=True, horizontal=True, fmt=".0%")
mc_text = f"Sur {N} tirages de poids, {share.index[0]} est première dans {share.iloc[0]:.0%} des cas ; le classement complet garde un rho de Spearman moyen de {mc['rho_mean']:.2f} avec le classement de base."
st.caption(mc_text)

# ------------------------------------------------------------------ 5. export
st.header("5. Dossier exportable")
verdict = f"{avis} {verdict_fr} {mc_text}"
html_doc = RP.build_report(f"Dossier de décision : {ds['name']}", ds["name"], pd.DataFrame(X, index=alts, columns=crit), ben, w, src, ranks, cons, tips, crit, alts, verdict, dict(text=mc_text))
st.download_button("Télécharger le dossier (HTML)", html_doc.encode("utf-8"), file_name="dossier_de_decision.html", mime="text/html")
st.caption("Fichier autonome : matrice, poids, frise des rangs, consensus, points de bascule et résultat de la simulation, imprimable en PDF depuis le navigateur.")
