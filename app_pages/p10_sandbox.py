import io, numpy as np, pandas as pd, streamlit as st
from mcdm_core import robustness as RB, weights as W, datasets as D, ranking as R
from mcdm_core.ui import *

header('11 · Vos données', 'Importez un CSV (première colonne : alternatives, autres colonnes : critères numériques) et appliquez la chaîne complète du cours.')
st.caption("Chaîne appliquée : types des critères, poids (saisis, entropie ou CRITIC), plusieurs méthodes, comparaison par Spearman. Le fichier n'est pas conservé sur le serveur.")
up = st.file_uploader("Fichier CSV", type=["csv"])
sep = st.radio("Séparateur", [",", ";"], horizontal=True)
if up is not None:
    try:
        raw = pd.read_csv(io.BytesIO(up.getvalue()), sep=sep, index_col=0, encoding="utf-8-sig")
    except Exception as e:
        st.error(f"Lecture impossible : {e}"); st.stop()
else:
    st.caption("Aucun fichier chargé : exemple intégré (portefeuille de 16 actions).")
    raw = D.df(D.portefeuille())
num = raw.apply(pd.to_numeric, errors="coerce")
if num.isna().any().any() or num.shape[0] < 3 or num.shape[1] < 2:
    st.error("Le tableau doit contenir ≥ 3 alternatives, ≥ 2 critères et uniquement des nombres."); st.stop()
ds = dict(alts=list(num.index), crit=list(num.columns), X=num.to_numpy(float), benefit=np.ones(num.shape[1], bool), w=None, source="créé")
X, ben, w, alts, crit = matrix_editor(ds, "sb")
src, w = weights_source(X, ben, w, "sb")
reg, dropped = RB.usable_methods(X, ben, w)
for k, m in dropped.items(): st.warning(f"**{k}** écartée : {m}")
names = st.multiselect("Méthodes", list(reg), default=[m for m in ["WSM", "WASPAS", "TOPSIS", "VIKOR", "PROMETHEE II"] if m in reg])
if len(names) < 1: st.stop()
cmp = RB.compare_methods(X, ben, w, names, reg); cmp.index = alts
scores = pd.DataFrame({n: reg[n](X, ben, w) for n in names}, index=alts)
c1, c2 = st.tabs(["Rangs", "Scores"])
with c1: st.dataframe(cmp.style.background_gradient(cmap=CMAP_R, axis=None), width="stretch")
with c2: st.dataframe(scores.style.format("{:.4f}"), width="stretch")
if len(names) > 1:
    heatmap(RB.spearman_matrix(cmp), "Accord entre méthodes (Spearman)", -1, 1)
st.info("**Meilleure alternative par méthode :** " + " · ".join(f"{m} → {alts[int(cmp[m].values.argmin())]}" for m in names))
out = cmp.copy(); out.columns = [f"rang_{c}" for c in out.columns]; out = out.join(scores.add_prefix("score_"))
df_download(out, "resultats_mcdm.csv")
