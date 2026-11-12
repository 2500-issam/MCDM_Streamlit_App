import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from mcdm_core import fuzzy as F, datasets as D
from mcdm_core.ui import *

header('7 · Logique floue', "Degrés d'appartenance, nombres flous et leurs extensions : intuitionniste, picturale, pythagoricienne, q-ROFN, hésitante.", 'cours, introduction à la logique floue')
t1, t2, t3, t4 = st.tabs(["Ensembles flous", "Nombres flous & intervalles", "IFN · PFN · Pythagoricien · q-ROFN", "Ensembles hésitants"])

with t1:
    course_box(r"""Ensemble flou $A=\{\langle x,\mu_A(x)\rangle\}$, $\mu_A(x)\in[0,1]$. Un ensemble classique est le cas particulier $\mu\in\{0,1\}$.
Fonctions : **trapézoïdale** $(a,b,c,d)$, **triangulaire** $(a,b,c)$, **gaussienne** $e^{-(x-m)^2/2\sigma^2}$.
**Hauteur** $h(A)=\sup\mu$ ; normalisé si $h=1$ · **support** $\{\mu>0\}$ · **noyau** $\{\mu=1\}$ · **α-coupe** $\{\mu\ge\alpha\}$.""")
    mode = st.radio("Exemple", ["Âge : jeune / moyen / âgé (cours)", "Fonction personnalisée"], horizontal=True)
    xs = np.linspace(0, 100, 1001)
    fig = go.Figure()
    if mode.startswith("Âge"):
        sets = F.age_sets(xs); x0 = st.slider("Âge x", 0, 100, 30)
        for i, (k, mu) in enumerate(sets.items()):
            fig.add_trace(go.Scatter(x=xs, y=mu, name=k, line=dict(width=3, color=PALETTE[i])))
        fig.add_vline(x=x0, line_dash="dot")
        deg = {k: float(F.age_sets(np.array([float(x0)]))[k][0]) for k in sets}
        st.write("Degrés d'appartenance à **x = %d ans** : " % x0 + " · ".join(f"{k} = **{v:.2f}**" for k, v in deg.items()))
        st.caption("Cours : à 45 ans on est pleinement « âge moyen » (1) ; à 30 ans, 0,5 « jeune » et 0,5 « âge moyen », la logique classique forcerait un seul choix.")
        mu_sel = sets["Âge moyen"]
    else:
        kind = st.selectbox("Type", ["Trapézoïdale (a,b,c,d)", "Triangulaire (a,b,c)", "Gaussienne (m,σ)"])
        if kind.startswith("Trap"):
            a, b, c, d = [st.number_input(n, value=v, key=f"t{n}") for n, v in zip("abcd", (20., 35., 55., 70.))]
            if not (a <= b <= c <= d): st.error("a ≤ b ≤ c ≤ d requis."); st.stop()
            mu_sel = F.mf_trap(xs, a, b, c, d)
        elif kind.startswith("Tri"):
            a, b, c = [st.number_input(n, value=v, key=f"g{n}") for n, v in zip("abc", (20., 50., 80.))]
            if not (a <= b <= c): st.error("a ≤ b ≤ c requis."); st.stop()
            mu_sel = F.mf_tri(xs, a, b, c)
        else:
            m = st.number_input("m", value=50.0); s = st.number_input("σ", value=12.0, min_value=0.1); mu_sel = F.mf_gauss(xs, m, s)
        fig.add_trace(go.Scatter(x=xs, y=mu_sel, line=dict(width=3, color=PALETTE[0]), name="μ(x)"))
    alpha = st.slider("α (pour l'α-coupe)", 0.0, 1.0, 0.5, 0.05)
    pr = F.set_properties(xs, mu_sel, alpha)
    fig.add_hline(y=alpha, line_dash="dash", line_color="#868e96", annotation_text=f"α = {alpha}")
    fig.update_layout(xaxis_title="x", yaxis_title="μ(x)", height=380, margin=dict(t=30)); st.plotly_chart(fig, width="stretch")
    fmt = lambda t: "∅" if t is None else f"[{t[0]:.1f} ; {t[1]:.1f}]"
    c = st.columns(5); c[0].metric("Hauteur", f"{pr['height']:.2f}"); c[1].metric("Normalisé", "oui" if pr["normalized"] else "non"); c[2].metric("Support", fmt(pr["support"])); c[3].metric("Noyau", fmt(pr["core"])); c[4].metric(f"α-coupe", fmt(pr["alpha_cut"]))
    st.caption("Convexe : " + ("oui" if pr["convex"] else "non"))

with t2:
    course_box(r"""**Nombre flou trapézoïdal** $N(a,b,c,d)$ (triangulaire si $b=c$). Pour $A,B$ positifs : $A+B=(a_1+b_1,\dots)$, $A-B=(a_1-b_4,a_2-b_3,a_3-b_2,a_4-b_1)$, $A\otimes B=(a_1b_1,\dots)$, $\lambda A$, $A\oslash B=(a_1/b_4,a_2/b_3,a_3/b_2,a_4/b_1)$.
**Intervalle** $[a_1,a_2]$ : centre $\frac{a_1+a_2}2$, largeur $\frac{a_2-a_1}2$ ; $A\times B=[\min,\max]$ des 4 produits ; $A/B$ si $0\notin B$.""")
    st.markdown("#### Nombres flous trapézoïdaux")
    c1, c2 = st.columns(2)
    A = tuple(float(x) for x in c1.text_input("A = (a₁,a₂,a₃,a₄)", "4,5,6,7").split(",")); B = tuple(float(x) for x in c2.text_input("B = (b₁,b₂,b₃,b₄)", "1,2,3,4").split(","))
    lam = st.number_input("λ (multiplication par un scalaire)", value=2.0)
    if len(A) != 4 or len(B) != 4: st.error("Quatre valeurs par nombre flou."); st.stop()
    res = {"A + B": F.tfn_add(A, B), "A − B": F.tfn_sub(A, B), "A ⊗ B": F.tfn_mul(A, B), "λ·A": F.tfn_scal(lam, A), "A ⊘ B": F.tfn_div(A, B)}
    st.dataframe(pd.DataFrame(res, index=["a₁", "a₂", "a₃", "a₄"]).T.round(4), width="stretch")
    xs = np.linspace(min(0, min(sum(res.values(), ()))) - 1, max(sum(res.values(), ())) + 1, 600); fig = go.Figure()
    for i, (k, v) in enumerate([("A", A), ("B", B)] + list(res.items())[:3]):
        fig.add_trace(go.Scatter(x=xs, y=F.mf_trap(xs, *v), name=k, line=dict(color=PALETTE[i], width=2.5)))
    fig.update_layout(height=340, xaxis_title="x", yaxis_title="μ", margin=dict(t=20)); st.plotly_chart(fig, width="stretch")
    st.markdown("#### Nombres d'intervalle")
    c1, c2 = st.columns(2)
    I = tuple(float(x) for x in c1.text_input("I = [a₁,a₂]", "2,5").split(",")); J = tuple(float(x) for x in c2.text_input("J = [b₁,b₂]", "1,3").split(","))
    out = {"I + J": F.int_add(I, J), "I − J": F.int_sub(I, J), "I × J": F.int_mul(I, J)}
    try: out["I / J"] = F.int_div(I, J)
    except ZeroDivisionError as e: st.warning(str(e))
    st.dataframe(pd.DataFrame(out, index=["borne inf", "borne sup"]).T.round(4), width="stretch")
    st.caption(f"Centre de I = {F.int_center(I):.2f}, largeur = {F.int_width(I):.2f} · I ≤ J ⇔ m(I) ≤ m(J) : {'vrai' if F.int_center(I) <= F.int_center(J) else 'faux'}")

with t3:
    course_box(r"""**IFN** $\alpha=(\mu,\nu)$, $\mu+\nu\le1$, hésitation $\pi=1-\mu-\nu$. $\alpha_1\oplus\alpha_2=(\mu_1+\mu_2-\mu_1\mu_2,\nu_1\nu_2)$, $\alpha_1\otimes\alpha_2=(\mu_1\mu_2,\nu_1+\nu_2-\nu_1\nu_2)$,
$\lambda\alpha=(1-(1-\mu)^\lambda,\nu^\lambda)$, $\alpha^\lambda=(\mu^\lambda,1-(1-\nu)^\lambda)$. Score $S=\mu-\nu$, précision $H=\mu+\nu$ (comparaison : $S$ puis $H$). **IFWA** : $(1-\prod(1-\mu_j)^{\sigma_j},\prod\nu_j^{\sigma_j})$.
**Pictural** $(\mu,\nu,\pi)$, $\mu+\nu+\pi\le1$ (refus = 1 − somme) · **Pythagoricien** $\mu^2+\nu^2\le1$ · **q-ROFN** $\mu^q+\nu^q\le1$, $\pi=\sqrt[q]{1-\mu^q-\nu^q}$.""")
    st.markdown("#### Nombres flous intuitionnistes")
    c1, c2, c3 = st.columns(3)
    a1 = (c1.number_input("μ₁", 0.0, 1.0, 0.6, 0.05), c1.number_input("ν₁", 0.0, 1.0, 0.3, 0.05)); a2 = (c2.number_input("μ₂", 0.0, 1.0, 0.5, 0.05), c2.number_input("ν₂", 0.0, 1.0, 0.4, 0.05))
    lam = c3.number_input("λ", 0.1, 10.0, 2.0, 0.1)
    if not (F.ifn_check(a1) and F.ifn_check(a2)): st.error("Contrainte μ + ν ≤ 1 non respectée."); st.stop()
    res = {"α₁ ⊕ α₂": F.ifn_add(a1, a2), "α₁ ⊗ α₂": F.ifn_mul(a1, a2), "λ·α₁": F.ifn_scal(lam, a1), "α₁^λ": F.ifn_pow(a1, lam)}
    tab = pd.DataFrame(res, index=["μ", "ν"]).T; tab["π"] = 1 - tab["μ"] - tab["ν"]; tab["Score μ−ν"] = tab["μ"] - tab["ν"]; tab["Précision μ+ν"] = tab["μ"] + tab["ν"]
    st.dataframe(tab.round(4), width="stretch")
    cmp = F.ifn_compare(a1, a2); st.write(f"Comparaison : S(α₁) = {F.ifn_score(a1):.2f}, S(α₂) = {F.ifn_score(a2):.2f} → **α₁ {'>' if cmp > 0 else '<' if cmp < 0 else '='} α₂**")
    c1, c2 = st.columns(2)
    sig = c1.slider("Poids du décideur 1 (σ₁) pour l'IFWA", 0.0, 1.0, 0.5, 0.05)
    ag = F.ifwa([a1, a2], [sig, 1 - sig]); c2.metric("IFWA(α₁, α₂)", f"({ag[0]:.3f} ; {ag[1]:.3f})", f"π = {F.ifn_pi(ag):.3f}", delta_color="off")
    st.caption("Exemple du cours : 10 votants, 5 pour, 3 contre, 2 absents → (μ, ν, π) = (0,5 ; 0,3 ; 0,2).")
    st.markdown("#### Pictural, pythagoricien, q-ROFN")
    q = st.slider("q (rang)", 1, 8, 3); mu = st.slider("μ", 0.0, 1.0, 0.8, 0.01); nu = st.slider("ν", 0.0, 1.0, 0.6, 0.01)
    ok = lambda k: F.qrof_check(mu, nu, k)
    st.write(f"μ = {mu:.2f}, ν = {nu:.2f} : IFN (q=1) {'oui' if ok(1) else 'non'} · Pythagoricien (q=2) {'oui' if ok(2) else 'non'} · q-ROFN (q={q}) {'oui' if ok(q) else 'non'}"
             + (f" · hésitation π = {F.qrof_pi(mu, nu, q):.3f}" if ok(q) else ""))
    th = np.linspace(0, np.pi / 2, 200); fig = go.Figure()
    for k, col, nm in [(1, PALETTE[2], "IFN (q=1)"), (2, PALETTE[1], "PFN (q=2)"), (q, PALETTE[0], f"q-ROFN (q={q})")]:
        xs = np.cos(th) ** (2 / k); ys = np.sin(th) ** (2 / k)
        fig.add_trace(go.Scatter(x=np.r_[0, xs, 0], y=np.r_[0, ys, 0], fill="toself", opacity=.25, name=nm, line=dict(color=col)))
    fig.add_trace(go.Scatter(x=[mu], y=[nu], mode="markers", marker=dict(size=12, color="black"), name="(μ, ν)"))
    fig.update_layout(title="Espaces admissibles (μ, ν) : la zone grandit avec q", xaxis_title="μ", yaxis_title="ν", height=420, yaxis_scaleanchor="x"); st.plotly_chart(fig, width="stretch")
    st.markdown("**Ensemble flou pictural** (oui, abstention, non, refus)")
    c = st.columns(3); p = (c[0].number_input("μ (oui)", 0.0, 1.0, 0.4, 0.05), c[1].number_input("η (neutre)", 0.0, 1.0, 0.2, 0.05), c[2].number_input("ν (non)", 0.0, 1.0, 0.3, 0.05))
    st.write("Valide" if F.pfn_check(p) else "Invalide (μ+η+ν > 1)", f"· degré de refus = {F.pfn_refusal(p):.2f}")

with t4:
    course_box(r"""Ensemble flou hésitant (Torra 2010) : $h=\{\gamma_1,\dots,\gamma_k\}$, plusieurs degrés d'appartenance possibles (plusieurs décideurs en désaccord). Score $s(h)=\frac1l\sum\gamma$, déviation $\sigma(h)=\sqrt{\frac1l\sum(\gamma-s)^2}$.
Comparaison : plus grand score gagne ; à score égal, **la plus petite déviation** gagne. Distance de Hamming hésitante après tri des valeurs.""")
    st.success("Exemple du cours : comparer h₁ = {0,2 ; 0,3 ; 0,5 ; 0,8}, h₂ = {0,4 ; 0,6 ; 0,8}, h₃ = {0,3 ; 0,45 ; 0,6}.")
    txt = {k: st.text_input(k, ", ".join(map(str, v)), key=f"h{k}") for k, v in D.HESITANT_COURS.items()}
    H = {k: sorted(float(x) for x in v.split(",")) for k, v in txt.items()}
    tab = pd.DataFrame({"Score s(h)": {k: F.hfn_score(h) for k, h in H.items()}, "Déviation σ(h)": {k: F.hfn_dev(h) for k, h in H.items()}})
    import functools
    order = sorted(H, key=functools.cmp_to_key(lambda a, b: -F.hfn_compare(H[a], H[b])))
    tab["Rang"] = [order.index(k) + 1 for k in tab.index]
    st.dataframe(tab.round(4).sort_values("Rang"), width="stretch"); st.write("Classement : " + " > ".join(order))
    st.markdown("**Opérations** (h₁, h₂, α) ")
    al = st.number_input("α", 0.1, 5.0, 2.0, 0.1); k1, k2 = list(H)[:2]
    st.write({f"{k1}^α": F.hfn_pow(H[k1], al), f"α·{k1}": F.hfn_scal(al, H[k1]), f"{k1} ⊕ {k2}": F.hfn_add(H[k1], H[k2]), f"{k1} ⊗ {k2}": F.hfn_mul(H[k1], H[k2])})
    ext = st.radio("Prolongation du plus court ensemble pour la distance", ["max", "min", "mean"], horizontal=True)
    st.metric(f"Distance de Hamming ||{k1} − {k2}||", f"{F.hfn_hamming(H[k1], H[k2], ext):.4f}")
