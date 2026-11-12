"""Tests du noyau : chaque valeur de référence provient des diapositives du cours (ou d'une propriété mathématique vérifiable)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from mcdm_core import weights as W, ranking as R, outranking as O, single as S, fuzzy as F, robustness as RB, hybrid as H, datasets as D


def test_ahp_voiture():
    ex = D.AHP_VOITURE
    A = W.ahp_reciprocal(np.array(ex["crit_upper"]))
    top = W.ahp(A)
    assert np.allclose(top["w"], [0.669, 0.088, 0.243], atol=1e-3)
    assert top["consistent"] and top["CR"] < 0.1
    alts = [W.ahp_reciprocal(np.array(ex["alt_matrices"][c])) for c in ex["criteria"]]
    G = W.ahp_rank(A, alts)["global_priorities"]
    assert np.allclose(G, [0.624, 0.376], atol=2e-3) and np.isclose(G.sum(), 1)


def test_ahp_inconsistent_detected():
    A = W.ahp_reciprocal(np.array([[1, 9, 1 / 9], [0, 1, 9], [0, 0, 1]], float))
    assert not W.ahp(A)["consistent"]


def test_bwm_lp_is_optimal_and_feasible():
    ex = D.BWM_FOURNISSEUR
    r = W.bwm(ex["best"], ex["worst"], ex["bo"], ex["ow"])
    w, bo, ow = r["w"], np.array(ex["bo"]), np.array(ex["ow"])
    assert np.isclose(w.sum(), 1) and (w >= -1e-9).all()
    dev = max(max(abs(w[0] - bo[j] * w[j]), abs(w[j] - ow[j] * w[3])) for j in range(4))
    assert np.isclose(dev, r["xi"], atol=1e-7)                 # ξ* = plus grand écart effectif
    assert abs(r["xi"] - 0.08) < 0.01                          # la diapo annonce ξ* = 0,08
    assert w.argmax() == 0 and w.argmin() == 3
    slide = np.array([.552, .207, .155, .086])                 # poids de la diapo : violent leurs propres contraintes
    dev_slide = max(max(abs(slide[0] - bo[j] * slide[j]), abs(slide[j] - ow[j] * slide[3])) for j in range(4))
    assert dev_slide > r["xi"] + 0.1


def test_entropy_uniform_column_has_zero_weight():
    X = np.array([[5., 1], [5, 9], [5, 4]])
    e = W.entropy_weights(X, np.array([True, True]), "sum")
    assert np.isclose(e["E"][0], 1) and np.isclose(e["w"][0], 0) and np.isclose(e["w"].sum(), 1)


def test_critic_properties():
    d = D.machines_critic(); c = W.critic_weights(d["X"], d["benefit"])
    assert np.isclose(c["w"].sum(), 1) and (c["w"] > 0).all()
    assert np.allclose(np.diag(c["rho"]), 1)
    dup = np.column_stack([d["X"][:, 0], d["X"][:, 0] * 2, d["X"][:, 2]])      # deux colonnes parfaitement corrélées
    c2 = W.critic_weights(dup, np.array([False, False, True]))
    assert c2["w"][2] > c2["w"][0]                                              # le critère redondant est pénalisé


def test_waspas_exercice():
    d = D.machines_waspas(); r = R.waspas(d["X"], d["benefit"], d["w"])
    assert list(r["rank"]) == [4, 3, 2, 1] and np.allclose(r["Q"], (r["Q1"] + r["Q2"]) / 2)
    assert (R.ratio_norm(d["X"], d["benefit"]) <= 1 + 1e-12).all()


def test_ratio_norm_rejects_nonpositive():
    with pytest.raises(ValueError):
        R.ratio_norm(np.array([[1., 0.], [2., 3.]]), np.array([True, False]))


def test_topsis_voitures():
    d = D.voitures_topsis(); t = R.topsis(d["X"], d["benefit"], d["w"])
    assert list(t["rank"]) == [3, 1, 4, 2] and ((t["RC"] >= 0) & (t["RC"] <= 1)).all()


def test_vikor_exercice():
    d = D.machines_vikor(); v = R.vikor(d["X"], d["benefit"], d["w"])
    assert np.allclose(v["S"], [.5, .375, .5]) and np.allclose(v["R"], [.5, .25, .5])
    assert v["best"] == 1 and v["C1"] and v["C2"]


def test_outranking_dominant_alternative_wins():
    X = np.array([[10., 1, 9], [5, 5, 5], [4, 6, 3], [2, 9, 1]]); ben = np.array([True, False, True]); w = np.ones(3) / 3
    p = O.promethee(X, ben, w, ["linear"] * 3, q=np.full(3, .5), p=np.full(3, 4.))
    e = O.electre3(X, ben, w, q=np.full(3, .5), p=np.full(3, 2.), v=np.full(3, 6.))
    assert p["rank"][0] == 1 and e["rank"][0] == 1 and np.isclose(p["phi"].sum(), 0)


def test_spearman_course_matrix():
    M = RB.spearman_matrix(D.RANGS_COURS)
    assert np.isclose(M.loc["TOPSIS", "VIKOR"], .771, atol=1e-3) and np.isclose(M.loc["TOPSIS", "PROMETHEE II"], .943, atol=1e-3) and np.isclose(M.loc["VIKOR", "PROMETHEE II"], .6)
    assert RB.spearman_critical(6) == 0.886


def test_roc_course_table():
    t = S.roc_table(D.SEUIL_COURS["Score S"], (D.SEUIL_COURS["Décision"] == "Accepter").astype(int), [.75, .65, .60]).round(2)
    assert list(t["TPR"]) == [.67, 1.0, 1.0] and list(t["FPR"]) == [0.0, .33, .67] and list(t["d"]) == [.33, .33, .67] and list(t["J"]) == [.67, .67, .33]


def test_fuzzy_single_course():
    ex = D.FUZZY_SINGLE; r = F.fuzzy_single(ex["tfn"], ex["w"])
    assert np.allclose(r["S"], [.74, .85, 1.0], atol=.01) and round(r["score"], 2) == 0.86 and round(r["width"], 2) == 0.26


def test_fuzzy_arithmetic():
    A, B = (4, 5, 6, 7), (1, 2, 3, 4)
    assert F.tfn_sub(A, B) == (0, 2, 4, 6) and F.tfn_div(A, B) == (1, 5 / 3, 3, 7) and F.int_mul((-1, 2), (3, 4)) == (-4, 8)


def test_ifn_course_vote_example():
    a = (.6, .3); assert np.isclose(F.ifn_pi(a), .1)
    assert np.allclose(F.ifwa([a, a, a], [.2, .3, .5]), a)           # IFWA idempotent
    assert F.ifn_compare((.6, .3), (.5, .4)) == 1


def test_hesitant_course_example():
    h = D.HESITANT_COURS
    assert np.isclose(F.hfn_score(h["h1"]), .45) and np.isclose(F.hfn_score(h["h2"]), .6)
    assert F.hfn_compare(h["h3"], h["h1"]) == 1                      # même score, h3 moins dispersé


def test_fuzzy_topsis_recruitment_and_if_topsis():
    ex = D.RECRUTEMENT
    Rr = np.array([[[F.ALT_SCALE[c] for c in ex["ratings"][a][k]] for a in ex["alts"]] for k in range(3)], float)
    Wt = np.array([[F.W_SCALE[c] for c in r] for r in ex["weights"]], float)
    r = F.fuzzy_topsis(Rr, Wt, np.ones(5, bool))
    assert list(r["rank"]) == [3, 1, 2] and ((r["CC"] > 0) & (r["CC"] < 1)).all()
    m = D.traitement_medical()
    Ri = np.array([[[D.IF_SCALE_ALT[c] for c in m["ratings"][a][k]] for a in m["alts"]] for k in range(3)], float)
    Wi = np.array([[D.IF_SCALE_W[c] for c in r_] for r_ in m["weights"]], float)
    o = F.if_topsis(Ri, Wi, m["sigma"], np.array(m["benefit"], bool))
    assert ((o["C"] >= 0) & (o["C"] <= 1)).all() and sorted(o["rank"]) == [1, 2, 3, 4]


def test_shapley_efficiency():
    rng = np.random.default_rng(0); bg = rng.normal(size=(30, 4)); f = lambda Z: Z @ np.array([1., -2, 0.5, 3]) + Z[:, 0] * Z[:, 1]
    x = rng.normal(size=4); phi = H.exact_shapley(f, x, bg)
    assert np.isclose(phi.sum(), f(x[None])[0] - f(bg).mean())       # propriété d'efficience
    lin = lambda Z: Z @ np.array([1., -2, 0.5, 3]); phi_l = H.exact_shapley(lin, x, bg)
    assert np.allclose(phi_l, np.array([1., -2, .5, 3]) * (x - bg.mean(0)))   # modèle linéaire : φ_j = a_j (x_j − E[x_j])


def test_cv_extraction_matches_course_example():
    e = H.extract_scores("Ingénieur logiciel avec 7 ans d'expérience en développement Python et Java, management d'une équipe de 5 personnes, certifié PMP. Prétentions : 85 000 €/an.")
    assert list(e["Score /10"])[:3] == [8, 7, 3]


# ---- ajouts : consensus, points de bascule, rapport, grands jeux de données
def _big():
    d = D.energie_village(); return d
def test_consensus_and_tipping_and_report():
    from mcdm_core import report as RP
    d = D.energie_village(); X = np.asarray(d["X"], float); ben = d["benefit"]; w = np.asarray(d["w"], float)
    alts = list(d["alts"]); crit = list(d["crit"])
    reg, _ = RB.usable_methods(X, ben, w)
    ranks = RB.compare_methods(X, ben, w, list(reg), reg); ranks.index = alts
    cons = RB.consensus(ranks)
    assert len(cons) == len(alts) and cons["Points de Borda"].iloc[0] == cons["Points de Borda"].max()
    tips = RB.tipping_points(reg["TOPSIS"], X, ben, w, steps=60)
    assert len(tips) == len(crit) and (tips["seuil_bas"].dropna() <= tips["poids"][tips["seuil_bas"].notna()]).all()
    html = RP.build_report("t", "d", pd.DataFrame(X, index=alts, columns=crit), ben, w, "test", ranks, cons, tips, crit, alts, "verdict")
    assert html.lower().startswith("<!doctype html") and alts[0] in html
def test_large_datasets_sizes():
    for f in (D.energie_village, D.portefeuille, D.sites_hopital, D.ml_models):
        d = f(); m, n = np.asarray(d["X"]).shape; assert 7 <= n <= 10 and m >= 12
