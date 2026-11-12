"""Datasets de démonstration. `source` = « cours » (valeurs issues des diapositives) ou « créé » (jeu pédagogique inventé).

Exécuter `python -m mcdm_core.datasets` écrit tous les jeux en CSV dans ./data (transparence / réutilisation).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path


def _ds(name, alts, crit, X, benefit, w=None, units=None, source="créé", desc=""):
    return dict(name=name, alts=list(alts), crit=list(crit), X=np.array(X, float), benefit=np.array(benefit, bool),
                w=None if w is None else np.array(w, float), units=units, source=source, desc=desc)


def df(ds) -> pd.DataFrame:
    return pd.DataFrame(ds["X"], index=ds["alts"], columns=ds["crit"])


# ================================================================= COURS
def machines_critic():   # diapo « CRITIC — exercice »
    return _ds("Machines de découpe (CRITIC)", ["A1", "A2", "A3", "A4"],
               ["C1 Épaisseur max (−)", "C2 Largeur de coupe (−)", "C3 Qualité de coupe (+)", "C4 Coût maintenance (−)"],
               [[30, .1, 1, 20], [100, .7, 1, 40], [50, 1, 2, 10], [300, 2, 3, 35]], [0, 0, 1, 0], source="cours",
               desc="Exercice du cours (méthodes objectives). Types indiqués sur la diapo : −, −, +, −.")


def machines_waspas():   # diapo « WASPAS — exercice »
    return _ds("Choix de machines (WSM / WPM / WASPAS)", ["A1", "A2", "A3", "A4"], ["C1 (−)", "C2 (−)", "C3 (+)", "C4 (−)", "C5 (+)"],
               [[.035, 847, .335, 1.76, .59], [.027, 834, .335, 1.68, .665], [.037, 808, .59, 2.4, .5], [.028, 821, .5, 1.59, .41]],
               [0, 0, 1, 0, 1], w=[.331, .181, .369, .072, .047], source="cours", desc="Exercice du cours, poids fournis.")


def voitures_topsis():   # diapo « TOPSIS — exercice »
    return _ds("Choix d'une voiture (TOPSIS)", ["M1", "M2", "M3", "M4"], ["C1 Style (+)", "C2 Fiabilité (+)", "C3 Éco. carburant (+)", "C4 Coût (−)"],
               [[7, 9, 9, 8], [8, 7, 8, 7], [9, 6, 8, 9], [6, 7, 8, 6]], [1, 1, 1, 0], w=[.2, .1, .4, .3], source="cours", desc="Exercice du cours.")


def machines_vikor():    # diapo « VIKOR — exercice »
    return _ds("Achat d'une machine (VIKOR)", ["A1", "A2", "A3"], ["C1 (−)", "C2 (+)"], [[1, 3000], [2, 3750], [5, 4500]], [0, 1],
               w=[.5, .5], source="cours", desc="Exercice du cours : deux critères de même poids.")


AHP_VOITURE = dict(
    criteria=["Coût", "Confort", "Sécurité"], alts=["Voiture 1", "Voiture 2"],
    crit_upper=[[1, 7, 3], [0, 1, 1 / 3], [0, 0, 1]],                       # a12=7, a13=3, a23=1/3 (diapo)
    alt_matrices={"Coût": [[1, 7], [1 / 7, 1]], "Confort": [[1, .2], [5, 1]], "Sécurité": [[1, 1 / 9], [9, 1]]},
    source="cours")

BWM_FOURNISSEUR = dict(criteria=["C1 Qualité", "C2 Prix", "C3 Délai de livraison", "C4 Engagement RSE"], best=0, worst=3,
                       bo=[1, 3, 4, 8], ow=[8, 5, 3, 1], source="cours")

RANGS_COURS = pd.DataFrame({"TOPSIS": [2, 1, 3, 5, 4, 6], "VIKOR": [1, 2, 4, 3, 5, 6], "PROMETHEE II": [2, 1, 3, 6, 4, 5]},
                           index=[f"A{i}" for i in range(1, 7)])

SEUIL_COURS = pd.DataFrame({"Alternative": [f"A{i}" for i in range(1, 7)], "Score S": [.85, .78, .71, .69, .62, .54],
                            "Décision": ["Accepter"] * 3 + ["Rejeter"] * 3})

FUZZY_SINGLE = dict(criteria=["C1 : ROA", "C2 : Liquidité"], tfn=[(1.0, 1.2, 1.5), (110, 120, 130)], w=[.6, .4], source="cours")

RECRUTEMENT = dict(
    alts=["A1", "A2", "A3"], dms=["DM1", "DM2", "DM3"],
    crit=["C1 Stabilité émotionnelle", "C2 Communication orale", "C3 Personnalité", "C4 Expérience passée", "C5 Confiance en soi"],
    ratings={"A1": [["MG", "G", "F", "VG", "F"], ["G", "MG", "G", "G", "F"], ["MG", "F", "G", "VG", "F"]],
             "A2": [["G", "VG", "VG", "VG", "VG"], ["G", "VG", "VG", "VG", "MG"], ["MG", "VG", "G", "VG", "G"]],
             "A3": [["VG", "MG", "G", "G", "G"], ["G", "G", "MG", "VG", "G"], ["F", "VG", "VG", "MG", "MG"]]},
    weights=[["H", "VH", "VH", "VH", "M"], ["VH", "VH", "H", "VH", "MH"], ["MH", "VH", "H", "VH", "MH"]], source="cours")

HESITANT_COURS = {"h1": [.2, .3, .5, .8], "h2": [.4, .6, .8], "h3": [.3, .45, .6]}

ACTIONS_DYN_COURS = pd.DataFrame({"Rendement r (%) ↑": [8, 6, 7], "Volatilité σ ↓": [.25, .15, .20], "Liquidité L ↑": [100, 80, 120]},
                                 index=["A1 Tech", "A2 Énergie", "A3 Bancaire"])


# ================================================================= CRÉÉS (volumes élargis : 12 à 16 alternatives, 8 à 10 critères)
def ml_models():
    rng = np.random.default_rng(11)
    names = ["SVM à noyau RBF", "Forêt aléatoire", "Régression logistique", "XGBoost", "LightGBM", "CatBoost", "k plus proches voisins",
             "Naive Bayes", "Arbre CART", "Perceptron multicouche", "Gradient Boosting", "AdaBoost", "Analyse discriminante", "Extra Trees"]
    q = np.array([.60, .72, .45, .88, .90, .89, .35, .25, .30, .70, .80, .55, .40, .70])          # qualité latente
    train = np.array([42, 18, 1.2, 25, 8, 30, 0.1, 0.05, 0.4, 120, 55, 9, 0.2, 12])               # s
    infer = np.array([9, 14, 0.2, 3, 1.5, 2, 35, 0.3, 0.1, 1.2, 4, 3, 0.2, 12])                   # ms
    n = len(names); noise = lambda s: rng.normal(0, s, n)
    acc = .74 + .19 * q + noise(.006); rec = acc - .04 + noise(.015); pre = acc - .02 + noise(.015)
    f1 = 2 * pre * rec / (pre + rec); auc = .80 + .18 * q + noise(.006); kap = 2 * acc - 1 - .06 + noise(.01)
    X = np.column_stack([acc, rec, pre, f1, auc, kap, train, infer]).round(3)
    return _ds("Sélection d'un modèle d'apprentissage", names,
               ["Accuracy", "Recall", "Precision", "F1", "AUC", "Kappa", "Temps d'entraînement (s)", "Temps d'inférence (ms)"], X,
               [1, 1, 1, 1, 1, 1, 0, 0], w=[.14, .14, .10, .14, .16, .08, .12, .12],
               desc="14 modèles, 8 critères : six métriques de performance fortement corrélées et deux critères de coût de calcul. "
                    "Valeurs générées (graine 11) à partir d'un niveau de qualité latent par modèle ; elles ne proviennent pas d'une expérience réelle.")


def energie_village():
    names = ["PV + batteries", "PV + diesel de secours", "Éolien + batteries", "Mini-hydroélectrique", "Hybride PV-éolien-diesel", "Groupe diesel",
             "Extension du réseau", "Biomasse (gazéification)", "Biogaz", "Géothermie", "Micro-réseau PV + hydrogène", "Solaire thermodynamique (CSP)",
             "Pile à combustible au gaz", "PV + éolien + batteries Li-ion"]
    X = [[620, 18, 45, .80, 8, 7, 8, 9, 2.5], [410, 42, 520, .90, 7, 6, 6, 9, 2.0], [700, 32, 40, .68, 6, 5, 12, 8, 1.2], [1150, 14, 20, .88, 7, 8, 24, 9, .8],
         [750, 24, 320, .93, 9, 9, 11, 8, 3.1], [200, 90, 2400, .97, 4, 3, 3, 9, .1], [980, 12, 1600, .99, 7, 2, 18, 9, .3], [540, 40, 150, .85, 7, 11, 14, 7, 1.5],
         [610, 38, 130, .82, 6, 9, 16, 7, 1.0], [1900, 20, 60, .94, 5, 6, 36, 6, .6], [1350, 28, 25, .86, 5, 5, 20, 5, 3.8], [1600, 26, 35, .83, 4, 6, 30, 6, 6.0],
         [880, 55, 900, .95, 5, 3, 10, 6, .2], [830, 22, 55, .87, 8, 8, 13, 8, 3.5]]
    return _ds("Électrification d'un village isolé", names,
               ["Investissement (k€)", "Exploitation (k€/an)", "CO₂ cycle de vie (t/an)", "Fiabilité (0-1)", "Acceptabilité sociale (0-10)", "Emplois locaux",
                "Délai de mise en service (mois)", "Maturité technologique (TRL)", "Emprise au sol (ha)"], X, [0, 0, 0, 1, 1, 1, 0, 1, 0],
               w=[.18, .10, .14, .14, .10, .08, .08, .10, .08],
               desc="14 filières, 9 critères. Cas évoqué en introduction du cours (solaire, éolien, hybride). Ordres de grandeur plausibles mais inventés. "
                    "Les émissions sont celles du cycle de vie, donc jamais nulles. Le CO₂ est le candidat naturel au veto d'ELECTRE III.")


def portefeuille():
    rng = np.random.default_rng(5)
    names = ["Atlas Digital", "Sahara Énergie", "Banque du Détroit", "PharmaVie", "AgriSud", "Télécom Maghreb", "BTP Occident", "Distri+",
             "Aéro Systèmes", "Assurances Horizon", "Cimentière du Nord", "Logis Immobilier", "Mines de l'Atlas", "Transport Rapide", "Cosmétique Rif", "Hôtels Océan"]
    n = len(names); g = rng.uniform(.1, .95, n); r = np.clip(.75 * (1 - g) * .5 + rng.uniform(.1, .7, n), .05, .95); v = np.clip(.6 * g + rng.uniform(.0, .5, n), .05, .95)
    e = lambda s: rng.normal(0, s, n)
    X = np.column_stack([4 + 9 * g + e(.8), 3 + 7 * r + e(.4), 1 + 9 * rng.uniform(.1, 1, n), .5 + 1.0 * r + e(.05), 1 + 5.5 * g + e(.3), 6 + 15 * g + e(.8),
                         1 + 3.5 * v + e(.15), .4 + 3.4 * v + e(.1), 3 + 8 * g + e(.5), 5.5 - 3.5 * g + e(.4)]).round(2)
    return _ds("Sélection d'actions", names,
               ["C1 Rendement moyen (%)", "C2 Écart-type (%)", "C3 Volume (M)", "C4 Bêta", "C5 BPA", "C6 ROE (%)", "C7 P/B", "C8 P/S", "C9 ROA (%)", "C10 Rendement du dividende (%)"],
               X, [1, 0, 1, 0, 1, 1, 0, 0, 1, 1], w=[.16, .12, .06, .10, .09, .12, .08, .07, .10, .10],
               desc="16 sociétés fictives, 10 critères. Les neuf premiers reprennent le cadre C1 à C9 cité dans le cours pour la finance (Vuković et al., 2020) ; "
                    "le dixième est ajouté. Valeurs simulées (graine 5) à partir d'un profil croissance / risque / valorisation latent.")


def sites_hopital():
    rng = np.random.default_rng(23)
    names = [f"Site {c}" for c in "ABCDEFGHIJKL"]; n = len(names)
    pop = rng.uniform(80, 420, n); acc = rng.uniform(4, 9.5, n); cen = rng.uniform(0, 1, n)           # centralité latente
    X = np.column_stack([pop, 55 - 4 * acc + rng.normal(0, 4, n), 20 + 90 * cen + rng.normal(0, 8, n), 140 + 25 * rng.uniform(0, 1, n) + rng.normal(0, 6, n),
                         acc, rng.uniform(0, 8, n), 3 + 5 * cen + rng.normal(0, 1, n), rng.uniform(3, 40, n)])
    X = np.clip(X, [50, 8, 10, 120, 1, 0.5, 1, 2], [500, 70, 130, 190, 10, 9, 10, 45]).round(1)
    return _ds("Implantation d'un hôpital régional", names,
               ["Population couverte (milliers)", "Temps d'accès moyen (min)", "Coût du foncier (M DH)", "Coût de construction (M DH)", "Accessibilité routière (0-10)",
                "Risque d'inondation (0-10)", "Disponibilité du personnel (0-10)", "Distance à l'hôpital le plus proche (km)"], X, [1, 0, 0, 0, 1, 0, 1, 1],
               w=[.18, .16, .08, .10, .12, .12, .12, .12], desc="12 sites candidats, 8 critères. Valeurs simulées (graine 23), à titre d'illustration d'un problème de localisation.")


def _saaty(x: float) -> float:
    grid = np.array([1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    return float(grid[np.argmin(np.abs(np.log(grid) - np.log(np.clip(x, 1/9, 9))))])


def _pair_upper(vals) -> list:
    n = len(vals); U = np.zeros((n, n))
    for i in range(n):
        U[i, i] = 1
        for j in range(i + 1, n): U[i, j] = _saaty(vals[i] / vals[j])
    return U.tolist()


def ahp_site():
    crit = ["Coût d'implantation", "Accès aux matières premières", "Main-d'œuvre qualifiée", "Infrastructures", "Fiscalité", "Risque environnemental", "Proximité des clients", "Potentiel d'extension"]
    alts = ["Kénitra", "Tanger", "Casablanca", "Settat", "Berrechid", "Oujda"]
    wlat = np.array([9, 5, 6, 7, 3, 4, 8, 2.5])
    rng = np.random.default_rng(3)
    S = np.clip(rng.uniform(2, 9, (len(alts), len(crit))), 1, 9)
    return dict(criteria=crit, alts=alts, crit_upper=_pair_upper(wlat), alt_matrices={c: _pair_upper(S[:, j]) for j, c in enumerate(crit)}, source="créé")


def bwm_large():
    crit = ["Qualité", "Prix", "Délai de livraison", "Engagement RSE", "Solidité financière", "Flexibilité", "Innovation", "Proximité géographique"]
    wl = np.array([9, 5.5, 4.5, 1.5, 3.5, 2.5, 2.2, 1.0]); b, w_ = int(wl.argmax()), int(wl.argmin())
    bo = [float(np.clip(round(wl[b] / x), 1, 9)) for x in wl]; ow = [float(np.clip(round(x / wl[w_]), 1, 9)) for x in wl]
    return dict(criteria=crit, best=b, worst=w_, bo=bo, ow=ow, source="créé")


def _levels(rng, latent, noise, scale_keys):
    k = len(scale_keys); idx = np.clip(np.rint((latent + rng.normal(0, noise)) * (k - 1)), 0, k - 1).astype(int); return scale_keys[idx]


def recrutement_large():
    rng = np.random.default_rng(31); keys = np.array(list(ALT_KEYS)); wkeys = np.array(list(W_KEYS))
    alts = [f"Candidat {i}" for i in range(1, 9)]; dms = ["DRH", "Responsable technique", "Manager", "Psychologue"]
    crit = ["Stabilité émotionnelle", "Communication orale", "Personnalité", "Expérience passée", "Confiance en soi", "Compétences techniques", "Esprit d'équipe"]
    q = rng.uniform(.2, .95, (len(alts), len(crit))); imp = rng.uniform(.45, 1, len(crit))
    ratings = {a: [[str(_levels(rng, q[i, j], .12, keys)) for j in range(len(crit))] for _ in dms] for i, a in enumerate(alts)}
    weights = [[str(_levels(rng, imp[j], .1, wkeys)) for j in range(len(crit))] for _ in dms]
    return dict(alts=alts, dms=dms, crit=crit, ratings=ratings, weights=weights, source="créé")


ALT_KEYS = ["VP", "P", "MP", "F", "MG", "G", "VG"]
W_KEYS = ["VL", "L", "ML", "M", "MH", "H", "VH"]


def traitement_large():
    rng = np.random.default_rng(37); keys = np.array(["TF", "F", "M", "E", "TE"]); wkeys = np.array(["NI", "PI", "M", "I", "TI"])
    alts = ["Chirurgie", "Chimiothérapie", "Radiothérapie", "Immunothérapie", "Thérapie ciblée", "Hormonothérapie", "Curiethérapie", "Protonthérapie", "Chirurgie robotique", "Traitement combiné"]
    dms = ["Oncologue", "Chirurgien", "Pharmacien", "Radiothérapeute"]
    crit = ["Taux de survie (+)", "Risque de récidive (−)", "Effets secondaires (−)", "Coût (−)", "Qualité de vie (+)", "Durée d'hospitalisation (−)", "Accessibilité (+)", "Niveau de preuve (+)"]
    benefit = [1, 0, 0, 0, 1, 0, 1, 1]
    q = rng.uniform(.15, .95, (len(alts), len(crit))); imp = rng.uniform(.4, 1, len(crit))
    ratings = {a: [[str(_levels(rng, q[i, j], .12, keys)) for j in range(len(crit))] for _ in dms] for i, a in enumerate(alts)}
    weights = [[str(_levels(rng, imp[j], .1, wkeys)) for j in range(len(crit))] for _ in dms]
    return dict(alts=alts, dms=dms, crit=crit, benefit=benefit, ratings=ratings, weights=weights, sigma=[.3, .25, .2, .25], source="créé",
                desc="10 traitements, 8 critères, 4 décideurs. Niveaux linguistiques tirés au hasard (graine 37) autour d'une valeur latente par couple traitement-critère.")


# IFN : échelle linguistique (niveaux) pour l'IF-TOPSIS — échelle usuelle de la littérature, créée pour la démonstration
IF_SCALE_ALT = {"TE": (.90, .05), "E": (.75, .15), "M": (.50, .40), "F": (.30, .60), "TF": (.10, .85)}   # (μ, ν) : niveau très élevé … très faible
IF_SCALE_W = {"TI": (.90, .05), "I": (.75, .15), "M": (.50, .40), "PI": (.30, .60), "NI": (.10, .85)}   # importance : très … non importante
IF_LABELS_ALT = {"TE": "Très élevé", "E": "Élevé", "M": "Moyen", "F": "Faible", "TF": "Très faible"}
IF_LABELS_W = {"TI": "Très importante", "I": "Importante", "M": "Moyenne", "PI": "Peu importante", "NI": "Non importante"}


def traitement_medical():
    return dict(
        alts=["T1 Chirurgie", "T2 Chimiothérapie", "T3 Radiothérapie", "T4 Immunothérapie"], dms=["Oncologue", "Chirurgien", "Pharmacien"],
        crit=["C1 Taux de survie (+)", "C2 Risque de récidive (−)", "C3 Effets secondaires (−)", "C4 Coût (−)", "C5 Qualité de vie (+)"],
        benefit=[1, 0, 0, 0, 1],
        ratings={"T1 Chirurgie": [["E", "M", "F", "M", "M"], ["TE", "M", "M", "F", "M"], ["E", "E", "M", "M", "F"]],
                 "T2 Chimiothérapie": [["M", "E", "E", "M", "F"], ["M", "M", "TE", "E", "F"], ["E", "M", "E", "M", "F"]],
                 "T3 Radiothérapie": [["M", "M", "M", "E", "M"], ["E", "M", "M", "E", "E"], ["M", "E", "M", "E", "M"]],
                 "T4 Immunothérapie": [["TE", "F", "F", "TE", "E"], ["E", "F", "F", "TE", "TE"], ["TE", "F", "M", "TE", "E"]]},
        weights=[["TI", "I", "I", "PI", "I"], ["TI", "I", "M", "M", "I"], ["I", "TI", "I", "PI", "M"]],
        sigma=[.4, .35, .25], source="créé",
        desc="Application « Médecine » du cours (sélection d'un traitement). Chaque décideur exprime le NIVEAU du traitement sur chaque critère par un nombre flou "
             "intuitionnistes (μ, ν) issu d'une échelle linguistique ; pour les critères (−) (récidive, effets secondaires, coût), un niveau élevé est défavorable. "
             "Valeurs inventées.")


def fournisseurs_ml(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Jeu synthétique pour le volet MCDM + IA : n fournisseurs × 8 critères (graine fixe)."""
    rng = np.random.default_rng(seed)
    q = np.clip(rng.normal(7, 1.5, n), 1, 10); lead = np.clip(rng.gamma(4, 4, n), 3, 60)
    price = np.clip(100 - 4 * (q - 7) + rng.normal(0, 10, n), 50, 160); rse = np.clip(rng.normal(5.5, 2, n) + 0.25 * (q - 7), 0, 10)
    risk = np.clip(rng.normal(4, 1.6, n) - 0.2 * (rse - 5), 0, 10); fin = np.clip(rng.normal(6.5, 1.7, n) + 0.3 * (q - 7), 0, 10)
    cap = np.clip(rng.lognormal(3.2, .6, n), 3, 400); dft = np.clip(rng.gamma(2, 150, n) / (1 + .25 * (q - 5).clip(0)), 20, 2500)
    return pd.DataFrame({"Qualité (0-10)": q.round(2), "Prix (€/u)": price.round(1), "Délai (jours)": lead.round(1), "RSE (0-10)": rse.round(2),
                         "Risque (0-10)": risk.round(2), "Solidité financière (0-10)": fin.round(2), "Capacité (k u/mois)": cap.round(1), "Taux de défaut (ppm)": dft.round(0)})


FOURNISSEURS_BENEFIT = [1, 0, 0, 1, 0, 1, 1, 0]
FOURNISSEURS_W_SUBJ = [.24, .16, .12, .08, .12, .08, .10, .10]


CATALOGUE = {
    "machines_critic": machines_critic, "machines_waspas": machines_waspas, "voitures_topsis": voitures_topsis,
    "machines_vikor": machines_vikor, "ml_models": ml_models, "energie_village": energie_village, "portefeuille": portefeuille, "sites_hopital": sites_hopital,
}


def export_all(out: str = "data") -> None:
    p = Path(out); p.mkdir(exist_ok=True)
    for k, f in CATALOGUE.items():
        d = f(); t = df(d); t.loc["TYPE (1=bénéfice, 0=coût)"] = d["benefit"].astype(int)
        if d["w"] is not None: t.loc["POIDS"] = d["w"]
        t.to_csv(p / f"{k}.csv", encoding="utf-8-sig")
    RANGS_COURS.to_csv(p / "rangs_cours.csv", encoding="utf-8-sig"); SEUIL_COURS.to_csv(p / "seuil_scores.csv", index=False, encoding="utf-8-sig")
    ACTIONS_DYN_COURS.to_csv(p / "actions_dynamiques.csv", encoding="utf-8-sig"); fournisseurs_ml().to_csv(p / "fournisseurs_ml.csv", index=False, encoding="utf-8-sig")
    from .hybrid import simulate_series
    simulate_series().to_csv(p / "series_actions.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    export_all()
    print("CSV écrits dans ./data")
