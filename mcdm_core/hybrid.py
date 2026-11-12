"""MCDM & IA : classification supervisée (scénario 1), mise à jour des poids par Shapley (scénario 2),
MCDM dynamique (prévision puis MCDM), extraction de scores depuis du texte (simulateur de « LLM extracteur »)."""
from __future__ import annotations
import itertools, math, re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score


# ----------------------------------------------------------------- Scénario 1 : MCDM → labels → classifieur
def label_by_quantiles(scores, labels=("D", "C", "B", "A")) -> np.ndarray:
    """Étiquette A (meilleurs) … D (moins bons) par quartiles du score MCDM."""
    q = pd.qcut(pd.Series(scores).rank(method="first"), len(labels), labels=list(labels))
    return q.astype(str).to_numpy()


def train_classifier(X: pd.DataFrame, y, seed: int = 0) -> dict:
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, min_samples_leaf=2)
    cv = cross_val_score(clf, X, y, cv=5)
    clf.fit(X, y)
    imp = pd.Series(clf.feature_importances_, index=X.columns)
    return dict(model=clf, cv_mean=float(cv.mean()), cv_std=float(cv.std()), importances=imp)


# ----------------------------------------------------------------- Scénario 2 : Shapley exact (sans dépendance)
def exact_shapley(predict, x: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Valeurs de Shapley exactes (espérance marginale sur `background`) pour n variables (n ≤ ~10).

    v(S) = E_b[ f(x_S, b_{\\S}) ] ;  φ_j = Σ_{S ⊆ N\\{j}} |S|!(n−|S|−1)!/n! · (v(S∪{j}) − v(S)).
    Même quantité que shap.KernelExplainer / TreeExplainer(interventional) — mais sans installer `shap`.
    """
    n = len(x); subsets = list(itertools.product([0, 1], repeat=n))
    B = len(background); batch = np.empty((len(subsets) * B, n))
    for s_idx, mask in enumerate(subsets):
        mask = np.array(mask, bool)
        blk = background.copy(); blk[:, mask] = x[mask]; batch[s_idx * B:(s_idx + 1) * B] = blk
    pred = np.asarray(predict(batch)).reshape(len(subsets), B).mean(1)
    v = {mask: pred[i] for i, mask in enumerate(subsets)}
    phi = np.zeros(n)
    for j in range(n):
        for mask in subsets:
            if mask[j]: continue
            k = sum(mask); wgt = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            with_j = list(mask); with_j[j] = 1
            phi[j] += wgt * (v[tuple(with_j)] - v[mask])
    return phi


def shap_weight_update(X: pd.DataFrame, score, w_mcdm, alpha: float = 0.5, n_explain: int = 60,
                       n_background: int = 40, seed: int = 0) -> dict:
    """Étapes 3-5 du cours : Ŝ = f(x) (RF) ; φ_j par Shapley ; w_new = α·w + (1−α)·Σ|φ_j| normalisé."""
    rng = np.random.default_rng(seed)
    reg = RandomForestRegressor(n_estimators=120, random_state=seed, min_samples_leaf=2).fit(X.to_numpy(float), score)
    Xv = X.to_numpy(float)
    bg = Xv[rng.choice(len(Xv), min(n_background, len(Xv)), replace=False)]
    idx = rng.choice(len(Xv), min(n_explain, len(Xv)), replace=False)
    phis = np.array([exact_shapley(reg.predict, Xv[i], bg) for i in idx])
    g = np.abs(phis).mean(0); g = g / g.sum()
    w_new = alpha * np.asarray(w_mcdm, float) + (1 - alpha) * g
    return dict(model=reg, r2=float(reg.score(X.to_numpy(float), score)), phi=phis, idx=idx, shap_importance=g, w_new=w_new / w_new.sum())


# ----------------------------------------------------------------- MCDM dynamique
DYN_CRITERIA = ["rendement", "volatilite", "liquidite", "beta", "sharpe", "dividende", "drawdown"]
DYN_LABELS = {"rendement": "Rendement (%)", "volatilite": "Volatilité", "liquidite": "Liquidité", "beta": "Bêta", "sharpe": "Ratio de Sharpe",
              "dividende": "Dividende (%)", "drawdown": "Perte maximale (%)"}
DYN_BENEFIT = [True, False, True, False, True, True, False]
DYN_NAMES = ["Atlas Digital", "Sahara Énergie", "Banque du Détroit", "PharmaVie", "AgriSud", "Télécom Maghreb", "BTP Occident", "Distri+", "Aéro Systèmes", "Assurances Horizon"]


def simulate_series(T: int = 60, seed: int = 7) -> pd.DataFrame:
    """Séries mensuelles synthétiques : 10 actions x 7 critères (niveau + tendance + saisonnalité + bruit)."""
    rng = np.random.default_rng(seed); t = np.arange(T); rows = []
    base = {"rendement": (7, 2.0), "volatilite": (.19, .05), "liquidite": (95, 25), "beta": (1.0, .3), "sharpe": (.6, .25), "dividende": (3.2, 1.2), "drawdown": (14, 5)}
    noise = {"rendement": .6, "volatilite": .015, "liquidite": 5, "beta": .05, "sharpe": .06, "dividende": .15, "drawdown": 1.0}
    for name in DYN_NAMES:
        for c in DYN_CRITERIA:
            m, sd = base[c]; level = max(m + rng.normal(0, sd), 0.05 * m); drift = rng.normal(0, .5 * sd)
            ph = rng.uniform(0, 6.28); amp = noise[c] * 1.2
            ser = level + drift * t / T + amp * np.sin(t / 5 + ph) + rng.normal(0, noise[c], T)
            base_series = np.maximum(ser, 0.02 * m) if c != "rendement" else ser
            rows.append(pd.DataFrame({"action": name, "critere": c, "t": t, "valeur": base_series}))
    return pd.concat(rows).pivot_table(index=["action", "t"], columns="critere", values="valeur").reset_index()[["action", "t"] + DYN_CRITERIA]


def forecast_next(y, method: str = "holt", alpha: float = 0.4, beta: float = 0.2, window: int = 6) -> float:
    y = np.asarray(y, float)
    if method == "naive": return float(y[-1])
    if method == "moyenne_mobile": return float(y[-window:].mean())
    if method == "ses":
        l = y[0]
        for v in y[1:]: l = alpha * v + (1 - alpha) * l
        return float(l)
    if method == "holt":
        l, b = y[0], y[1] - y[0]
        for v in y[1:]:
            ln = alpha * v + (1 - alpha) * (l + b); b = beta * (ln - l) + (1 - beta) * b; l = ln
        return float(l + b)
    if method == "ar1":
        x0, x1 = y[:-1], y[1:]; A = np.c_[np.ones_like(x0), x0]
        c, phi = np.linalg.lstsq(A, x1, rcond=None)[0]; return float(c + phi * y[-1])
    raise ValueError(method)


# ----------------------------------------------------------------- LLM extracteur (simulateur à base de règles)
CV_RUBRIC = [
    ("Expérience technique", "benefit", "2 + 0,5 par année d'expérience + 1 par certification + 0,5 par technologie citée (max 10)"),
    ("Communication / leadership", "benefit", "4 de base ; si management d'équipe : 5 + 0,4 x taille de l'équipe (max 10)"),
    ("Adéquation salariale (coût)", "cost", "score inversé : 10 x (100 k€ - prétention) / (100 k€ - 50 k€), borné à [0, 10]"),
    ("Formation", "benefit", "doctorat 10 ; master ou ingénieur 8 ; licence 6 ; sinon 4"),
    ("Langues", "benefit", "3 + 1,5 par langue citée (max 10)"),
    ("Disponibilité", "benefit", "immédiate 10 ; sinon 10 - 2 par mois de préavis ; non précisée 5"),
    ("Réalisations", "benefit", "3 + 2 par élément cité parmi publication, brevet, open source, projet majeur (max 10)"),
]
_TECH = ["python", "java", "sql", "javascript", "c++", "docker", "aws", "react", "scala", "kubernetes"]
_CERT = ["pmp", "scrum", "itil", "aws certified", "azure", "cissp", "togaf"]
_LANG = ["français", "anglais", "espagnol", "allemand", "arabe", "chinois"]
_ACH = ["publication", "brevet", "open source", "projet majeur"]


def extract_scores(text: str) -> pd.DataFrame:
    """Simule f_LLM(T, C) → s ∈ [0,10]^n + justification. Déterministe, hors-ligne, à base de règles."""
    t = text.lower()
    yrs = [int(x) for x in re.findall(r"(\d+)\s*ans?", t)]
    years = max(yrs) if yrs else 0
    techs = [k for k in _TECH if re.search(rf"\b{re.escape(k)}\b", t)]
    certs = [k for k in _CERT if k in t]
    exp = float(np.clip(2 + 0.5 * years + 1.0 * len(certs) + 0.5 * len(techs), 0, 10))
    team = re.search(r"(?:équipe|equipe|team)\s*(?:de)?\s*(\d+)", t) or re.search(r"(\d+)\s*personnes", t)
    lead = float(np.clip(5 + 0.4 * int(team.group(1)), 0, 10)) if (team or "manag" in t) and team else (5.0 if "manag" in t else 4.0)
    sal = re.search(r"(\d[\d\s .]{1,8})\s*(?:€|eur|k€)", t.replace(" ", " "))
    salary = None
    if sal:
        raw = re.sub(r"[^\d]", "", sal.group(1)); salary = int(raw) * (1000 if int(raw) < 1000 else 1)
    cost = float(np.clip(10 * (100_000 - salary) / 50_000, 0, 10)) if salary else 5.0
    degree = 10 if re.search(r"doctorat|phd", t) else 8 if re.search(r"master|ingénieur|ingenieur", t) else 6 if re.search(r"licence|bachelor", t) else 4
    langs = [k for k in _LANG if k in t]; lang = float(np.clip(3 + 1.5 * len(langs), 0, 10))
    notice = re.search(r"préavis\s*(?:de)?\s*(\d+)\s*mois", t)
    avail = 10.0 if "immédiat" in t or "immediat" in t else float(np.clip(10 - 2 * int(notice.group(1)), 0, 10)) if notice else 5.0
    ach = [k for k in _ACH if k in t]; real = float(np.clip(3 + 2 * len(ach), 0, 10))
    scores = [round(exp), round(lead), round(cost), degree, round(lang), round(avail), round(real)]
    why = [
        f"{years} an(s) d'expérience, {len(techs)} technologie(s) ({', '.join(techs) or 'aucune'}), certification(s) : {', '.join(certs) or 'aucune'}.",
        f"Management d'équipe de {team.group(1)} personne(s) détecté." if team else "Aucun élément de management détecté.",
        f"Prétention de {salary:,} €/an comparée à la fourchette 50 à 100 k€ (score inversé, critère de coût)." if salary else "Salaire non précisé, score neutre (5/10).",
        f"Niveau de formation repéré : {degree}/10.", f"Langue(s) citée(s) : {', '.join(langs) or 'aucune'}.",
        f"Préavis de {notice.group(1)} mois." if notice else ("Disponibilité immédiate." if avail == 10 else "Disponibilité non précisée."),
        f"Réalisation(s) citée(s) : {', '.join(ach) or 'aucune'}."]
    return pd.DataFrame({"Critère": [c[0] for c in CV_RUBRIC], "Score /10": scores, "Justification": why})
