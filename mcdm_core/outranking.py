"""Méthodes de surclassement (sans compensation) : PROMETHEE I/II et ELECTRE III.

NB : dans le support de cours, ces deux méthodes n'ont qu'une page de titre. Elles sont implémentées ici
selon leurs formulations standard (Brans & Vincke 1985 ; Roy 1978, distillation de Roy & Bouyssou 1993).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

PREF_TYPES = {
    "usual": "Usuelle (critère vrai)",
    "ushape": "En U (seuil q)",
    "vshape": "En V (seuil p)",
    "level": "Par paliers (q, p)",
    "linear": "Linéaire (q, p)",
    "gaussian": "Gaussienne (σ)",
}


def pref_function(kind: str, d: np.ndarray, q: float = 0.0, p: float = 1.0, s: float = 1.0) -> np.ndarray:
    d = np.asarray(d, float)
    pos = np.maximum(d, 0)
    if kind == "usual":
        return (d > 0).astype(float)
    if kind == "ushape":
        return (d > q).astype(float)
    if kind == "vshape":
        return np.clip(pos / p, 0, 1) if p > 0 else (d > 0).astype(float)
    if kind == "level":
        return np.where(d <= q, 0.0, np.where(d <= p, 0.5, 1.0))
    if kind == "linear":
        den = max(p - q, 1e-12)
        return np.where(d <= q, 0.0, np.where(d >= p, 1.0, (d - q) / den))
    if kind == "gaussian":
        return np.where(d <= 0, 0.0, 1 - np.exp(-(d ** 2) / (2 * s ** 2)))
    raise ValueError(kind)


def promethee(X: np.ndarray, benefit: np.ndarray, w: np.ndarray, kinds: list[str],
              q: np.ndarray | None = None, p: np.ndarray | None = None, s: np.ndarray | None = None) -> dict:
    X = np.asarray(X, float)
    m, n = X.shape
    q = np.zeros(n) if q is None else np.asarray(q, float)
    p = np.ones(n) if p is None else np.asarray(p, float)
    s = np.ones(n) if s is None else np.asarray(s, float)
    w = np.asarray(w, float) / np.sum(w)
    pi = np.zeros((m, m))
    for j in range(n):
        diff = X[:, [j]] - X[:, [j]].T                    # g_j(a) − g_j(b)
        if not benefit[j]:
            diff = -diff
        Pj = pref_function(kinds[j], diff, q[j], p[j], s[j])
        np.fill_diagonal(Pj, 0)
        pi += w[j] * Pj
    phi_p = pi.sum(1) / (m - 1)
    phi_m = pi.sum(0) / (m - 1)
    phi = phi_p - phi_m
    rel = np.full((m, m), "R", dtype=object)               # P1 : relation partielle
    for a in range(m):
        for b in range(m):
            if a == b:
                rel[a, b] = "="; continue
            ge_p, le_m = phi_p[a] >= phi_p[b] - 1e-12, phi_m[a] <= phi_m[b] + 1e-12
            eq = abs(phi_p[a] - phi_p[b]) < 1e-12 and abs(phi_m[a] - phi_m[b]) < 1e-12
            if eq: rel[a, b] = "I"
            elif ge_p and le_m: rel[a, b] = "P"            # a surclasse b
            elif phi_p[b] >= phi_p[a] - 1e-12 and phi_m[b] <= phi_m[a] + 1e-12: rel[a, b] = "P-"
            else: rel[a, b] = "R"
    rank = pd.Series(phi).rank(ascending=False, method="min").astype(int).to_numpy()
    return dict(pi=pi, phi_plus=phi_p, phi_minus=phi_m, phi=phi, relation=rel, rank=rank)


# ----------------------------------------------------------------- ELECTRE III
def _s(l: float) -> float:
    """Seuil de discrimination de la distillation : s(λ) = 0,3 − 0,15 λ."""
    return 0.3 - 0.15 * l


def electre3_credibility(X, benefit, w, q, p, v):
    X = np.asarray(X, float)
    m, n = X.shape
    w = np.asarray(w, float) / np.sum(w)
    C = np.zeros((m, m)); cj = np.zeros((n, m, m)); dj = np.zeros((n, m, m))
    for j in range(n):
        delta = X[:, [j]].T - X[:, [j]]                    # delta[a,b] = g_j(b) − g_j(a) (a « moins bon » si > 0)
        if not benefit[j]:
            delta = -delta
        c = np.where(delta <= q[j], 1.0, np.where(delta >= p[j], 0.0, (p[j] - delta) / max(p[j] - q[j], 1e-12)))
        d = np.where(delta <= p[j], 0.0, np.where(delta >= v[j], 1.0, (delta - p[j]) / max(v[j] - p[j], 1e-12)))
        cj[j], dj[j] = c, d
        C += w[j] * c
    sigma = C.copy()
    for a in range(m):
        for b in range(m):
            if a == b:
                sigma[a, b] = 1.0; continue
            F = dj[:, a, b] > C[a, b]
            if F.any():
                sigma[a, b] = C[a, b] * np.prod((1 - dj[F, a, b]) / (1 - C[a, b]))
    return C, sigma, cj, dj


def _distill(sigma: np.ndarray, descending: bool) -> list[list[int]]:
    """Distillation descendante (on extrait les meilleures) ou ascendante (les moins bonnes). Renvoie des classes."""
    remaining = list(range(sigma.shape[0]))
    classes = []
    while remaining:
        sub = list(remaining)
        for _ in range(50):
            if len(sub) == 1: break
            S = sigma[np.ix_(sub, sub)].copy(); np.fill_diagonal(S, -1)
            l0 = S.max()
            cut = l0 - _s(l0)
            cand = S[(S < cut) & (S >= 0)]
            if cand.size == 0: break                       # plus aucune coupure possible → classe d'indifférence
            l1 = cand.max()
            qual = {}
            for a in sub:
                st = sum(1 for b in sub if b != a and sigma[a, b] > l1 and sigma[a, b] > sigma[b, a] + _s(sigma[a, b]))
                wk = sum(1 for b in sub if b != a and sigma[b, a] > l1 and sigma[b, a] > sigma[a, b] + _s(sigma[b, a]))
                qual[a] = st - wk
            target = max(qual.values()) if descending else min(qual.values())
            new = [a for a in sub if qual[a] == target]
            if len(new) == len(sub): break
            sub = new
        classes.append(sub)
        remaining = [a for a in remaining if a not in sub]
    return classes


def electre3(X, benefit, w, q, p, v) -> dict:
    X = np.asarray(X, float)
    m = X.shape[0]
    C, sigma, cj, dj = electre3_credibility(X, benefit, w, np.asarray(q, float), np.asarray(p, float), np.asarray(v, float))
    desc = _distill(sigma, True)                           # meilleur → pire
    asc = _distill(sigma, False)[::-1]                     # extrait pire → meilleur, on inverse
    r_d, r_a = np.zeros(m, int), np.zeros(m, int)
    for k, cl in enumerate(desc, 1):
        for a in cl: r_d[a] = k
    for k, cl in enumerate(asc, 1):
        for a in cl: r_a[a] = k
    rel = np.full((m, m), "R", dtype=object)
    for a in range(m):
        for b in range(m):
            if a == b: rel[a, b] = "="
            elif r_d[a] == r_d[b] and r_a[a] == r_a[b]: rel[a, b] = "I"
            elif (r_d[a] <= r_d[b] and r_a[a] <= r_a[b]): rel[a, b] = "P"
            elif (r_d[b] <= r_d[a] and r_a[b] <= r_a[a]): rel[a, b] = "P-"
    # Rang médian (moyenne des deux préordres) pour obtenir un classement total lisible
    mean_rank = (r_d + r_a) / 2
    rank = pd.Series(mean_rank).rank(method="min").astype(int).to_numpy()
    return dict(C=C, sigma=sigma, cj=cj, dj=dj, rank_desc=r_d, rank_asc=r_a, relation=rel,
                mean_rank=mean_rank, rank=rank, classes_desc=desc, classes_asc=asc)
