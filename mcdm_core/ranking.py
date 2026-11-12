"""Classement des alternatives (méthodes additives/compensatoires) : WSM, WPM, WASPAS, TOPSIS, VIKOR."""
from __future__ import annotations
import numpy as np
import pandas as pd


def ranks_desc(scores: np.ndarray) -> np.ndarray:
    """Rang 1 = plus grand score (ex aequo → rang moyen... ici rang 'min' pour lisibilité)."""
    return pd.Series(scores).rank(ascending=False, method="min").astype(int).to_numpy()


def ranks_asc(scores: np.ndarray) -> np.ndarray:
    return pd.Series(scores).rank(ascending=True, method="min").astype(int).to_numpy()


def ideal_points(X: np.ndarray, benefit: np.ndarray):
    """Solutions idéale positive x+ et négative x− (meilleure / pire valeur de chaque critère)."""
    X = np.asarray(X, float)
    xp = np.where(benefit, X.max(0), X.min(0))
    xm = np.where(benefit, X.min(0), X.max(0))
    return xp, xm


def ratio_norm(X: np.ndarray, benefit: np.ndarray) -> np.ndarray:
    """Normalisation du cours (WSM/WPM/WASPAS) : positif x/x+ ; négatif x+/x (x+ = meilleure valeur)."""
    X = np.asarray(X, float)
    if (X <= 0).any():
        raise ValueError("La normalisation x/x⁺ (bénéfice) et x⁺/x (coût) du cours exige des valeurs strictement positives ; "
                         "WSM/WPM/WASPAS sont donc inapplicables à ces données (utiliser TOPSIS, VIKOR ou PROMETHEE).")
    xp, _ = ideal_points(X, benefit)
    return np.where(benefit, X / xp, xp / X)


# ----------------------------------------------------------------- WSM / WPM / WASPAS
def wsm(R: np.ndarray, w: np.ndarray) -> np.ndarray:
    return R @ w


def wpm(R: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.prod(R ** w, axis=1)


def waspas(X: np.ndarray, benefit: np.ndarray, w: np.ndarray, lam: float = 0.5) -> dict:
    R = ratio_norm(X, benefit)
    q1, q2 = wsm(R, w), wpm(R, w)
    Q = lam * q1 + (1 - lam) * q2
    return dict(R=R, Q1=q1, Q2=q2, Q=Q, rank=ranks_desc(Q), rank_wsm=ranks_desc(q1), rank_wpm=ranks_desc(q2))


# ----------------------------------------------------------------- TOPSIS
def topsis(X: np.ndarray, benefit: np.ndarray, w: np.ndarray) -> dict:
    """TOPSIS (Hwang & Yoon 1981) : normalisation vectorielle, pondération, idéaux, séparations, RC*."""
    X = np.asarray(X, float)
    R = X / np.sqrt((X ** 2).sum(0))
    V = R * w
    vp = np.where(benefit, V.max(0), V.min(0))
    vm = np.where(benefit, V.min(0), V.max(0))
    Sp = np.sqrt(((V - vp) ** 2).sum(1))
    Sm = np.sqrt(((V - vm) ** 2).sum(1))
    RC = Sm / (Sp + Sm)
    return dict(R=R, V=V, v_plus=vp, v_minus=vm, S_plus=Sp, S_minus=Sm, RC=RC, rank=ranks_desc(RC))


# ----------------------------------------------------------------- VIKOR
def vikor(X: np.ndarray, benefit: np.ndarray, w: np.ndarray, v: float = 0.5) -> dict:
    """VIKOR : S (utilité de groupe), R (regret individuel), Q ; conditions C1 (avantage) et C2 (stabilité)."""
    X = np.asarray(X, float)
    m = X.shape[0]
    xs, xm = ideal_points(X, benefit)
    span = np.where(xs - xm == 0, 1, xs - xm)
    T = w * (xs - X) / span
    S, Rr = T.sum(1), T.max(1)
    Ss, Sm, Rs, Rm = S.min(), S.max(), Rr.min(), Rr.max()
    qs = (S - Ss) / (Sm - Ss) if Sm != Ss else np.zeros(m)
    qr = (Rr - Rs) / (Rm - Rs) if Rm != Rs else np.zeros(m)
    Q = v * qs + (1 - v) * qr
    order = np.argsort(Q, kind="stable")
    best, second = order[0], order[1] if m > 1 else order[0]
    DQ = 1.0 / (m - 1) if m > 1 else 1.0
    c1 = (Q[second] - Q[best]) >= DQ - 1e-12
    rank_S, rank_R, rank_Q = ranks_asc(S), ranks_asc(Rr), ranks_asc(Q)
    c2 = (rank_S[best] == 1) or (rank_R[best] == 1)
    # Ensemble de solutions de compromis
    if c1 and c2:
        comp = [best]; kind = "A' seule (C1 et C2 vérifiées)"
    elif c1 and not c2:
        comp = [best, second]; kind = "A' et A'' (C2 non vérifiée)"
    else:
        comp = [i for i in order if Q[i] - Q[best] < DQ]
        kind = f"A', A'', …, A^({len(comp)}) (C1 non vérifiée)"
    return dict(T=T, S=S, R=Rr, Q=Q, rank_S=rank_S, rank_R=rank_R, rank_Q=rank_Q, DQ=DQ,
                C1=bool(c1), C2=bool(c2), compromise=comp, compromise_kind=kind, best=int(best), v=v)
