"""Pondération des critères : AHP, BWM (subjectives) ; Entropie, CRITIC (objectives)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import linprog

# Table RI du cours (n = 3..10). Pour n = 1, 2 la matrice est toujours cohérente (RI = 0).
RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.56}


# ----------------------------------------------------------------- AHP
def ahp_reciprocal(upper: np.ndarray) -> np.ndarray:
    """Complète une matrice par réciprocité : a_ji = 1 / a_ij (on lit la partie triangulaire supérieure)."""
    n = upper.shape[0]
    A = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = upper[i, j]
            A[j, i] = 1.0 / upper[i, j]
    return A


def ahp(A: np.ndarray, method: str = "approx") -> dict:
    """AHP : poids + test de consistance.

    method='approx' : méthode du cours (normalisation par colonne puis moyenne des lignes).
    method='eigen'  : vecteur propre principal exact.
    λmax = Σ s_i w_i ; CI = (λmax − n)/(n − 1) ; CR = CI / RI.
    """
    A = np.asarray(A, float)
    n = A.shape[0]
    s = A.sum(axis=0)
    norm = A / s
    if method == "eigen":
        vals, vecs = np.linalg.eig(A)
        k = np.argmax(vals.real)
        w = np.abs(vecs[:, k].real)
        w = w / w.sum()
    else:
        w = norm.mean(axis=1)
    lam = float(np.sum(s * w))
    CI = (lam - n) / (n - 1) if n > 1 else 0.0
    RI = RI_TABLE.get(n, 1.56)
    CR = CI / RI if RI > 0 else 0.0
    return dict(w=w, col_sums=s, normalized=norm, lambda_max=lam, CI=CI, RI=RI, CR=CR, consistent=CR < 0.1)


def ahp_rank(crit_A: np.ndarray, alt_As: list[np.ndarray], method: str = "approx") -> dict:
    """AHP complet : poids des critères, priorités locales (1 matrice par critère), priorités globales."""
    top = ahp(crit_A, method)
    locs = [ahp(M, method) for M in alt_As]
    L = np.column_stack([r["w"] for r in locs])  # m x n
    G = L @ top["w"]
    return dict(criteria=top, local=locs, local_matrix=L, global_priorities=G)


# ----------------------------------------------------------------- BWM
def bwm(best: int, worst: int, bo: np.ndarray, ow: np.ndarray) -> dict:
    """Best-Worst Method (Rezaei 2015) — programme linéaire du cours.

    Variables x = (w_1..w_n, ξ). min ξ  s.c.  |w_B − a_Bj w_j| ≤ ξ ; |w_j − a_jW w_W| ≤ ξ ; Σw = 1 ; w ≥ 0.
    """
    bo = np.asarray(bo, float)
    ow = np.asarray(ow, float)
    n = len(bo)
    rows, rhs, labels = [], [], []
    for j in range(n):
        # w_B − a_Bj w_j − ξ ≤ 0   et   −w_B + a_Bj w_j − ξ ≤ 0
        r = np.zeros(n + 1); r[best] += 1; r[j] -= bo[j]; r[-1] = -1
        rows.append(r); rhs.append(0); labels.append(f"w{best+1} − {bo[j]:g}·w{j+1} − ξ ≤ 0")
        r = np.zeros(n + 1); r[best] -= 1; r[j] += bo[j]; r[-1] = -1
        rows.append(r); rhs.append(0); labels.append(f"−w{best+1} + {bo[j]:g}·w{j+1} − ξ ≤ 0")
        # w_j − a_jW w_W − ξ ≤ 0   et   −w_j + a_jW w_W − ξ ≤ 0
        r = np.zeros(n + 1); r[j] += 1; r[worst] -= ow[j]; r[-1] = -1
        rows.append(r); rhs.append(0); labels.append(f"w{j+1} − {ow[j]:g}·w{worst+1} − ξ ≤ 0")
        r = np.zeros(n + 1); r[j] -= 1; r[worst] += ow[j]; r[-1] = -1
        rows.append(r); rhs.append(0); labels.append(f"−w{j+1} + {ow[j]:g}·w{worst+1} − ξ ≤ 0")
    A_ub = np.array(rows); b_ub = np.array(rhs, float)
    A_eq = np.zeros((1, n + 1)); A_eq[0, :n] = 1
    c = np.zeros(n + 1); c[-1] = 1
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1], bounds=[(0, None)] * (n + 1), method="highs")
    if not res.success:
        raise ValueError("BWM : le programme linéaire n'a pas de solution — vérifier les vecteurs BO/OW.")
    w = res.x[:n]
    return dict(w=w, xi=float(res.x[-1]), A_ub=A_ub, b_ub=b_ub, labels=labels)


def bwm_checks(best: int, worst: int, bo, ow) -> list[str]:
    """Contrôles de bon sens du cours : a_BB = a_WW = 1 ; la note max de BO porte sur W."""
    msgs = []
    if bo[best] != 1: msgs.append("a_BB doit valoir 1.")
    if ow[worst] != 1: msgs.append("a_WW doit valoir 1.")
    if int(np.argmax(bo)) != worst and bo[worst] < max(bo): msgs.append("La note la plus élevée de BO devrait porter sur le critère W.")
    if int(np.argmax(ow)) != best and ow[best] < max(ow): msgs.append("La note la plus élevée de OW devrait porter sur le critère B.")
    return msgs


# ----------------------------------------------------------------- Normalisations
def minmax_norm(X: np.ndarray, benefit: np.ndarray) -> np.ndarray:
    """Min-max du cours (CRITIC) : positif (x−min)/(max−min) ; négatif (max−x)/(max−min)."""
    X = np.asarray(X, float)
    mn, mx = X.min(0), X.max(0)
    rng = np.where(mx - mn == 0, 1, mx - mn)
    return np.where(benefit, (X - mn) / rng, (mx - X) / rng)


# ----------------------------------------------------------------- Entropie
def entropy_weights(X: np.ndarray, benefit: np.ndarray, norm: str = "sum") -> dict:
    """Entropie de Shannon. p_ij = d_ij / Σ_i d_ij ; E_j = −k Σ p ln p, k = 1/ln m ; w_j = (1−E_j)/(n−ΣE).

    norm='sum'    : d = x (bénéfice) ou 1/x (coût)   — requiert x > 0
    norm='minmax' : d = min-max du cours (les 0 sont gérés par la convention 0·ln0 = 0)
    """
    X = np.asarray(X, float)
    m, n = X.shape
    if norm == "minmax":
        D = minmax_norm(X, benefit)
    else:
        if (X <= 0).any():
            raise ValueError("Normalisation 'somme' : toutes les valeurs doivent être > 0 (sinon choisir min-max).")
        D = np.where(benefit, X, 1.0 / X)
    colsum = D.sum(0)
    P = np.divide(D, colsum, out=np.zeros_like(D), where=colsum != 0)
    k = 1.0 / np.log(m)
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(P > 0, P * np.log(P), 0.0)
    E = -k * plogp.sum(0)
    div = 1 - E
    w = div / div.sum() if div.sum() > 0 else np.full(n, 1 / n)   # = (1−E)/(n−ΣE)
    return dict(w=w, D=D, P=P, E=E, diversification=div)


# ----------------------------------------------------------------- CRITIC
def critic_weights(X: np.ndarray, benefit: np.ndarray) -> dict:
    """CRITIC (Diakoulaki 1995) : min-max, corrélation de Pearson, σ (m−1), C_j = σ_j Σ(1−|ρ_jk|), w = C/ΣC."""
    X = np.asarray(X, float)
    R = minmax_norm(X, benefit)
    rho = np.corrcoef(R, rowvar=False)
    rho = np.nan_to_num(rho, nan=0.0)          # colonne constante → corrélation indéfinie → 0
    np.fill_diagonal(rho, 1.0)
    sigma = R.std(axis=0, ddof=1)
    indep = (1 - np.abs(rho)).sum(axis=0)
    C = sigma * indep
    w = C / C.sum()
    return dict(w=w, R=R, rho=rho, sigma=sigma, independence=indep, C=C)


def combine_weights(ws: list[np.ndarray], mode: str = "mean", alpha: float | None = None) -> np.ndarray:
    """Poids hybrides (N.B. du cours : on combine souvent subjectif + objectif). mean | product (multiplicatif)."""
    W = np.vstack(ws)
    if mode == "product":
        p = np.prod(W, axis=0); return p / p.sum()
    if alpha is not None and len(ws) == 2:
        w = alpha * ws[0] + (1 - alpha) * ws[1]; return w / w.sum()
    w = W.mean(axis=0); return w / w.sum()
