"""Logique floue et MCDM flou : fonctions d'appartenance, nombres flous (TFN/trapézoïdaux, intervalles, IFN, PFN,
Pythagoriciens, q-ROFN, hésitants), TOPSIS flou (Chen 2000), TOPSIS flou intuitionniste."""
from __future__ import annotations
import itertools
import numpy as np

# ----------------------------------------------------------------- fonctions d'appartenance
def mf_trap(x, a, b, c, d):
    x = np.asarray(x, float)
    up = np.where(b > a, (x - a) / max(b - a, 1e-12), (x >= a).astype(float))
    dn = np.where(d > c, (d - x) / max(d - c, 1e-12), (x <= d).astype(float))
    return np.clip(np.minimum(np.minimum(up, 1.0), dn), 0, 1)


def mf_tri(x, a, b, c):
    return mf_trap(x, a, b, b, c)


def mf_gauss(x, m, sigma):
    x = np.asarray(x, float)
    return np.exp(-((x - m) ** 2) / (2 * sigma ** 2))


def age_sets(x):
    """Exemple du cours : Jeune / Âge moyen / Âgé (Âge moyen = trapèze 25-35-55-65)."""
    return {"Jeune": mf_trap(x, -1e9, -1e9 + 1, 25, 35), "Âge moyen": mf_trap(x, 25, 35, 55, 65),
            "Âgé": mf_trap(x, 55, 65, 1e9 - 1, 1e9)}


def set_properties(xs, mu, alpha: float = 0.5) -> dict:
    """Hauteur, support, noyau, α-coupe (échantillonnées sur la grille xs)."""
    xs, mu = np.asarray(xs), np.asarray(mu)
    sup, core, cut = xs[mu > 1e-9], xs[np.isclose(mu, 1.0)], xs[mu >= alpha - 1e-12]
    rng = lambda v: (float(v.min()), float(v.max())) if v.size else None
    # convexité : μ(λx1+(1−λ)x2) ≥ min(μ(x1), μ(x2)) ⇔ μ est « unimodale » sur la grille
    d = np.diff(mu); sign = np.sign(d[np.abs(d) > 1e-12])
    convex = not np.any(np.diff(sign) > 0) if sign.size else True   # pas de « remontée » après une descente
    return dict(height=float(mu.max()), normalized=bool(np.isclose(mu.max(), 1)), support=rng(sup), core=rng(core),
                alpha_cut=rng(cut), convex=bool(convex))


# ----------------------------------------------------------------- nombres flous trapézoïdaux / intervalles
def tfn_add(A, B): return tuple(a + b for a, b in zip(A, B))
def tfn_sub(A, B): return (A[0] - B[3], A[1] - B[2], A[2] - B[1], A[3] - B[0])
def tfn_mul(A, B): return tuple(a * b for a, b in zip(A, B))
def tfn_scal(l, A): return tuple(l * a for a in A)
def tfn_div(A, B): return (A[0] / B[3], A[1] / B[2], A[2] / B[1], A[3] / B[0])


def int_add(A, B): return (A[0] + B[0], A[1] + B[1])
def int_sub(A, B): return (A[0] - B[1], A[1] - B[0])
def int_mul(A, B):
    p = [A[0] * B[0], A[0] * B[1], A[1] * B[0], A[1] * B[1]]; return (min(p), max(p))
def int_div(A, B):
    if B[0] <= 0 <= B[1]: raise ZeroDivisionError("0 ∈ B : division impossible")
    return int_mul(A, (1 / B[1], 1 / B[0]))
def int_center(A): return (A[0] + A[1]) / 2
def int_width(A): return (A[1] - A[0]) / 2


# ----------------------------------------------------------------- intuitionnistes (IFN)
def ifn_check(a): mu, nu = a; return 0 <= mu <= 1 and 0 <= nu <= 1 and mu + nu <= 1 + 1e-12
def ifn_pi(a): return 1 - a[0] - a[1]
def ifn_add(a, b): return (a[0] + b[0] - a[0] * b[0], a[1] * b[1])
def ifn_mul(a, b): return (a[0] * b[0], a[1] + b[1] - a[1] * b[1])
def ifn_scal(l, a): return (1 - (1 - a[0]) ** l, a[1] ** l)
def ifn_pow(a, l): return (a[0] ** l, 1 - (1 - a[1]) ** l)
def ifn_score(a): return a[0] - a[1]
def ifn_accuracy(a): return a[0] + a[1]


def ifn_compare(a, b) -> int:
    """1 si a > b, −1 si a < b, 0 si égaux (score puis fonction de précision)."""
    sa, sb = ifn_score(a), ifn_score(b)
    if abs(sa - sb) > 1e-12: return 1 if sa > sb else -1
    ha, hb = ifn_accuracy(a), ifn_accuracy(b)
    if abs(ha - hb) > 1e-12: return 1 if ha > hb else -1
    return 0


def ifwa(alphas, sigma):
    """IFWA_σ(α¹..α^k) = (1 − Π(1−μ^j)^σj , Π ν^j^σj)."""
    sigma = np.asarray(sigma, float)
    mu = np.array([a[0] for a in alphas]); nu = np.array([a[1] for a in alphas])
    return (float(1 - np.prod((1 - mu) ** sigma)), float(np.prod(nu ** sigma)))


# ----------------------------------------------------------------- picturaux, pythagoriciens, q-ROFN
def pfn_check(a): return all(0 <= v <= 1 for v in a) and sum(a) <= 1 + 1e-12
def pfn_refusal(a): return 1 - sum(a)
def pfn_add(A, B): return (A[0] + B[0] - A[0] * B[0], A[1] * B[1], A[2] * B[2])
def pfn_scal(l, A): return (1 - (1 - A[0]) ** l, A[1] ** l, A[2] ** l)
def pfn_pow(A, l): return (A[0] ** l, 1 - (1 - A[1]) ** l, 1 - (1 - A[2]) ** l)


def qrof_check(mu, nu, q): return 0 <= mu <= 1 and 0 <= nu <= 1 and mu ** q + nu ** q <= 1 + 1e-12
def qrof_pi(mu, nu, q): return max(0.0, 1 - mu ** q - nu ** q) ** (1 / q)
def pyth_pi(mu, nu): return qrof_pi(mu, nu, 2)


# ----------------------------------------------------------------- hésitants
def hfn_score(h): return float(np.mean(h))
def hfn_dev(h): h = np.asarray(h, float); return float(np.sqrt(np.mean((h - h.mean()) ** 2)))
def hfn_pow(h, a): return sorted({round(g ** a, 10) for g in h})
def hfn_scal(a, h): return sorted({round(1 - (1 - g) ** a, 10) for g in h})
def hfn_add(h1, h2): return sorted({round(g1 + g2 - g1 * g2, 10) for g1, g2 in itertools.product(h1, h2)})
def hfn_mul(h1, h2): return sorted({round(g1 * g2, 10) for g1, g2 in itertools.product(h1, h2)})


def hfn_compare(h1, h2) -> int:
    s1, s2 = hfn_score(h1), hfn_score(h2)
    if abs(s1 - s2) > 1e-12: return 1 if s1 > s2 else -1
    d1, d2 = hfn_dev(h1), hfn_dev(h2)
    if abs(d1 - d2) > 1e-12: return 1 if d1 < d2 else -1     # écart plus petit = meilleur (cours)
    return 0


def hfn_hamming(h1, h2, extend: str = "max") -> float:
    """Distance de Hamming hésitante : les deux ensembles sont triés puis le plus court est prolongé (max/min/mean)."""
    a, b = sorted(h1), sorted(h2); l = max(len(a), len(b))
    def ext(h):
        if len(h) == l: return h
        fill = {"max": max(h), "min": min(h), "mean": float(np.mean(h))}[extend]
        return sorted(h + [fill] * (l - len(h)))
    a, b = ext(list(a)), ext(list(b))
    return float(np.mean(np.abs(np.array(a) - np.array(b))))


# ----------------------------------------------------------------- MCDM flou : une alternative
def fuzzy_single(D, w, benefit=None) -> dict:
    """Évaluation d'une alternative à TFN (a,b,c) : normalisation ÷ c_max, somme pondérée, défuzzification (a+b+c)/3.

    D : liste de n TFN ; c_max = plus grand c parmi les *alternatives* (ici la valeur de l'unique alternative,
    ou la référence fournie par l'utilisateur via D_ref).
    """
    D = np.asarray(D, float); w = np.asarray(w, float)
    cmax = D[:, 2]
    R = D / cmax[:, None]
    S = (w[:, None] * R).sum(0)
    return dict(R=R, S=S, score=float(S.mean()), width=float(S[2] - S[0]))


# ----------------------------------------------------------------- Échelles linguistiques de Chen (2000)
ALT_SCALE = {"VP": (0, 0, 1), "P": (0, 1, 3), "MP": (1, 3, 5), "F": (3, 5, 7), "MG": (5, 7, 9), "G": (7, 9, 10), "VG": (9, 10, 10)}
W_SCALE = {"VL": (0, 0, .1), "L": (0, .1, .3), "ML": (.1, .3, .5), "M": (.3, .5, .7), "MH": (.5, .7, .9), "H": (.7, .9, 1), "VH": (.9, 1, 1)}


def fuzzy_topsis(ratings: np.ndarray, weights: np.ndarray, benefit: np.ndarray) -> dict:
    """TOPSIS flou de Chen (2000), 9 étapes du cours.

    ratings : (K, m, n, 3) — évaluation des K décideurs ; weights : (K, n, 3) ; benefit : (n,) bool.
    """
    K, m, n, _ = ratings.shape
    X = ratings.mean(0)                                    # (m,n,3) agrégation collective : 1/K ⊕
    W = weights.mean(0)                                    # (n,3)
    R = np.zeros_like(X)
    for j in range(n):
        if benefit[j]:
            c_star = X[:, j, 2].max(); R[:, j, :] = X[:, j, :] / c_star
        else:
            a_min = X[:, j, 0].min(); R[:, j, :] = a_min / X[:, j, ::-1]
    V = R * W[None]                                        # v_ij = r_ij ⊗ w_j
    d_pos = np.sqrt(((V - 1) ** 2).mean(-1))               # distance vertex à (1,1,1)
    d_neg = np.sqrt((V ** 2).mean(-1))                     # distance vertex à (0,0,0)
    Dp, Dm = d_pos.sum(1), d_neg.sum(1)
    CC = Dm / (Dp + Dm)
    rank = (-CC).argsort().argsort() + 1
    return dict(X=X, W=W, R=R, V=V, d_pos=d_pos, d_neg=d_neg, D_plus=Dp, D_minus=Dm, CC=CC, rank=rank)


# ----------------------------------------------------------------- TOPSIS flou intuitionniste
def if_topsis(ratings: np.ndarray, weights: np.ndarray, sigma: np.ndarray, benefit: np.ndarray) -> dict:
    """IF-TOPSIS du cours. ratings : (K, m, n, 2) = (μ, ν) ; weights : (K, n, 2) ; sigma : poids des décideurs (Σ=1)."""
    K, m, n, _ = ratings.shape
    sigma = np.asarray(sigma, float)
    A = np.zeros((m, n, 2)); Wc = np.zeros((n, 2))
    for i in range(m):
        for j in range(n):
            A[i, j] = ifwa([tuple(ratings[k, i, j]) for k in range(K)], sigma)
    for j in range(n):
        Wc[j] = ifwa([tuple(weights[k, j]) for k in range(K)], sigma)
    Ap = np.zeros_like(A)                                  # a ⊗ w = (μa·μw , νa + νw − νa·νw)
    for i in range(m):
        for j in range(n):
            Ap[i, j] = ifn_mul(tuple(A[i, j]), tuple(Wc[j]))
    mu, nu = Ap[..., 0], Ap[..., 1]
    Ip, Im = np.zeros((n, 2)), np.zeros((n, 2))
    for j in range(n):
        if benefit[j]:
            Ip[j] = (mu[:, j].max(), nu[:, j].min()); Im[j] = (mu[:, j].min(), nu[:, j].max())
        else:
            Ip[j] = (mu[:, j].min(), nu[:, j].max()); Im[j] = (mu[:, j].max(), nu[:, j].min())
    pi = 1 - mu - nu; pip, pim = 1 - Ip[:, 0] - Ip[:, 1], 1 - Im[:, 0] - Im[:, 1]
    Sp = ((mu - Ip[:, 0]) ** 2 + (nu - Ip[:, 1]) ** 2 + (pi - pip) ** 2).sum(1) / (2 * n)
    Sm = ((mu - Im[:, 0]) ** 2 + (nu - Im[:, 1]) ** 2 + (pi - pim) ** 2).sum(1) / (2 * n)
    C = Sm / (Sm + Sp)
    rank = (-C).argsort().argsort() + 1
    return dict(A=A, W=Wc, Aw=Ap, I_plus=Ip, I_minus=Im, S_plus=Sp, S_minus=Sm, C=C, rank=rank)
