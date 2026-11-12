"""Robustesse et comparaison des méthodes : Spearman, balayage des poids, Monte-Carlo sur les poids."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import ranking as rk
from .outranking import promethee, electre3

# valeurs critiques de Spearman (bilatéral, α = 5 %) — Zar ; au-delà de n=10 : approximation de Student
SPEARMAN_CRIT_5 = {5: 1.000, 6: 0.886, 7: 0.786, 8: 0.738, 9: 0.683, 10: 0.648}


def spearman(r1, r2) -> float:
    """ρ = 1 − 6 Σ d² / (n(n²−1)) — formule du cours (rangs sans ex æquo)."""
    r1, r2 = np.asarray(r1, float), np.asarray(r2, float); n = len(r1)
    return float(1 - 6 * np.sum((r1 - r2) ** 2) / (n * (n ** 2 - 1)))


def spearman_matrix(ranks: pd.DataFrame) -> pd.DataFrame:
    cols = list(ranks.columns)
    M = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for a in cols:
        for b in cols:
            M.loc[a, b] = spearman(ranks[a], ranks[b])
    return M


def spearman_critical(n: int) -> float:
    if n in SPEARMAN_CRIT_5: return SPEARMAN_CRIT_5[n]
    if n < 5: return 1.0
    from scipy import stats
    t = stats.t.ppf(0.975, n - 2); return float(t / np.sqrt(n - 2 + t ** 2))


# ----------------------------------------------------------------- registre de méthodes (score : grand = meilleur)
def method_registry(X: np.ndarray) -> dict:
    rng = np.ptp(X, axis=0); rng = np.where(rng == 0, 1, rng)
    def _wsm(X, b, w): return rk.wsm(rk.ratio_norm(X, b), w)
    def _wpm(X, b, w): return rk.wpm(rk.ratio_norm(X, b), w)
    def _wasp(X, b, w): return rk.waspas(X, b, w)["Q"]
    def _top(X, b, w): return rk.topsis(X, b, w)["RC"]
    def _vik(X, b, w): return -rk.vikor(X, b, w)["Q"]
    def _pro(X, b, w): return promethee(X, b, w, ["linear"] * X.shape[1], q=0.05 * rng, p=0.5 * rng)["phi"]
    def _ele(X, b, w):
        r = electre3(X, b, w, q=0.05 * rng, p=0.2 * rng, v=0.8 * rng); return -r["mean_rank"]
    return {"WSM": _wsm, "WPM": _wpm, "WASPAS": _wasp, "TOPSIS": _top, "VIKOR": _vik, "PROMETHEE II": _pro, "ELECTRE III": _ele}


def rank_of(score) -> np.ndarray:
    return pd.Series(score).rank(ascending=False, method="min").astype(int).to_numpy()


def compare_methods(X, benefit, w, names=None, registry=None) -> pd.DataFrame:
    reg = registry or method_registry(np.asarray(X, float))
    names = names or list(reg)
    return pd.DataFrame({n: rank_of(reg[n](np.asarray(X, float), benefit, w)) for n in names})


def sweep_weight(fn, X, benefit, w, j: int, grid=None) -> pd.DataFrame:
    """Fait varier w_j de 0 à 1 ; les autres poids sont rééchelonnés proportionnellement. Renvoie scores et rangs."""
    grid = np.linspace(0, 1, 41) if grid is None else grid
    w = np.asarray(w, float); others = np.delete(w, j); others = others / others.sum()
    out = []
    for g in grid:
        wn = np.empty_like(w); wn[j] = g; wn[np.arange(len(w)) != j] = (1 - g) * others
        s = fn(np.asarray(X, float), benefit, wn)
        out.append(dict(w=g, **{f"s{i}": v for i, v in enumerate(s)}, **{f"r{i}": v for i, v in enumerate(rank_of(s))}))
    return pd.DataFrame(out)


def monte_carlo(fn, X, benefit, w, n_iter: int = 500, concentration: float = 50.0, seed: int = 0) -> dict:
    """Poids tirés d'une loi de Dirichlet centrée sur w (concentration élevée = perturbation faible).

    Renvoie la matrice d'acceptabilité des rangs (m × m), la fréquence de la 1re place et le ρ de Spearman moyen
    entre le classement de base et les classements perturbés.
    """
    rng = np.random.default_rng(seed); X = np.asarray(X, float); w = np.asarray(w, float); m = X.shape[0]
    base = rank_of(fn(X, benefit, w)); acc = np.zeros((m, m)); rhos = []
    for _ in range(n_iter):
        wp = rng.dirichlet(np.maximum(w * concentration, 1e-3))
        r = rank_of(fn(X, benefit, wp))
        for i in range(m): acc[i, min(r[i], m) - 1] += 1
        rhos.append(spearman(base, r))
    return dict(base_rank=base, acceptability=acc / n_iter, top1=acc[:, 0] / n_iter, rho_mean=float(np.mean(rhos)),
                rho_min=float(np.min(rhos)))


def usable_methods(X, benefit, w) -> tuple[dict, dict]:
    """Registre filtré : écarte les méthodes dont les hypothèses ne sont pas satisfaites (ex. valeurs ≤ 0 pour WSM/WPM/WASPAS)."""
    X = np.asarray(X, float); ok, dropped = {}, {}
    for n, fn in method_registry(X).items():
        try:
            s = fn(X, benefit, w)
            if not np.isfinite(s).all(): raise ValueError("scores non finis")
            ok[n] = fn
        except (ValueError, ZeroDivisionError, FloatingPointError) as e:
            dropped[n] = str(e)
    return ok, dropped


# ----------------------------------------------------------------- Dossier de décision : consensus et points de bascule
def consensus(ranks: pd.DataFrame) -> pd.DataFrame:
    """Synthèse des rangs obtenus par plusieurs méthodes (lignes = alternatives, colonnes = méthodes)."""
    m = ranks.shape[0]
    out = pd.DataFrame({"Rang moyen": ranks.mean(axis=1), "Rang médian": ranks.median(axis=1), "Meilleur rang": ranks.min(axis=1),
                        "Pire rang": ranks.max(axis=1), "Points de Borda": (m - ranks).sum(axis=1),
                        "Premières places": (ranks == 1).sum(axis=1)})
    out["Étendue"] = out["Pire rang"] - out["Meilleur rang"]
    out["Rang de consensus"] = out["Points de Borda"].rank(ascending=False, method="min").astype(int)
    return out.sort_values(["Rang de consensus", "Rang moyen"])


def _winner(fn, X, benefit, w) -> int:
    return int(np.argmax(fn(X, benefit, w)))


def _with_weight(w, j, g):
    w = np.asarray(w, float); others = np.delete(w, j); others = others / others.sum()
    wn = np.empty_like(w); wn[j] = g; wn[np.arange(len(w)) != j] = (1 - g) * others
    return wn


def tipping_points(fn, X, benefit, w, steps: int = 200, refine: int = 18) -> pd.DataFrame:
    """Pour chaque critère j : plus petit déplacement de w_j (autres poids rééchelonnés) qui change l'alternative classée première.

    Balayage de [0, 1] puis dichotomie entre le dernier point sans changement et le premier point avec changement,
    des deux côtés du poids courant. Renvoie le seuil bas, le seuil haut et le nouveau premier de chaque côté.
    """
    X = np.asarray(X, float); w = np.asarray(w, float); n = len(w); base = _winner(fn, X, benefit, w); rows = []
    def search(j, direction):
        w0 = w[j]; grid = np.linspace(w0, 1.0 if direction > 0 else 0.0, steps + 1)[1:]
        prev = w0
        for g in grid:
            if _winner(fn, X, benefit, _with_weight(w, j, g)) != base:
                lo, hi = prev, g
                for _ in range(refine):
                    mid = (lo + hi) / 2
                    if _winner(fn, X, benefit, _with_weight(w, j, mid)) != base: hi = mid
                    else: lo = mid
                return hi, _winner(fn, X, benefit, _with_weight(w, j, hi))
            prev = g
        return None, None
    for j in range(n):
        lo_w, lo_new = search(j, -1); hi_w, hi_new = search(j, +1)
        d = [abs(w[j] - v) for v in (lo_w, hi_w) if v is not None]
        rows.append(dict(j=j, poids=w[j], seuil_bas=lo_w, nouveau_si_bas=lo_new, seuil_haut=hi_w, nouveau_si_haut=hi_new,
                         marge=min(d) if d else np.nan, marge_relative=(min(d) / w[j]) if d and w[j] > 0 else np.nan))
    return pd.DataFrame(rows)
