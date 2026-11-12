"""Évaluation d'une alternative isolée (évaluation absolue) : MAUT, seuil de décision, distance à l'idéal."""
from __future__ import annotations
import numpy as np
import pandas as pd


def utilities(x, xmin, xmax, benefit) -> np.ndarray:
    """u_j(x_j) du cours : benefit (x−min)/(max−min) ; cost (x−max)/(min−max) = (max−x)/(max−min)."""
    x, xmin, xmax = (np.asarray(a, float) for a in (x, xmin, xmax))
    span = np.where(xmax - xmin == 0, 1, xmax - xmin)
    return np.where(benefit, (x - xmin) / span, (xmax - x) / span)


def maut(x, xmin, xmax, benefit, w) -> dict:
    u = utilities(x, xmin, xmax, benefit)
    return dict(u=u, contrib=np.asarray(w) * u, S=float(np.sum(np.asarray(w) * u)))


def roc_table(scores, y, thresholds) -> pd.DataFrame:
    """Pour chaque seuil τ : ŷ = 1 si S ≥ τ ; TPR, FPR, d = √(FPR² + (1−TPR)²), J = TPR − FPR (Youden)."""
    scores, y = np.asarray(scores, float), np.asarray(y, int)
    P, N = (y == 1).sum(), (y == 0).sum()
    rows = []
    for t in thresholds:
        yh = scores >= t
        tp, fp = int(((yh) & (y == 1)).sum()), int(((yh) & (y == 0)).sum())
        tpr, fpr = tp / P if P else np.nan, fp / N if N else np.nan
        rows.append(dict(seuil=float(t), TP=tp, FP=fp, FN=int(P - tp), TN=int(N - fp), TPR=tpr, FPR=fpr,
                         d=float(np.hypot(fpr, 1 - tpr)), J=float(tpr - fpr),
                         exactitude=float((tp + (N - fp)) / len(y))))
    return pd.DataFrame(rows)


def candidate_thresholds(scores) -> np.ndarray:
    """Seuils candidats = scores observés (chaque score est un point de rupture de la courbe ROC)."""
    return np.unique(np.round(np.asarray(scores, float), 10))[::-1]


def topsis_single(x, x_plus, x_minus, w=None, normalise: bool = True) -> dict:
    """TOPSIS pour une alternative : D+ et D− à l'idéal / l'anti-idéal fournis ; C = D−/(D+ + D−).

    normalise=True : chaque écart est divisé par |x+ − x−| (sinon les unités différentes faussent la distance).
    """
    x, xp, xm = (np.asarray(a, float) for a in (x, x_plus, x_minus))
    span = np.where(np.abs(xp - xm) == 0, 1, np.abs(xp - xm)) if normalise else np.ones_like(x)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    Dp = float(np.sqrt(np.sum(w * ((x - xp) / span) ** 2)))
    Dm = float(np.sqrt(np.sum(w * ((x - xm) / span) ** 2)))
    return dict(D_plus=Dp, D_minus=Dm, C=Dm / (Dp + Dm) if (Dp + Dm) else np.nan)
