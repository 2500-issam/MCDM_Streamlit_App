"""Génération du dossier de décision au format HTML autonome (aucune dépendance externe)."""
from __future__ import annotations
import html
import numpy as np
import pandas as pd

CSS = """
body{font-family:Georgia,'Times New Roman',serif;color:#1b1f2a;background:#f6f3ec;max-width:980px;margin:32px auto;padding:0 24px;line-height:1.5}
h1{font-size:28px;margin:0 0 4px;border-bottom:2px solid #1b1f2a;padding-bottom:8px}h2{font-size:15px;letter-spacing:.12em;text-transform:uppercase;margin:34px 0 8px;border-bottom:1px solid #b9b2a0;padding-bottom:4px}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums;margin:8px 0}th,td{border-bottom:1px solid #d6d0c0;padding:4px 8px;text-align:right}
th:first-child,td:first-child{text-align:left}th{font-weight:600;border-bottom:1.5px solid #1b1f2a}.meta{color:#5b5f69;font-size:13px}.verdict{border-left:4px solid #9a3412;background:#efe9da;padding:10px 14px;margin:14px 0}
"""
PAL = ["#1f4e5f", "#9a3412", "#6b7f3a", "#7a4e7e", "#b08900", "#3b6ea5", "#8a5a44", "#5c5c5c"]


def _tbl(df: pd.DataFrame, fmt="{:.3f}") -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in [df.index.name or ""] + list(df.columns))
    body = ""
    for idx, row in df.iterrows():
        cells = "".join(f"<td>{fmt.format(v) if isinstance(v, (float, np.floating)) else html.escape(str(v))}</td>" for v in row)
        body += f"<tr><td>{html.escape(str(idx))}</td>{cells}</tr>"
    return f"<table><tr>{head}</tr>{body}</table>"


def bump_svg(ranks: pd.DataFrame, highlight: list[str], w: int = 900, h: int = 360) -> str:
    """Frise des rangs : une ligne par alternative, une colonne par méthode (SVG statique)."""
    m, k = ranks.shape; L, R, T, B = 150, 120, 28, 24
    x = lambda j: L + (w - L - R) * j / max(k - 1, 1); y = lambda r: T + (h - T - B) * (r - 1) / max(m - 1, 1)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" font-family="Georgia,serif" font-size="11">']
    for j, c in enumerate(ranks.columns):
        out.append(f'<text x="{x(j):.1f}" y="14" text-anchor="middle" fill="#5b5f69">{html.escape(str(c))}</text>')
        out.append(f'<line x1="{x(j):.1f}" y1="{T-6}" x2="{x(j):.1f}" y2="{h-B+6}" stroke="#d6d0c0"/>')
    for i, a in enumerate(ranks.index):
        hot = a in highlight; col = PAL[highlight.index(a) % len(PAL)] if hot else "#b9b2a0"
        pts = " ".join(f"{x(j):.1f},{y(ranks.iloc[i, j]):.1f}" for j in range(k))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{2.6 if hot else 1.1}"/>')
        out.append(f'<text x="{L-8}" y="{y(ranks.iloc[i, 0])+4:.1f}" text-anchor="end" fill="{col if hot else "#8a8577"}">{html.escape(str(a))}</text>')
        out.append(f'<text x="{w-R+8}" y="{y(ranks.iloc[i, -1])+4:.1f}" fill="{col if hot else "#8a8577"}">{html.escape(str(a))}</text>')
    out.append("</svg>"); return "".join(out)


def build_report(title: str, dataset: str, X: pd.DataFrame, benefit, w, w_origin: str, ranks: pd.DataFrame, cons: pd.DataFrame,
                 tips: pd.DataFrame, crit: list[str], alts: list[str], verdict: str, mc: dict | None = None) -> str:
    mat = X.copy(); mat.index.name = "Alternative"
    wt = pd.DataFrame({"Type": ["bénéfice" if b else "coût" for b in benefit], "Poids": np.asarray(w, float)}, index=crit); wt.index.name = "Critère"
    tp = tips.copy(); tp.insert(0, "Critère", [crit[j] for j in tp["j"]])
    tp = tp.drop(columns="j"); tp["nouveau_si_bas"] = [alts[int(v)] if pd.notna(v) else "-" for v in tp["nouveau_si_bas"]]
    tp["nouveau_si_haut"] = [alts[int(v)] if pd.notna(v) else "-" for v in tp["nouveau_si_haut"]]
    tp = tp.rename(columns={"poids": "Poids", "seuil_bas": "Seuil bas", "nouveau_si_bas": "Premier si bas", "seuil_haut": "Seuil haut",
                            "nouveau_si_haut": "Premier si haut", "marge": "Marge", "marge_relative": "Marge relative"}).set_index("Critère")
    tp = tp.fillna("-")
    hl = list(cons.index[:3])
    parts = [f"<!doctype html><html lang='fr'><meta charset='utf-8'><title>{html.escape(title)}</title><style>{CSS}</style><body>",
             f"<h1>{html.escape(title)}</h1><div class='meta'>Jeu de données : {html.escape(dataset)} ; {len(alts)} alternatives, {len(crit)} critères ; origine des poids : {html.escape(w_origin)}.</div>",
             f"<div class='verdict'>{html.escape(verdict)}</div>",
             "<h2>1. Matrice de décision</h2>", _tbl(mat, "{:.4g}"), "<h2>2. Critères et poids</h2>", _tbl(wt),
             "<h2>3. Frise des rangs par méthode</h2>", bump_svg(ranks, hl), "<h2>4. Consensus</h2>", _tbl(cons, "{:.2f}"),
             "<h2>5. Points de bascule des poids</h2>", "<p class='meta'>Poids du critère à partir duquel un autre premier apparaît (les autres poids restent proportionnels).</p>", _tbl(tp, "{:.3f}")]
    if mc is not None:
        parts += ["<h2>6. Simulation de Monte-Carlo</h2>", f"<p>{html.escape(mc['text'])}</p>"]
    parts.append("</body></html>")
    return "".join(parts)
