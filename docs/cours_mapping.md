# Correspondance diapositives → code

| Diapositives (≈) | Contenu | Fichier / fonction |
|---|---|---|
| 5–8 | Taxonomie MADM, choix d'une méthode | `app_pages/home.py` (guide interactif) |
| 22–30 | AHP | `mcdm_core/weights.py` : `ahp`, `ahp_rank`, `RI_TABLE` |
| 31–40 | BWM (LP) | `weights.bwm`, `bwm_checks` |
| 41–42 | Entropie | `weights.entropy_weights` |
| 43–51 | CRITIC | `weights.critic_weights` |
| 52–59 | WSM, WPM, WASPAS | `ranking.ratio_norm`, `wsm`, `wpm`, `waspas` |
| 60–65 | TOPSIS | `ranking.topsis` |
| 66–72 | VIKOR | `ranking.vikor` |
| 73–76 | PROMETHEE, ELECTRE III (titres) | `outranking.py` (formulations standard) |
| 77–79 | Spearman | `robustness.spearman`, `spearman_matrix` |
| 80–94 | Alternative isolée, MAUT, seuil ROC/Youden, distance à l'idéal | `single.py` |
| 95–103 | Hybrides ML, XAI/SHAP, dynamique | `hybrid.py` |
| 96–121 | Logique floue (ensembles, TFN, intervalles, IFN, PFN, PFS, q-ROFN, hésitants) | `fuzzy.py` |
| 122–141 | MCDM flou, TOPSIS flou (Chen), exercice recrutement | `fuzzy.fuzzy_single`, `fuzzy_topsis` |
| 142–151 | TOPSIS flou intuitionniste | `fuzzy.if_topsis` |
| 152–154 | LLM extracteur de scores | `hybrid.extract_scores` |
| (ajout) | Consensus, points de bascule, rapport HTML | `robustness.consensus`, `tipping_points`, `report.build_report`, `app_pages/p11_dossier.py` |

Numéros de diapositives indicatifs (pagination du PDF).
