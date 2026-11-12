# Plan de réalisation — Laboratoire MCDM (Streamlit)

## 1. Objectif
Application web qui **démontre, méthode par méthode, le fonctionnement de chaque technique de pondération et de décision** du cours
« Aide à la décision » (EMI Rabat, Prof. Lamrani Alaoui, 154 diapositives), déployable sur **Streamlit Community Cloud**.

## 2. Principes directeurs
1. **Fidélité au cours** : formules, notations et exercices des diapositives ; chaque valeur donnée dans le cours est vérifiée par un test automatique.
2. **Transparence** : chaque page = rappel du cours → dataset modifiable → *calcul pas à pas* → graphiques → lecture du résultat.
3. **Datasets** : on utilise les données du cours quand elles existent (exercices) ; sinon on **crée** un jeu pédagogique relié à un exemple cité dans le cours, signalé « créé » dans l'interface.
4. **Séparation calcul / interface** : `mcdm_core/` (pur NumPy/SciPy, testable) ≠ `app_pages/` (Streamlit).
5. **Honnêteté** : les écarts entre le cours et la littérature, ou les parties absentes du support, sont signalés dans l'application.

## 3. Architecture
```
streamlit_app.py            point d'entrée (navigation par sections)
app_pages/                  une page par chapitre du cours (home, p1 … p10) + p11_dossier
mcdm_core/                  weights · ranking · outranking · single · fuzzy · robustness · hybrid · datasets · report · ui
data/                       CSV exportés des datasets (python -m mcdm_core.datasets)
tests/test_core.py          21 tests (valeurs du cours + propriétés mathématiques)
```

## 4. Couverture du cours → pages → datasets

| # | Chapitre du cours | Méthodes démontrées | Dataset | Origine |
|---|---|---|---|---|
| 0 | Introduction, taxonomie | Guide interactif « quelle méthode choisir ? » (données, décideurs, compensation, effort) | — | cours |
| 1 | Pondération subjective | **AHP** (comparaisons, λmax, CI, CR, priorités locales/globales, sensibilité) · **BWM** (programme linéaire, ξ\*) | Achat d'une voiture · Choix d'un fournisseur | cours |
| 2 | Pondération objective | **Entropie** · **CRITIC** · comparaison · poids hybrides | Machines de découpe (cours) · Modèles de ML · Énergie · Portefeuille | cours + créés |
| 3 | Classement additif | **WSM · WPM · WASPAS** · **TOPSIS** · **VIKOR** (C1/C2, ensemble de compromis) | Machines (WASPAS) · Voitures (TOPSIS) · Machines (VIKOR) | cours |
| 4 | Surclassement | **PROMETHEE I/II** (6 fonctions de préférence) · **ELECTRE III** (q, p, v, crédibilité, distillations, effet du veto) | Électrification d'un village | créé |
| 5 | Robustesse | **Spearman** (+ valeurs critiques) · comparaison de 7 méthodes · balayage d'un poids · **Monte-Carlo** (acceptabilité des rangs) | Table de rangs A1–A6 du cours · 5 datasets | cours + créés |
| 6 | Alternative isolée | **MAUT** + seuil · seuil par **ROC / Youden / distance à l'idéal** · TOPSIS d'une alternative | Scoring de crédit (créé) · 6 décisions du cours | cours + créé |
| 7 | Logique floue | Ensembles flous (trapèze, triangle, gauss, α-coupe) · TFN & intervalles · **IFN, PFN, Pythagoricien, q-ROFN** · **hésitants** | Exemples du cours (âge, h₁/h₂/h₃, vote) | cours |
| 8 | MCDM flou | Une alternative (TFN) · **TOPSIS flou de Chen** · **TOPSIS flou intuitionniste** | Recrutement (cours) · Traitement médical (créé) | cours + créé |
| 9 | MCDM & IA | Scénario 1 (MCDM → classifieur) · Scénario 2 (**Shapley** → mise à jour des poids) · **MCDM dynamique** · **LLM extracteur de scores** | 300 fournisseurs synthétiques · Actions (cours) · CV (cours) | cours + créés |
| 10 | Vos données | Pipeline complet sur un CSV importé | CSV utilisateur | — |
| 11 | Dossier de décision | Frise des rangs, consensus (Borda), règle de fragilité des poids, tirages de poids, export HTML | Au choix parmi 6 jeux | cours + créés |

## 5. Phases
| Phase | Contenu | Statut |
|---|---|---|
| A | Lecture rigoureuse du cours (texte + diapositives-images : matrices des exercices, échelles linguistiques) | ✅ |
| B | Noyau de calcul + **vérification contre les valeurs du cours** | ✅ 21 tests |
| C | Datasets (cours + créés), export CSV | ✅ |
| D | Interface : 12 pages, matrices modifiables, pas-à-pas, graphiques | ✅ testée automatiquement (AppTest) |
| E | Documentation (README, plan) et archive ZIP | ✅ |
| F | **Déploiement sur Streamlit Community Cloud** | ⏳ à faire par vous (voir README §4) |
| G | Revue visuelle par l'enseignant / relecture des datasets créés | ⏳ |

## 6. Points d'attention relevés pendant la lecture du cours
- **BWM** : la diapositive annonce w = (0,552 ; 0,207 ; 0,155 ; 0,086) mais ces poids violent le programme linéaire de la diapositive précédente (écart max ≈ 0,22). L'optimum exact est unique : (0,563 ; 0,216 ; 0,162 ; 0,060), ξ\* = 0,084 (≈ 0,08 annoncé). L'application affiche les deux.
- **PROMETHEE, ELECTRE III** : une page de titre seulement → formulations standard de la littérature, signalées.
- **Entropie** : la normalisation préalable n'est pas précisée → deux options proposées.
- **WSM/WPM/WASPAS** : la normalisation x⁺/x (coût) est indéfinie pour une valeur nulle → l'application refuse proprement ces données (au lieu de produire des NaN).
- **TOPSIS d'une alternative** : les écarts sont normalisés par |x⁺ − x⁻| pour neutraliser les unités (ajout documenté).
- **IF-TOPSIS** : le cours écrit « −1 ≤ Cᵢ ≤ 1 » ; avec la formule S⁻/(S⁻+S⁺), Cᵢ ∈ [0, 1].
- **LLM** : la partie « MCDM & IA générative » est simulée par des règles déterministes (aucun appel d'API), pour garder une démo reproductible et gratuite.

## 7. Évolutions possibles
Méthodes absentes du cours mais citées (SWARA, ANP, QUALIFLEX/TODIM) · export PDF des résultats · version anglaise · vrai appel LLM via `st.secrets`.

## 8. Révision 2
- Aucun pictogramme ni emoji ; typographie Newsreader / IBM Plex, palette papier, encre, rouille et bleu-vert, angles droits ; textes rédigés de façon sobre.
- Jeux de test agrandis : 7 à 10 critères et 12 à 16 alternatives pour tous les exemples créés (les exemples du cours restent inchangés).
- Nouvelle page « Dossier de décision » (voir README, section 3 bis).
