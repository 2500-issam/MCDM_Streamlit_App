# MCDM, laboratoire d'aide à la décision

Application **Streamlit** qui démontre pas à pas chaque méthode de **pondération** et de **décision multicritère** du cours
« Aide à la décision » (EMI Rabat — Prof. Lamrani Alaoui) : AHP, BWM, Entropie, CRITIC, WSM, WPM, WASPAS, TOPSIS, VIKOR, PROMETHEE, ELECTRE III,
Spearman/robustesse, évaluation d'une alternative isolée, logique floue, TOPSIS flou (Chen), TOPSIS flou intuitionniste, et les hybrides MCDM + IA.

Le plan détaillé est dans [`PLAN.md`](PLAN.md).

## 1. Contenu de l'archive
```
streamlit_app.py            point d'entrée
app_pages/                  accueil + 11 pages (10 chapitres du cours + le Dossier de décision)
mcdm_core/                  noyau de calcul (weights, ranking, outranking, single, fuzzy, robustness, hybrid, datasets, report, ui)
data/                       datasets au format CSV
tests/test_core.py          21 tests unitaires
.streamlit/config.toml      thème
requirements.txt            dépendances
PLAN.md                     plan de réalisation
```

## 2. Lancer en local
```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Python ≥ 3.10 recommandé (testé en 3.13). Les dépendances incluent matplotlib (dégradés des tableaux). Tests : `pip install pytest && python -m pytest tests -q`.

## 3. Datasets
| Dataset | Origine | Utilisé par |
|---|---|---|
| Achat d'une voiture (matrices AHP) | **cours** | AHP |
| Choix d'un fournisseur (BO / OW) | **cours** | BWM |
| Machines de découpe 4×4 | **cours** | CRITIC, Entropie |
| Choix de machines 4×5 + poids | **cours** | WSM/WPM/WASPAS |
| Choix d'une voiture 4×4 + poids | **cours** | TOPSIS |
| Achat d'une machine 3×2 | **cours** | VIKOR |
| Rangs A1–A6 (TOPSIS/VIKOR/PROMETHEE II) | **cours** | Spearman |
| 6 décisions passées (scores) | **cours** | seuil ROC/Youden |
| ROA/Liquidité (TFN) · Recrutement (3 DM × 5 critères) · h₁,h₂,h₃ · actions dynamiques · CV | **cours** | MCDM flou, hésitants, dynamique, LLM |
| Modèles d'apprentissage (14×8) · Électrification d'un village (14×9) · Portefeuille d'actions (16×10) · Sites d'hôpital (12×8) | créés, thèmes cités dans le cours | pondération, classement, surclassement, robustesse, Dossier |
| AHP à 8 critères et 6 sites · BWM à 8 critères · Recrutement (5 candidats, 4 décideurs, 7 critères) · Traitement médical (6 traitements, 4 décideurs, 7 critères, IFN) | créés | AHP, BWM, MCDM flou |
| 300 fournisseurs (8 critères, graine 42) · 10 actions × 7 critères sur 24 périodes · 6 CV à 7 critères · 60 décisions passées | créés (simulés) | MCDM et IA, alternative isolée |

Les datasets créés portent des valeurs **inventées** (signalées « créé » dans l'application). Régénérer les CSV : `python -m mcdm_core.datasets`.
Chaque matrice est **modifiable** dans l'interface ; la page « Vos données » accepte votre propre CSV (1re colonne = alternatives).

## 3 bis. Page « Dossier de décision »
Le même tableau est passé à toutes les méthodes applicables. La page montre :
- une frise des rangs (une ligne par alternative, une colonne par méthode) qui met en évidence les trois premières du consensus ;
- un consensus (rang moyen, médian, points de Borda, étendue) ;
- une règle de fragilité : pour chaque critère, l'intervalle de poids dans lequel la première alternative reste première (balayage puis dichotomie) ;
- des tirages aléatoires de poids (Dirichlet) ;
- un dossier HTML autonome à télécharger (matrice, poids, frise, consensus, points de bascule, conclusion), imprimable en PDF.

Sur les grandes matrices les méthodes ne s'accordent pas toujours (sur l'électrification, le groupe diesel est premier en WSM et dernier ou presque en TOPSIS) : c'est ce que la page rend visible.

## 4. Déployer sur Streamlit Community Cloud (gratuit)
1. Créez un dépôt GitHub (public ou privé) et poussez le **contenu du dossier** (le fichier `streamlit_app.py` doit être à la racine du dépôt) :
   ```bash
   git init && git add . && git commit -m "MCDM app" && git branch -M main
   git remote add origin https://github.com/<vous>/<depot>.git && git push -u origin main
   ```
2. Allez sur <https://share.streamlit.io>, connectez-vous avec GitHub → **Create app** → *Deploy a public app from GitHub*.
3. Repository = votre dépôt · Branch = `main` · **Main file path = `streamlit_app.py`** · (Advanced settings → Python 3.11 ou 3.12).
4. **Deploy**. L'installation de `requirements.txt` prend 2-3 minutes ; l'URL publique `https://<nom>.streamlit.app` est ensuite partageable.

Aucun secret, aucune base de données et aucun appel d'API externe ne sont nécessaires.

## 5. Fidélité au cours — à savoir
- **Toutes les valeurs numériques données dans le cours sont reproduites** (λmax/CR de l'AHP voiture, matrice de Spearman 0,77/0,94/0,60, tableau ROC, TFN 0,86 / largeur 0,26, extraction CV 8/7/3…) et testées.
- **BWM** : la diapositive donne w = (0,552 ; 0,207 ; 0,155 ; 0,086), mais ces poids ne satisfont pas son propre programme linéaire. L'optimum exact est (0,563 ; 0,216 ; 0,162 ; 0,060), ξ\* = 0,084. L'application affiche cet écart.
- **PROMETHEE I/II et ELECTRE III** : seulement des pages de titre dans le support → formulations standard de la littérature (distillation de Roy avec s(λ) = 0,3 − 0,15 λ).
- **Entropie** : normalisation préalable non précisée dans le cours → deux options.
- **LLM extracteur** : simulateur à règles (hors-ligne, déterministe) ; il applique la rubrique affichée, sans appeler de modèle.
- **Shapley** : valeurs exactes (espérance marginale sur un échantillon de référence) calculées sans la bibliothèque `shap`.
- Les exercices du cours n'ont pas de corrigé dans le support : les résultats affichés sont **calculés**, non copiés.

## 6. Licence & crédits
Contenu pédagogique : cours « Aide à la décision », Prof. Youssef Lamrani Alaoui, EMI Rabat. Code de l'application : libre d'usage pédagogique.
