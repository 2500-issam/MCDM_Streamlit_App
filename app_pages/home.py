import streamlit as st
import pandas as pd
from mcdm_core.ui import header

header("Atelier de décision multicritère",
       "Chaque méthode de pondération et de classement du cours, appliquée pas à pas à des tableaux de décision que vous pouvez modifier.",
       kicker="Aide à la décision")

st.markdown("""
Le plan suit celui du cours : pondération des critères, classement des alternatives, robustesse, évaluation d'une alternative isolée,
logique floue et méthodes hybrides avec l'apprentissage automatique. Les exercices du support servent de références de calcul ;
les tableaux plus volumineux (de 12 à 16 alternatives et de 8 à 10 critères) montrent comment les méthodes se comportent
quand les données ne sont plus de taille scolaire. Toutes les valeurs sont éditables, les résultats se recalculent immédiatement.
""")

st.header("Quelle méthode choisir")
st.caption("Le cours insiste sur ce point : aucune méthode n'est meilleure dans l'absolu, le choix dépend du problème.")
c1, c2 = st.columns(2)
with c1:
    data = st.radio("Nature des données", ["Valeurs exactes", "Jugements imprécis ou linguistiques"])
    nb = st.radio("Nombre de décideurs", ["Un seul", "Plusieurs"])
with c2:
    comp = st.radio("Compensation entre critères", ["Acceptée", "Refusée (sécurité, environnement)"])
    effort = st.radio("Effort demandé à l'expert", ["Élevé acceptable", "Réduit", "Aucun"])

w_rec = {"Élevé acceptable": ("AHP", "comparaisons par paires et test de consistance"),
         "Réduit": ("BWM", "2n - 3 comparaisons seulement"),
         "Aucun": ("CRITIC ou entropie", "poids calculés sur les données, sans avis d'expert")}[effort]
if data.startswith("Jugements"):
    r_rec = ("TOPSIS flou (Chen) ou TOPSIS flou intuitionniste", "variables linguistiques converties en nombres flous")
    if nb == "Plusieurs": r_rec = (r_rec[0], r_rec[1] + ", agrégés sur l'ensemble des décideurs")
elif comp == "Acceptée":
    r_rec = ("WSM, WASPAS, TOPSIS ou VIKOR", "un bon score peut compenser un mauvais")
else:
    r_rec = ("ELECTRE III ou PROMETHEE", "seuils de préférence et veto, sans compensation totale")
st.success(f"Pondération : {w_rec[0]} ({w_rec[1]}).")
st.success(f"Classement : {r_rec[0]} ({r_rec[1]}).")
st.info("Appliquer plusieurs méthodes, comparer les classements, tester la sensibilité aux poids et justifier le choix final : c'est ce que fait la page « Dossier de décision ».")

st.header("Correspondance avec le cours")
st.dataframe(pd.DataFrame([
    ("Pondération subjective : AHP, BWM", "1", "voiture et fournisseur (cours), site industriel à 8 critères"),
    ("Pondération objective : entropie, CRITIC", "2", "machines (cours), 14 modèles x 8 métriques, 14 filières x 9 critères"),
    ("WSM, WPM, WASPAS, TOPSIS, VIKOR", "3", "exercices du cours, portefeuille de 16 actions x 10 critères"),
    ("PROMETHEE, ELECTRE III", "4", "électrification d'un village, hôpital régional"),
    ("Spearman, sensibilité, robustesse", "5", "table de rangs du cours, jeux volumineux"),
    ("MCDM à une seule alternative", "6", "scoring à 8 critères, seuils ROC et Youden"),
    ("Logique floue", "7", "exemples du cours"),
    ("TFN, TOPSIS flou, TOPSIS flou intuitionniste", "8", "recrutement (cours), 8 candidats x 7 critères x 4 décideurs, 10 traitements x 8 critères"),
    ("Comparaison globale, consensus, points de bascule", "9", "n'importe quel tableau"),
    ("MCDM et IA", "10", "300 à 500 fournisseurs x 8 critères, 10 actions x 7 critères"),
    ("Import d'un fichier", "11", "votre CSV"),
], columns=["Contenu", "Chapitre", "Données"]), width="stretch", hide_index=True)
st.caption("PROMETHEE et ELECTRE III ne sont présentées que par leur titre dans le support ; elles suivent les formulations usuelles de la littérature.")
