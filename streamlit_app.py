"""Point d'entrée Streamlit : `streamlit run streamlit_app.py`."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import streamlit as st
from mcdm_core.ui import inject_css

st.set_page_config(page_title="Atelier MCDM", page_icon=None, layout="wide", initial_sidebar_state="expanded")
inject_css()

P = lambda f, t, **k: st.Page(f"app_pages/{f}.py", title=t, **k)
pages = {
    "Introduction": [P("home", "Accueil et guide de choix", default=True)],
    "Pondération": [P("p1_weights_subjective", "1. Subjective : AHP, BWM"), P("p2_weights_objective", "2. Objective : Entropie, CRITIC")],
    "Classement": [P("p3_ranking_additive", "3. Additif : WSM, WASPAS, TOPSIS, VIKOR"), P("p4_outranking", "4. Surclassement : PROMETHEE, ELECTRE III")],
    "Analyse": [P("p5_robustness", "5. Robustesse et comparaison"), P("p6_single_alternative", "6. Alternative isolée")],
    "Logique floue": [P("p7_fuzzy_logic", "7. Logique floue"), P("p8_fuzzy_mcdm", "8. MCDM flou")],
    "Synthèse": [P("p11_dossier", "9. Dossier de décision"), P("p9_hybrid_ai", "10. MCDM et IA"), P("p10_sandbox", "11. Vos données")],
}
st.navigation(pages).run()
with st.sidebar:
    st.caption("d'aide à la décision")
