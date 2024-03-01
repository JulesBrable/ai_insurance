"""Menu Page of the App"""
import streamlit as st


st.set_page_config(
    page_title="AI for Actuarial Science: Final Project",
    page_icon="🧐",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Report a bug': "https://github.com/JulesBrable/ai_insurance/issues/new",
        'About': """
        If you want to read more about the project, you would be interested in going to the corresponding
        [GitHub](https://github.com/JulesBrable/ai_insurance) repository.

        Contributions:
        - [Jules Brablé](https://github.com/JulesBrable) - jules.brable@ensae.fr
        - [Eunice Koffi]() - eunice.koffi@ensae.fr
        """
    }
)

st.header("🏠 🏥 Health Insurance Cross Sell Prediction")

st.subheader("Problematic")

st.markdown("bla bla bla")

st.subheader("Descriptive Statistics")

tabs = st.tabs(["Charts", "Tables"])

with tabs[0]:
    st.write("Some Charts Here")
with tabs[1]:
    st.write("Some Tables Here")
