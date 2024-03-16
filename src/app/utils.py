import streamlit as st


def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def make_config():
    return st.set_page_config(
        page_title="AI for Actuarial Science: Final Project",
        page_icon="🧐",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Report a bug': "https://github.com/JulesBrable/ai_insurance/issues/new",
            'About': """
            If you want to read more about the project, you would be interested in going to the
            corresponding [GitHub](https://github.com/JulesBrable/ai_insurance) repository.

            Contributions:
            - [Jules Brablé](https://github.com/JulesBrable) - jules.brable@ensae.fr
            - [Eunice Koffi]() - eunice.koffi@ensae.fr
            - [Berthe Magajie Wamsa]() - berthe.magajie@ensae.fr
            - [Leela Thamaraikkannan]() - leela.thamaraikkannan@ensae.fr
            """
        }
    )


def display_classif_metrics(metrics_df, classif_report):
    cols = st.columns(2)
    with cols[0]:
        st.dataframe(metrics_df, hide_index=True)
    with cols[1]:
        st.dataframe(classif_report)
