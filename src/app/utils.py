import streamlit as st
import json


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


def get_names_by_type(description_filepath: str = 'static/description.json'):
    f = json.load(open(description_filepath))
    num_features, cat_features = f["num"], f["cat"]
    return num_features, cat_features
