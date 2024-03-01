import pandas as pd
from io import BytesIO
import streamlit as st

from utils.models import get_model_configs

def manage_sidebar(df: pd.DataFrame, SEED):
    all_features = df.drop(columns=["Response"]).columns.tolist()

    selected_features = st.sidebar.multiselect("Select features to include", all_features, default=all_features)

    models_params = get_model_configs()
    model_type = st.sidebar.selectbox("Select Model Type", list(models_params.keys()))
    selected_model_info = models_params[model_type]

    model_kwargs = {}
    for param in selected_model_info["params"]:
        if param["type"] == "slider":
            value = st.sidebar.slider(param["name"], param["min"], param["max"], param["default"])
            model_kwargs[param["name"]] = value

    model = selected_model_info["model"](**model_kwargs, random_state=SEED)
    test_size = st.sidebar.slider("Test Size (Proportion)", 0.01, 0.99, 0.20)
    return selected_features, model_type, model, model_kwargs, test_size


@st.cache_data
def to_excel(classif, model_info):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        classif.to_excel(writer, sheet_name='Results', index=True)
        model_info.to_excel(writer, sheet_name='Params', index=True)
    return buffer


def download_results(results):
    return st.download_button(
        label="📥 Download classification report as XLSX",
        data=results,
        file_name='results.csv'
    )


def get_model_info(model_type, model_kwargs, SEED):
    return pd.DataFrame({
        "Model": [model_type],
        **{k: [v] for k, v in model_kwargs.items()},
        "Seed": [SEED]
    }, index=[0])
