import streamlit as st
import pandas as pd

@st.cache_data
def load_data(url):
    return pd.read_csv(url)

def get_features_by_type(df, selected_features):
    num_features = df[selected_features].select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = df[selected_features].select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    return num_features, cat_features

