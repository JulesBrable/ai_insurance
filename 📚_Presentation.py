"""Menu Page of the App"""
import streamlit as st
import pandas as pd
import json

from src.app.utils import load_css, make_config
from src.etl import load_data
import src.app.page0 as p0

make_config()
load_css('static/styles.css')

st.markdown("<h1 class='h1-title'> 🏠 Health Insurance Cross Sell Prediction 🏥 </h1>",
            unsafe_allow_html=True)
st.subheader(" ")
st.markdown("<h1 class='h1-problematic'> Problematic </h1>", unsafe_allow_html=True)

st.markdown("""<h1 class='h1-insurance-goal'>An Insurance company that has provided Health Insurance
                to its customers now need to build a model to predict whether the policyholders
                (customers) from past year will also be interested in Vehicle Insurance provided by
                the company. Building a model to predict whether a customer would be interested in
                Vehicle Insurance is extremely helpful for the company because it can then
                accordingly plan its communication strategy to reach out to those customers and
                optimise its business model and revenue. So, the aim of this project is to help the
                company reach its goal.</h1>""", unsafe_allow_html=True)
st.subheader(" ")

st.markdown("<h1 class='h1-dataset-desc'> Dataset column descriptions </h1>",
            unsafe_allow_html=True)
st.subheader(' ')

column_data = json.load(open('static/description.json'))

data_column_description = pd.DataFrame(column_data)
st.dataframe(data_column_description, hide_index=True)

df = load_data()

if st.checkbox('Show Dataset'):
    st.subheader('Dataset first five rows')
    st.dataframe(df.head(), hide_index=True)

st.markdown("<h1 class='h1-descriptive-stats'> Descriptive Statistics </h1>",
            unsafe_allow_html=True)
st.subheader(" ")

num_features = ['Age', 'Annual_Premium', 'Vintage']  
cat_features = [
    'Gender', 'Driving_License', 'Region_Code', 'Previously_Insured',
    'Vehicle_Damage', 'Policy_Sales_Channel', 'Vehicle_Age'
    ]

tabs = st.tabs(["Charts", "Tables"])

with tabs[0]:
    st.markdown("<h1 class='h1-univariate-analysis'> Univariate Analysis </h1>",
                unsafe_allow_html=True)
    st.subheader(" ")

    p0.plot_continuous(df, num_features)

    cols = st.columns([2/3, 1/3], gap="medium")
    with cols[0]:
        # Modify Categorical variables list
        cat_features_modified = cat_features.copy()
        cat_features_modified.remove("Region_Code")
        cat_features_modified.remove("Policy_Sales_Channel")

        p0.plot_categorical_distributions(df, cat_features_modified)
    with cols[1]:
        p0.plot_pie(df)

    st.markdown("<h1 class='h1-multivariate-analysis'> Multivariate Analysis </h1>",
                unsafe_allow_html=True)
    st.subheader(" ")

    p0.plot_kde_by_response(df)

    cols = st.columns(2)
    with cols[0]:
        p0.plot_heatmap(df, num_features)
    with cols[1]:
        p0.plot_cramers_v_heatmap(df, cat_features)

with tabs[1]:
    st.markdown("<h1 class='h1-stats-numeric'> Statistics for numeric variables </h1>",
                unsafe_allow_html=True)
    st.table(df[num_features].describe())

    st.markdown("<h1 class='h1-stats-categorical'> Statistics for categorical variables </h1>",
                unsafe_allow_html=True)
    st.table(df[cat_features].astype(str).describe())
