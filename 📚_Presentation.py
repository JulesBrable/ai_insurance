"""Menu Page of the App"""
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from src.etl import load_data
from src.app.page0 import plot_continuous, plot_heatmap, plot_cramers_v_heatmap, plot_kde_by_response, plot_categorical_distributions, plot_pie


st.set_page_config(
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

st.markdown("<h1 style='font-family:Lucida Caligraphy;font-size:40px;color:DarkSlateBlue;text-align: center;'> 🏠 Health Insurance Cross Sell Prediction 🏥 </h1>", unsafe_allow_html=True)
st.subheader(" ")
st.markdown("<h1 style='font-family:Lucida Caligraphy;font-size:35px ;text-align: left;'> Problematic </h1>", unsafe_allow_html=True)

st.markdown("<h1 style='font-family:Lucida Caligraphy;font-size:20px ;text-align: justify;'>An Insurance company that has provided Health Insurance to its customers now need to build a model to predict whether the policyholders (customers) from past year will also be interested in Vehicle Insurance provided by the company. Building a model to predict whether a customer would be interested in Vehicle Insurance is extremely helpful for the company because it can then accordingly plan its communication strategy to reach out to those customers and optimise its business model and revenue. So, the aim of this project is to help the company reach its goal.</h1>", unsafe_allow_html=True)
st.subheader(" ")

st.markdown("<h1 style='font-family:Lucida Caligraphy;font-size:35px ;text-align: left;'> Dataset column descriptions </h1>", unsafe_allow_html=True)
st.subheader(' ')


column_data = {
    "Column Name": [
        "id", "Gender", "Age", "Driving_License", "Region_Code", 
        "Previously_Insured", "Vehicle_Age", "Vehicle_Damage", 
        "Annual_Premium", "Policy_Sales_Channel", "Vintage", "Response"
    ],
    "Column Description": [
        "Unique ID for the customer", 
        "Gender of the customer", 
        "Age of the customer", 
        "0 - Customer does not have DL, 1 - Customer already has DL", 
        "Unique code for the region of the customer", 
        "1 - Customer already has Vehicle Insurance, 0 - Customer doesn't have Vehicle Insurance", 
        "Age of the Vehicle", 
        "1 - Customer got his/her vehicle damaged in the past. 0 - Customer didn't get his/her vehicle damaged in the past.", 
        "The amount customer needs to pay as premium in the year", 
        "Anonymized Code for the channel of outreaching to the customer ie. Different Agents, Over Mail, Over Phone, In Person, etc.", 
        "Number of Days, Customer has been associated with the company", 
        "1 - Customer is interested, 0 - Customer is not interested"
    ]
}

data_column_description= pd.DataFrame(column_data)
st.dataframe(data_column_description, hide_index=True)

df = load_data()

if st.checkbox('Show Dataset'):
    st.subheader('Dataset first five rows')
    st.dataframe(df.head(), hide_index=True)

st.markdown("<h1 style='font-family:Lucida Caligraphy;font-size:35px ;text-align: left;'> Descriptive Statistics </h1>", unsafe_allow_html=True)
st.subheader(" ")



num_features = ['Age', 'Annual_Premium', 'Vintage']  
cat_features = ['Gender', 'Driving_License', 'Region_Code', 'Previously_Insured', 
            'Vehicle_Damage', 'Policy_Sales_Channel', 'Vehicle_Age']

tabs = st.tabs(["Charts", "Tables"])

with tabs[0]:
    st.markdown("<h1 style='font-family:Lucida Caligraphy;font-size:35px ;text-align: left;'> Univariate Analysis </h1>", unsafe_allow_html=True)
    st.subheader(" ")
    
    plot_continuous(df, num_features)
    
    cols=st.columns([2/3, 1/3], gap="medium")
    with cols[0]:
        # Modify Categorical variables list
        cat_features_modified = cat_features.copy()
        cat_features_modified.remove("Region_Code")
        cat_features_modified.remove("Policy_Sales_Channel")

        plot_categorical_distributions(df, cat_features_modified)
    with cols[1]:
        plot_pie(df)

    st.markdown("<h1 style='font-family:Lucida Caligraphy;font-size:35px ;text-align: left;'> Multivariate Analysis </h1>", unsafe_allow_html=True)
    st.subheader(" ")

    plot_kde_by_response(df)
    
    cols = st.columns(2)
    with cols[0]:
        plot_heatmap(df, num_features)
    with cols[1]:
        plot_cramers_v_heatmap(df, cat_features)
    
with tabs[1]:
    st.markdown("<h1 style='font-family:Lucida Caligraphy;font-size:30px;color:DarkSlateBlue;text-align: center;'> Statistics for numeric variables </h1>", unsafe_allow_html=True)
    st.table(df[num_features].describe())

    st.markdown("<h1 style='font-family:Lucida Caligraphy;font-size:30px;color:DarkSlateBlue;text-align: center;'> Statistics for categorical variables </h1>", unsafe_allow_html=True)
    st.table(df[cat_features].astype(str).describe())    


