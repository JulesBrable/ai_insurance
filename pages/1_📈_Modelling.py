"""Second page of the App"""
import streamlit as st

from sklearn.model_selection import train_test_split
from src import models as md

from src.app.utils import load_css
from src.etl import load_data

load_css('static/styles.css')

st.header("Machine and Deep Learning Models")
st.subheader("Experimentation Results")

tabs = st.tabs(["Machine Learning", "Deep Learning"])

df = load_data()

# Split dataset
X = df.drop('Response', axis=1)  
y = df['Response']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

with tabs[0]:

    st.markdown("""
                As we see in the first section, we face a binary classification
                with class imbalanced. So we tried to find a good model to predict
                our label that is the variable.
                """, unsafe_allow_html=True)
    st.subheader(" ")

    num_features = ['Age', 'Annual_Premium', 'Vintage']
    cat_features = ['Gender', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Damage', 'Policy_Sales_Channel', 'Vehicle_Age']

    # Preprocessing
    preprocessor = md.create_preprocessor(num_features, cat_features)

    
    # Choose model and method

    options = [
        "Logistic Regression Classifier",
        "Logistic Regression Classifier with SMOTENC",
        "Logistic Regression Classifier with RandomOverSampling",
        "Random Forest Classifier",
        "Random Forest Classifier with RandomOverSampling",
        "Random Forest Classifier with SMOTENC"
    ]

    # Let the user choose a model and method
    model_choose = st.radio("Choose model and method", options)

    ## Display model results
    if model_choose == "Logistic Regression Classifier":
        md.get_metrics("logisticregression", X_val, y_val, 0.52, show_roc=True, show_features_importance=True, show_tree=True)
    
    elif model_choose == "Logistic Regression Classifier with SMOTENC":
        md.get_metrics("logisticregression_smote", X_val, y_val, 0.52, show_roc=True, show_features_importance=True, show_tree=True)
    
    elif model_choose == "Logistic Regression Classifier with RandomOverSampling":
       md.get_metrics("logisticregression_over", X_val, y_val, 0.52, show_roc=True, show_features_importance=True, show_tree=True)
    
    elif model_choose == "Random Forest Classifier":
       md.get_metrics("randomforestclassifier", X_val, y_val, 0.52, show_roc=True, show_features_importance=True, show_tree=True)
    
    elif model_choose == "Random Forest Classifier with RandomOverSampling":
       md.get_metrics("randomforestclassifier_over", X_val, y_val, 0.52, show_roc=True, show_features_importance=True, show_tree=True)
    
    elif model_choose == "Random Forest Classifier with SMOTENC":
       md.get_metrics("randomforestclassifier_smote", X_val, y_val, 0.52, show_roc=True, show_features_importance=True, show_tree=True)
    


with tabs[1]:
    st.write("Deep Learning")


