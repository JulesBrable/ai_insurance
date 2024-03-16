"""Third page of the App"""
import streamlit as st
from sklearn.model_selection import train_test_split

from src.etl import load_data, get_features_by_type
from src.models.preprocessing import create_preprocessor
from src.models.training import train_model_oneshot
from src.models.evaluation import predict_from_fitted_model, get_metrics
from src.app.page2 import manage_sidebar, to_excel, download_results, get_model_info
from src.app.utils import make_config, display_classif_metrics


make_config()
st.title("Machine Learning Model Configuration")

SEED = 42

df = load_data()

selected_features, model_type, model, model_kwargs, test_size, method = manage_sidebar(df, SEED)

if selected_features:
    num_features = ['Age', 'Annual_Premium', 'Vintage']  
    cat_features = ['Gender', 'Vehicle_Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Damage', 'Policy_Sales_Channel'] 

preprocessor = create_preprocessor(num_features, cat_features)

X = df[selected_features]
y = df["Response"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED)

if st.button("Train Model"):
    with st.spinner(r"_🧠 Training model..._"):
        pipeline = train_model_oneshot(
            model, X_train, y_train, preprocessor, resampling_method=method
            )
        y_proba, y_pred = predict_from_fitted_model(pipeline, X_test)
        metrics_df, classif_report = get_metrics(y_test, y_pred, y_proba)

    display_classif_metrics(metrics_df, classif_report)

    # export results
    model_info = get_model_info(model_type, model_kwargs, SEED)
    df_xlsx = to_excel(classif_report, metrics_df, model_info)
    download_results(df_xlsx)
