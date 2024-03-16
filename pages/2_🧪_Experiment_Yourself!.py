"""Third page of the App"""
import streamlit as st

from src.etl import load_data
from src.models.preprocessing import create_preprocessor, split_data
from src.models.training import train_model_oneshot
from src.models.evaluation import predict_from_fitted_model, get_metrics
from src.app.page2 import manage_sidebar, to_excel, download_results, get_model_info
from src.app.utils import make_config, get_names_by_type


make_config()
st.title("Machine Learning Model Configuration")

SEED = 42

df = load_data()
selected_features, model_type, model, model_kwargs, test_size, method = manage_sidebar(df, SEED)

if selected_features:
    num_features, cat_features = get_names_by_type()
    preprocessor = create_preprocessor(num_features, cat_features)
    X_train, X_test, y_train, y_test = split_data(df, test_size, selected_features)

if st.button("Train Model"):
    with st.spinner(r"_🧠 Training model..._"):
        pipeline = train_model_oneshot(
            model, X_train, y_train, preprocessor, resampling_method=method
            )
        y_proba, y_pred = predict_from_fitted_model(pipeline, X_test)
        metrics_df, classif_report = get_metrics(y_test, y_pred, y_proba)
    cols = st.columns(2, gap="small")
    with cols[0]:
        st.dataframe(metrics_df, hide_index=True)
    with cols[1]:
        st.dataframe(classif_report)

    # export results
    model_info = get_model_info(model_type, model_kwargs, SEED)
    df_xlsx = to_excel(classif_report, metrics_df, model_info)
    download_results(df_xlsx)
