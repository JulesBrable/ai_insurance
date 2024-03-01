"""Third page of the App"""
import streamlit as st
from sklearn.model_selection import train_test_split

from utils.etl import load_data, get_features_by_type
from utils.models import create_preprocessor, train_model, get_metrics
from utils.utils import manage_sidebar, to_excel, download_results, get_model_info

st.title("Machine Learning Model Configuration")

SEED = 42

df = load_data("https://minio.lab.sspcloud.fr/jbrablx/ai_insurance/raw/train.csv")
df.drop(['id'], axis=1, inplace=True)

selected_features, model_type, model, model_kwargs, test_size = manage_sidebar(df, SEED)

if selected_features:
    num_features, cat_features = get_features_by_type(df, selected_features)

preprocessor = create_preprocessor(num_features, cat_features)

X = df[selected_features]
y = df["Response"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED)

if st.button("Train Model"):
    pipeline = train_model(preprocessor, model, X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy, class_report = get_metrics(y_test, y_pred)
    st.markdown(f"**Accuracy Score:** {accuracy}")
    st.markdown("**Classification Report:**")
    st.dataframe(class_report)

    # export results
    model_info = get_model_info(model_type, model_kwargs, SEED)
    df_xlsx = to_excel(class_report, model_info)
    download_results(df_xlsx)
