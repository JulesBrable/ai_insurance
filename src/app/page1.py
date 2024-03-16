import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import src.models.evaluation as eval


def get_numerical_results(model_name, method, X_test, y_test):
    model = eval.get_model(model_name, method)
    y_proba, y_pred = eval.predict_from_fitted_model(model, X_test)
    metrics_df, classif_report = eval.get_metrics(y_test, y_pred, y_proba)
    return metrics_df, classif_report


def get_best_params(model_name, method):
    model = eval.get_model(model_name, method)
    best_params = model['model'].get_params()
    return best_params


@st.cache_resource
def cache_numerical_results(models, methods, X_test, y_test):
    dfs = []
    classif_reports = {}
    best_params = {}
    with st.spinner("_Loading Results..._", cache=True):
        for model_idx, model in enumerate(models):
            classif_reports[model] = {}
            best_params[model] = {}
            for method in methods:
                metrics, classif_reports[model][method] = get_numerical_results(
                    model, method, X_test, y_test
                    )
                full_name = f"{model} (resampling = {str(method)})"
                metrics.set_index('Metric', inplace=True)
                metrics.columns = [full_name]
                dfs.append(metrics)
                best_params[model][method] = get_best_params(model, method)
        df_combined = pd.concat(dfs, axis=1)
        return df_combined, classif_reports, best_params


def plot_kde_after_normalization(df):
    df_normalized = df.copy()
    scaler = StandardScaler()
    columns_to_normalize = ['Age', 'Annual_Premium', 'Vintage']
    df_normalized[columns_to_normalize] = scaler.fit_transform(df_normalized[columns_to_normalize])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, x in enumerate(columns_to_normalize):
        sns.kdeplot(data=df_normalized, x=x, ax=axes[i], palette="coolwarm", common_norm=False)
        axes[i].set_xlabel(x)
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'Distribution of {x} by Response')
    plt.tight_layout()

    return st.pyplot(fig)


def generate_model_params_html(model_name, model_params, best_params):
    html_content = f"""
    <div style="background-color: #FFCCCC; padding: 10px; border-radius: 5px;
    font-family: Lucida Caligraphy;">
        <p style="text-decoration: underline;text-align: center;font-size: 18px;">
                <b>{model_name}</b>
        </p>
        <p> Optimized parameters & GridSearch : </p>
        <ul>
            <li><b>Resampling Methods :</b> None, SmoteNC, RandomOverSampling</li>"""

    cleaned_params = []
    for param, values in model_params.items():
        cleaned_param = param.replace('model__', '')
        html_content += f"<li><b>{cleaned_param} :</b> {', '.join(map(str, values))}</li>"
        cleaned_params.append(cleaned_param)

    html_content += "</ul><p> Best Model Parameters : </p><ul>"
    for method, params in best_params.items():
        method_name = "None" if method is None else method
        params_list = ', '.join([
            f"{k} = {v}" for k, v in params.items() if k in cleaned_params
            ])
        html_content += f"<li><b>{method_name} :</b> {params_list}</li>"

    html_content += "</ul></div>"
    return html_content
