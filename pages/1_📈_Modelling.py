"""Second page of the App"""
import streamlit as st

from src.app.utils import load_css, make_config
from src.models.preprocessing import prepare_data
import src.app.page1 as p1
import src.models.evaluation as eval
from src.etl import get_params

make_config()
load_css('static/styles.css')

st.header("Machine and Deep Learning Models")
st.subheader("Experimentation Results")

tabs = st.tabs(["Machine Learning", "Deep Learning"])

X_train, X_test, y_train, y_test = prepare_data()

models = ["Logistic Regression", "Random Forest"]
methods = [None, "SMOTE", "OVER"]
df_combined, classif_reports, best_params = p1.cache_numerical_results(
    models, methods, X_test, y_test
    )

logistic_params = get_params('grid', models[0], 'conf/params.yaml')
random_forest_params = get_params('grid', models[1], 'conf/params.yaml')
with tabs[0]:
    st.markdown("""
                As we see in the first section, we face a binary classification
                with class imbalanced. So we tried to find a good model to predict
                our label that is the variable. \n
                """, unsafe_allow_html=True)
    st.markdown(
        """<h1 class='h1-preprocessing-steps'> Preprocessing </h1> \n""", unsafe_allow_html=True
        )
    st.markdown("""
                <div class="h1-details">
                From the foregoing, we need to process our variables before training a model.
                <ul>
                    <li>Handling missing values we will <b>imputate by mean or mode</b></li>
                    <li>For numerical variables, we will <b>normalize</b> them to prevent assigning too much weight to large values</li>
                    <li>For categorical variables, <b>one hot encoding</b> will be necessary</li>
                    <li>Given that the target variable is imbalanced, it can be managed directly through parameters in the models or by employing <b>resampling methods</b> to balance it</li>
                    <li>Finally, for any chosen model, we will train the model using <b>cross-validation</b> to find the best parameters and thereby reduce overfitting</li>
                </ul>
            </div> \n
                """, unsafe_allow_html=True)

    if st.checkbox('Show Numerical features distributions after normalization'):
        st.subheader('KDE plot for normalized features')
        p1.plot_kde_after_normalization(X_train)

    st.markdown(
        """<h1 class='h1-preprocessing-steps'> Model Choice </h1>""", unsafe_allow_html=True)

    st.markdown("""<h1 class='h1-sub-title'> Metrics </h1>""", unsafe_allow_html=True)

    st.markdown("""
            <div class="h1-details">
            To compare and select our final model, we use two key metrics :
            <ul>
                <li><b>Roc Auc Score : </b> Indicates how effectively our model distinguishes between classes.</li>
                <li><b>Recall : </b> Measures the proportion of actual positives correctly identified by the model.</li>
            </ul>
                In the context of prospecting, prioritizing a model that identifies all individuals with a high likelihood
                of subscription is preferable, even at the cost of including some who won't subscribe.
                This approach ensures that no potential subscriber is overlooked.
            </div> \n
            """, unsafe_allow_html=True)

    st.markdown(
        """<h1 class='h1-sub-title'> Cross-Validation</h1> \n""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        html_content_logistic = p1.generate_model_params_html(
            models[0], logistic_params, best_params[models[0]]
            )
        st.markdown(html_content_logistic, unsafe_allow_html=True)

    with col2:
        html_content_rf = p1.generate_model_params_html(
            models[1], random_forest_params, best_params[models[1]]
            )
        st.markdown(html_content_rf, unsafe_allow_html=True)

    st.markdown(
        """<h1 class='h1-sub-title'> Models results on the test set</h1> \n""",
        unsafe_allow_html=True)

    classifs = st.columns(2, gap="medium")
    for model_idx, model in enumerate(models):
        with classifs[model_idx]:
            for method in methods:
                full_name = f"{model} (resampling = {str(method)})"
                st.markdown(f"{full_name}")
                st.dataframe(classif_reports[model][method])
    st.table(df_combined)

    st.markdown(
        """\n <h1 class='h1-sub-title'> If we have to choose ... </h1> \n """,
        unsafe_allow_html=True)

    choices = st.columns(2)
    with choices[0]:
        model_choice = st.radio("Choose model", models)
    with choices[1]:
        method_choice = st.radio("Choose resampling method", methods)
    model = eval.get_model(model_choice, method_choice)

    st.divider()
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        eval.display_confusion_matrix(model, X_test, y_test)
    with col2:
        eval.get_roc_auc(model, X_test, y_test)

    if "Random Forest" in model_choice:
        st.divider()
        cols = st.columns(2, gap="medium")
        with cols[0]:
            top_importances, top_feature_names = eval.get_feature_importance_names(model)
            eval.plot_feature_importance(top_importances, top_feature_names)
        with cols[1]:
            eval.plot_model_tree(model)

with tabs[1]:
    st.markdown(
        "Details about the deep learning model can be found [here](https://github.com/JulesBrable/ai_insurance/blob/Leela/Neural_Network.ipynb)"
        )
