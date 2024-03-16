"""Second page of the App"""
import streamlit as st
from sklearn.model_selection import train_test_split

from src.app.utils import load_css, make_config
from src.etl import load_data
from src.app.page1 import display_numerical_results

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


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


make_config()
load_css('static/styles.css')

st.header("Machine and Deep Learning Models")
st.subheader("Experimentation Results")

tabs = st.tabs(["Machine Learning", "Deep Learning"])

df = load_data()
X = df.drop('Response', axis=1)
y = df['Response']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with tabs[0]:

    st.markdown("""
                As we see in the first section, we face a binary classification
                with class imbalanced. So we tried to find a good model to predict
                our label that is the variable.
                """, unsafe_allow_html=True)
    st.subheader(" ")

    num_features = ['Age', 'Annual_Premium', 'Vintage']
    cat_features = [
        'Gender', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Damage',
        'Policy_Sales_Channel', 'Vehicle_Age'
        ]

    st.markdown(
        """<h1 class='h1-preprocessing-steps'> Preprocessing </h1>""", unsafe_allow_html=True
        )
    st.subheader(" ")

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
                </div>
                """, unsafe_allow_html=True)

    st.subheader(" ")

    if st.checkbox('Show Numerical features distributions after normalization'):
        st.subheader('KDE plot for normalized features')
        plot_kde_after_normalization(df)

    st.markdown("""<h1 class='h1-preprocessing-steps'> Model Choice </h1>""",
            unsafe_allow_html=True)

    st.markdown("""<h1 class='h1-sub-title'> Metrics </h1>""",
            unsafe_allow_html=True)

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
            </div>
            """, unsafe_allow_html=True)
    st.subheader(" ")

    st.markdown("""<h1 class='h1-sub-title'> Cross-Validation</h1>""",
            unsafe_allow_html=True)
    st.subheader(" ")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background-color: #FFCCCC; padding: 10px; border-radius: 5px; font-family: Lucida Caligraphy;">
            <p style="text-decoration: underline;text-align: center;font-size: 18px;">
                    <b>Logistic Regression</b>
            </p>
            <p> Optimized parameters : </p>
                <ul>
                    <li><b>Resampling Methods : </b> None, SmoteNC, RandomOverSampling</li>
                    <li><b>C : </b> 0.001, 0.01, 0.1, 1, 10, 100 </li>
                    <li><b>solver : </b> 'newton-cg', 'lbfgs', 'liblinear' </li>
                    <li><b>max_iter : </b>100, 200, 500 </li>
                </ul> 
            <p> Best Model Parameters : </p>  
                <ul>
                    <li><b>None : </b> C =  0.1, max_iter = 100, solver ='newton-cg'
                    Roc AUC score: 0.85 </li>
                    <li><b>Smote :</b> C = 0.1, max_iter = 100, solver ='lbfgs'}
                    Roc AUC score: 0.90 </li>
                    <li><b>OverSampling : </b> C = 10, max_iter = 100, solver = 'liblinear'}
                    Roc AUC score: 0.85 </li>
                </ul>   
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background-color: #FFCCCC; padding: 10px; border-radius: 5px; font-family: Lucida Caligraphy;">
            <p style="text-decoration: underline;text-align: center;font-size: 18px;">
                    <b>Random Forest</b>
            </p>
            <p> Optimized parameters : </p>
                <ul>
                    <li><b>Resampling Methods : </b> None, SmoteNC, RandomOverSampling</li>
                    <li><b>n_estimators : </b> 50, 100, 200</li>
                    <li><b>max_depth : </b> None, 5, 10, 20 </li>
                    <li><b>min_samples_split : </b> 2, 5, 10 </li>
                </ul> 
            <p> Best Model Parameters : </p> 
                <ul>
                    <li><b>None : </b> max_depth =20, min_samples_split= 10, _n_estimators = 200
                    Roc AUC score: 0.85</li>
                    <li><b>Smote :</b> max_depth = None, min_samples_split=10, n_estimators =200
                    Roc AUC score: 0.92</li>
                    <li><b>OverSampling : </b> max_depth = None, min_samples_split = 2, n_estimators = 200
                    Roc AUC score : 0.991</li>
                </ul>  
        </div>
        """, unsafe_allow_html=True)
    st.subheader(" ")

    st.markdown("""<h1 class='h1-sub-title'> Models results </h1>""",
            unsafe_allow_html=True)
    st.subheader(" ")

    models = ["Logistic Regression", "Random Forest"]
    methods = [None, "SMOTE", "OVER"]
    import pandas as pd
    dfs = []
    for mod in models:
        for method in methods:
            metrics = display_numerical_results(mod, method, X_test, y_test)
            full_name = mod + " (resampling = " + str(method) + ")"
            metrics.set_index('Metric', inplace=True)
            metrics.columns = [full_name]
            dfs.append(metrics)
    df_combined = pd.concat(dfs, axis=1)
    st.table(df_combined)
    # for model_name in models:
    #     for method in methods:
    #         set_index('Metric', inplace=True)
    #     mods = st.columns(2)
    #     with mods[0]:
    #         st.markdown(f"**Model = {model_name}, resampling = SMOTE**")
    #         display_numerical_results(model_name, "SMOTE", X_test, y_test)
    #     with mods[1]:
    #         st.markdown(f"**Model = {model_name}, resampling = OVER**")
    #         display_numerical_results(model_name, "OVER", X_test, y_test)
    #     with st.columns(2)[0]:
    #         st.markdown(f"**Model = {model_name}, resampling = None**")
    #         display_numerical_results(model_name, None, X_test, y_test)

    # with mod1 :
    #     st.dataframe(p1.lr_df, hide_index=True)
    # with mod2 :
    #     st.dataframe(p1.lr_smote_df, hide_index=True)
    # with mod3 :
    #     st.dataframe(p1.lr_over_df, hide_index=True)
    # with mod4 :
    #     st.dataframe(p1.rf_df, hide_index=True)
    # with mod5 :
    #     st.dataframe(p1.rf_smote_df, hide_index=True)
    # with mod6 :
    #     st.dataframe(p1.rf_over_df, hide_index=True)
    
    st.subheader(" ")
    
    st.markdown("""<h1 class='h1-sub-title'> If we have to choose ... </h1>""",
            unsafe_allow_html=True)
    st.subheader(" ")

    col1, col2 = st.columns(2)
    # with col1:
    #     st.image('images/cm_lr_over.png',use_column_width ='always')
    # with col2:
    #     st.image('images/roc_lr_over.png',use_column_width ='always')

    from sklearn.metrics import confusion_matrix, roc_auc_score
    import src.models.evaluation as eval

    def display_confusion_matrix(model_name, method, X_test):
        model = eval.get_model(model_name, method)
        y_proba, y_pred = eval.predict_from_fitted_model(model, X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"], ax=ax
            )
        ax.set_title("Confusion Matrix")
        ax.set_ylabel("Actual label")
        ax.set_xlabel("Predicted label")
        return st.pyplot(fig)

    if st.checkbox('Display results for each model'):
        # options = [
        #     "Logistic Regression",
        #     "Logistic Regression with SMOTENC",
        #     "Logistic Regression with RandomOverSampling",
        #     "Random Forest",
        #     "Random Forest with RandomOverSampling",
        #     "Random Forest with SMOTENC"
        # ]

        model_choice = st.radio("Choose model", models)
        method_choice = st.radio("Choose resampling method", methods)

        display_confusion_matrix(model_name, method_choice, X_test)
        if model_choice == "Logistic Regression":
            col1, col2 = st.columns(2)
            with col1:
                display_confusion_matrix(model_name, method_choice, X_test)
            #     st.image('images/cm_lr.png',use_column_width ='always')
            # with col2:
            #     st.image('images/roc_lr.png',use_column_width ='always')
            pass
        elif model_choice == "Logistic Regression with SMOTENC":
            col1, col2 = st.columns(2)
            with col1:
                st.image('images/cm_lr_smote.png',use_column_width ='always')
            with col2:
                st.image('images/roc_lr_smote.png',use_column_width ='always')
        
        elif model_choice == "Logistic Regression Classifier with RandomOverSampling":
            col1, col2 = st.columns(2)
            with col1:
                st.image('images/cm_lr_over.png',use_column_width ='always')
            with col2:
                st.image('images/roc_lr_over.png',use_column_width ='always')
        
        elif model_choice == "Random Forest":
            col1, col2 = st.columns(2)
            with col1:
                st.image('images/roc_rf.png',use_column_width ='always')
            with col2:
                st.image('images/imp_rf.png',use_column_width ='always')
            st.image('images/tree_rf.png',use_column_width ='always')
        
        elif model_choice == "Random Forest with RandomOverSampling":
            col1, col2 = st.columns(2)
            with col1:
                st.image('images/roc_rf_over.png',use_column_width ='always')
            with col2:
                st.image('images/imp_rf_over.png',use_column_width ='always')
            st.image('images/tree_rf_over.png',use_column_width ='always')
        
        elif model_choice == "Random Forest with SMOTENC":
            col1, col2 = st.columns(2)
            with col1:
                st.image('images/roc_rf_smote.png',use_column_width ='always')
            with col2:
                st.image('images/imp_rf_smote.png',use_column_width ='always')
            st.image('images/tree_rf_smote.png',use_column_width ='always')


with tabs[1]:
    st.write("Deep Learning")
