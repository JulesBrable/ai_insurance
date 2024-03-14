"""Second page of the App"""
import streamlit as st
from sklearn.model_selection import train_test_split
from PIL import Image
from src import models as md
from src.app.utils import load_css
import src.app.page1 as p1
from src.etl import load_data
load_css('static/styles.css')


st.markdown("""<h1 class='h1-title'> Machine and Deep Learning Models</h1>""",
            unsafe_allow_html=True)
st.subheader(" ")

tabs = st.tabs(["Machine Learning", "Deep Learning"])

# Data loading
df = load_data()

# Split dataset
X = df.drop('Response', axis=1)  
y = df['Response']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Data type
num_features = ['Age', 'Annual_Premium', 'Vintage']
cat_features = ['Gender', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Damage', 'Policy_Sales_Channel', 'Vehicle_Age']

# Preprocessing
preprocessor = md.create_preprocessor(num_features, cat_features)

#"""--------------------MACHINE LEARNING-------------------------"""

with tabs[0]:
    st.markdown("""<h1 class='h1-preprocessing-steps'> Preprocessing </h1>""",
            unsafe_allow_html=True)
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
        p1.plot_kde_after_normalization(df)

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

    # Models
    col1, col2 = st.columns(2)

    # Text block in the first column with a light red background
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
    # Text block in the second column with a light red background
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

    # Display model results
    mod1, mod2, mod3, mod4, mod5, mod6 = st.columns(6)

    with mod1 :
        st.dataframe(p1.lr_df, hide_index=True)
    with mod2 :
        st.dataframe(p1.lr_smote_df, hide_index=True)
    with mod3 :
        st.dataframe(p1.lr_over_df, hide_index=True)
    with mod4 :
        st.dataframe(p1.rf_df, hide_index=True)
    with mod5 :
        st.dataframe(p1.rf_smote_df, hide_index=True)
    with mod6 :
        st.dataframe(p1.rf_over_df, hide_index=True)
    
    st.subheader(" ")
    
    st.markdown("""<h1 class='h1-sub-title'> If we have to choose ... </h1>""",
            unsafe_allow_html=True)
    st.subheader(" ")

    col1, col2 = st.columns(2)
    with col1 :
        st.image('images/cm_lr_over.png',use_column_width ='always')
    with col2 :
        st.image('images/roc_lr_over.png',use_column_width ='always')


    if st.checkbox('Display results for each model'):
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
            col1, col2 = st.columns(2)
            with col1 :
                st.image('images/cm_lr.png',use_column_width ='always')
            with col2 :
                st.image('images/roc_lr.png',use_column_width ='always')
        
        elif model_choose == "Logistic Regression Classifier with SMOTENC":
            col1, col2 = st.columns(2)
            with col1 :
                st.image('images/cm_lr_smote.png',use_column_width ='always')
            with col2 :
                st.image('images/roc_lr_smote.png',use_column_width ='always')
        
        elif model_choose == "Logistic Regression Classifier with RandomOverSampling":
            col1, col2 = st.columns(2)
            with col1 :
                st.image('images/cm_lr_over.png',use_column_width ='always')
            with col2 :
                st.image('images/roc_lr_over.png',use_column_width ='always')
        
        elif model_choose == "Random Forest Classifier":
            col1, col2 = st.columns(2)
            with col1 :
                st.image('images/roc_rf.png',use_column_width ='always')
            with col2 :
                st.image('images/imp_rf.png',use_column_width ='always')
            st.image('images/tree_rf.png',use_column_width ='always')
        
        elif model_choose == "Random Forest Classifier with RandomOverSampling":
            col1, col2 = st.columns(2)
            with col1 :
                st.image('images/roc_rf_over.png',use_column_width ='always')
            with col2 :
                st.image('images/imp_rf_over.png',use_column_width ='always')
            st.image('images/tree_rf_over.png',use_column_width ='always')
        
        elif model_choose == "Random Forest Classifier with SMOTENC":
            col1, col2 = st.columns(2)
            with col1 :
                st.image('images/roc_rf_smote.png',use_column_width ='always')
            with col2 :
                st.image('images/imp_rf_smote.png',use_column_width ='always')
            st.image('images/tree_rf_smote.png',use_column_width ='always')

        else :
             pass

    #"""--------------------DEEP LEARNING-------------------------"""
    with tabs[1]:
        st.write("Deep Learning")