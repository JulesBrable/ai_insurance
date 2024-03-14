"""Functions that are used during modelling step"""
import pandas as pd
import streamlit as st
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")
import scipy.stats as ss

from imblearn.over_sampling import SMOTENC, RandomOverSampler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV


def get_model_configs():
    return {
        "Random Forest": {
            "model": RandomForestClassifier,
            "params": [
                {"name": "n_estimators", "type": "slider", "min": 10, "max": 500, "default": 100},
                {"name": "max_depth", "type": "slider", "min": 1, "max": 100, "default": 20}
            ]
        },
        "Logistic Regression": {
            "model": LogisticRegression,
            "params": [
                {"name": "C", "type": "slider", "min": 0.01, "max": 10.0, "default": 1.0}
            ]
        }
    }


def create_preprocessor(num_features, cat_features):
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ])
    return preprocessor

def get_feature_names(column_transformer):
    """Obtient les noms des caractéristiques après transformation par ColumnTransformer"""
    output_features = []
    
    for name, pipe, features in column_transformer.transformers_:
        if name == "remainder":
            continue  # ne pas traiter les colonnes 'remainder'
        if hasattr(pipe, 'get_feature_names_out'):
            # Pour les transformateurs avec cette méthode (version récente de sklearn)
            output_features.extend(pipe.get_feature_names_out(features))
        elif hasattr(pipe, 'get_feature_names'):
            # Pour OneHotEncoder dans les versions antérieures de sklearn
            output_features.extend(pipe.get_feature_names(features))
        else:
            # Si le transformateur ne modifie pas les noms des caractéristiques, les renvoyer tels quels
            output_features.extend(features)
    
    return output_features

def train_model(model, X_train, y_train, preprocessor, method=None):
    """
    Trains a model with optional resampling methods and saves it into a pickle file.
    
    Args:
        model: The machine learning model to be trained.
        X_train: Training features DataFrame.
        y_train: Training target vector.
        method (optional): Resampling method to apply. Options are "SMOTE" or "OVER". If None, no resampling is applied.
    
    The function supports categorical feature handling for SMOTE resampling with predefined categorical features. 
    It defines a pipeline with a preprocessor (assumed to be defined outside this function) and the given model, 
    fits the pipeline to the training data, and then saves the trained model into a pickle file based on the model 
    and method used.
    """
    # Decide whether to resample or not
    if method == "SMOTE":
        cat_features = ['Gender', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Damage', 'Policy_Sales_Channel', 'Vehicle_Age']
        smote = SMOTENC(random_state=42, k_neighbors=15, categorical_features=cat_features)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    elif method == "OVER":
        rs = RandomOverSampler(random_state=42)
        X_train, y_train = rs.fit_resample(X_train, y_train)

    else:
        pass

    # Choose the correct param_grid
    if isinstance(model, RandomForestClassifier):
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 5, 10, 20],
            'model__min_samples_split': [2, 5, 10]
        }
    elif isinstance(model, LogisticRegression):
        param_grid = {
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'model__solver': ['newton-cg', 'lbfgs', 'liblinear'],
            'model__max_iter' : [100, 200, 500]
        }
    else:
        raise ValueError("Grid search parameters are not defined for this model type.")

    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    
    print("Performing grid search...")

    # Define pipeline and fit model
    pipeline = GridSearchCV(pipeline, param_grid, cv=5, verbose = 3, n_jobs=-1, scoring=['recall', 'roc_auc'], refit='roc_auc')
    pipeline.fit(X_train, y_train)  
    print(f"Best params: {pipeline.best_params_}")  
    print(f"Best cross-validation score: {pipeline.best_score_}")  

    # Save model into a pickle file
    model_name = type(model).__name__.lower()  # Adjusting to dynamically capture the model's name
    file_name_suffix = "_smote.pkl" if method == "SMOTE" else "_over.pkl" if method == "OVER" else ".pkl"
    file_path = f'pickle/{model_name}{file_name_suffix}'

    with open(file_path, 'wb') as file:
        pickle.dump(pipeline, file)
    return


def get_metrics(model, X_val, y_val, threshold):
    """
    Loads a model from a pickle file, makes predictions, and evaluates several metrics.
    
    Args:
        data: Dataset used for plotting feature importances and decision trees. Should have `feature_names` and `target_names`.
        model: The filename (without '.pkl' extension) of the model to load and evaluate.
        X_val: Validation features DataFrame.
        y_val: Validation target vector.
        threshold: The threshold for converting probabilities to class labels.
        show_roc (bool, optional): If True, plots the ROC curve. Defaults to False.
        show_features_importance (bool, optional): If True, plots feature importances. Defaults to False.
        show_tree (bool, optional): If True, plots a decision tree from the model. Defaults to False.

    Returns:
        Tuple of arrays: (y_pred_threshold, y_proba), where `y_pred_threshold` are the predictions made using the specified threshold,
        and `y_proba` are the model's class probabilities.
    """

    # Load model from pickle file
    with open(f'pickle/{model}.pkl', 'rb') as fichier:
        model_used = pickle.load(fichier)

    # Predict probabilities
    y_proba = model_used.predict_proba(X_val)[:, 1]

    # Predict labels based on threshold
    y_pred_threshold = np.where(y_proba > threshold, 1, 0)

    # For displaying metrics
    accuracy = accuracy_score(y_val, y_pred_threshold)
    roc_auc = roc_auc_score(y_val, y_proba)
    recall_class_1 = recall_score(y_val, y_pred_threshold, pos_label=1)
    recall_class_0 = recall_score(y_val, y_pred_threshold, pos_label=0)

    metrics_data = {
        "Metric": ["Accuracy Score", "ROC AUC Score", "Recall Class 1", "Recall Class 0", "Best cut-off"], 
        "Value": [f"{accuracy:.2f}", f"{roc_auc:.2f}", f"{recall_class_1:.2f}", f"{recall_class_0:.2f}", f"{threshold:.2f}"],
    }
    metrics_df = pd.DataFrame(metrics_data)

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred_threshold)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted 0", "Predicted 1"], yticklabels=["Actual 0", "Actual 1"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")

    # Dysplay results
    col1, col2 = st.columns(2)
    with col1 :
        st.dataframe(metrics_df)
    with col2 :
        st.pyplot(plt) 

    # Provide fpr, tpr for ROC curve
    fpr, tpr, thresholds = roc_curve(y_val, y_proba)

    # Get feature name
    feature_names_transformed = get_feature_names(model_used.named_steps['preprocessor'])
    
    # For ROC curve
    roc_auc = roc_auc_score(y_val, y_proba)
    st.set_option('deprecation.showPyplotGlobalUse', False) # To avoid warning messages for global use
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    st.pyplot()

    if hasattr(model_used, 'named_steps'):
        specific_model = model_used.named_steps['model']
        
    if hasattr(specific_model, 'feature_importances_'):
        importances = specific_model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Show the top 10 features
        top_indices = indices[:10]
        top_importances = importances[top_indices]
        top_feature_names = [feature_names_transformed[i] for i in top_indices]

        plt.figure(figsize=(10, 6))
        plt.title("Top 10 Feature Importances")
        plt.bar(range(len(top_importances)), top_importances, color="r", align="center")
        plt.xticks(range(len(top_importances)), top_feature_names, rotation=45, ha="right")
        
        plt.xlim([-1, len(top_importances)])
        plt.tight_layout()
        st.pyplot()

    if hasattr(specific_model, 'estimators_'):
        estimator = specific_model.estimators_[0]
        plt.figure(figsize=(20, 10))
        plot_tree(estimator, feature_names=feature_names_transformed, class_names=["Class0", "Class1"], filled=True, impurity=True, rounded=True, max_depth=3)
        st.pyplot()
    
    return