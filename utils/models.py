"""Functions that are used during modelling step"""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


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
    num_transformer = SimpleImputer(strategy='median')
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


def train_model(preprocessor, model, X_train, y_train):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
        ])
    pipeline.fit(X_train, y_train)
    return pipeline


def get_metrics(y_true, y_pred):
    accuracy = round(accuracy_score(y_true, y_pred), 2)
    classif = pd.DataFrame(
        classification_report(y_true, y_pred, output_dict=True)
    ).transpose().iloc[:2, :]
    return accuracy, classif
