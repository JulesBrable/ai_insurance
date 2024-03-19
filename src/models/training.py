"""Functions that are used during modelling step"""
import pickle
import os
import logging
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.app.utils import get_names_by_type


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
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier,
            "params": [
                {"name": "C", "type": "slider", "min": 0.01, "max": 10.0, "default": 1.0}
            ]
        }
    }


def fit_smote(X_train, y_train, cat_features):
    print("SMOTE")
    print("--------")
    smote = SMOTENC(random_state=42, k_neighbors=15, categorical_features=cat_features)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    return X_train, y_train


def fit_over(X_train, y_train):
    print("RandomOverSampler")
    print("--------")
    rs = RandomOverSampler(random_state=42)
    X_train, y_train = rs.fit_resample(X_train, y_train)
    return X_train, y_train


def build_pipeline(preprocessor, model):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    return pipeline


def save_model_pickle(model_name, method, fitted_pipeline):
    if method not in ["SMOTE", "OVER", None]:
        raise ValueError("Invalid method. Expected 'SMOTE', 'OVER', or None.")
    if fitted_pipeline is None:
        raise ValueError("Pipeline object is None.")

    file_name_suffix = (
        "resampling_smote.pkl" if method == "SMOTE"
        else "resampling_over.pkl" if method == "OVER"
        else "resampling_none.pkl"
    )

    directory = f"../data/outputs/{model_name}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = f"{directory}/{file_name_suffix}"

    with open(file_path, 'wb') as file:
        pickle.dump(fitted_pipeline, file)

    logging.info(f"Model saved successfully to {file_path}")


def build_fit_gcv(X_train, y_train, model_pipe, param_grid, params_skf, params_gscv):
    """
    Get the best model using a StratifiedKold & GridSearchCV
    """

    cv = StratifiedKFold(**params_skf)

    grid_search = GridSearchCV(
        model_pipe,
        param_grid,
        cv=cv,
        **params_gscv
    )

    grid_search.fit(X_train, y_train)
    return grid_search


def train_model_gscv(
    model,
    X_train,
    y_train,
    preprocessor,
    param_grid,
    params_skf,
    params_gscv,
    resampling_method=None
        ):
    """
    Train model with StratifiedKFold and fine tune hyperparams using GridSearchCV
    """
    if resampling_method == "SMOTE":
        num_features, cat_features = get_names_by_type()
        X_train, y_train = fit_smote(X_train, y_train, cat_features)

    elif resampling_method == "OVER":
        X_train, y_train = fit_over(X_train, y_train)
    else:
        pass

    pipeline = build_pipeline(preprocessor, model)
    grid_search_fitted = build_fit_gcv(
        X_train, y_train, pipeline, param_grid, params_skf, params_gscv
        )
    return grid_search_fitted


def train_model_oneshot(
    model,
    X_train,
    y_train,
    preprocessor,
    resampling_method=None
        ):
    if resampling_method == "SMOTE":
        num_features, cat_features = get_names_by_type()
        X_train, y_train = fit_smote(X_train, y_train, cat_features)

    elif resampling_method == "OVER":
        X_train, y_train = fit_over(X_train, y_train)
    else:
        pass

    pipeline = build_pipeline(preprocessor, model)
    pipeline.fit(X_train, y_train)
    return pipeline
