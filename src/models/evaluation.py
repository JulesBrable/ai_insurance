import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_curve, roc_auc_score, f1_score
)
from sklearn.tree import plot_tree
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.preprocessing import get_feature_names

sns.set_style("whitegrid")


def get_model(model_name, method, remote="https://minio.lab.sspcloud.fr/jbrablx/ai_insurance"):
    method = str(method).lower()
    with open(f'{remote}/outputs/{model_name}/resampling_{method}.pkl', 'rb') as f:
        model_used = pickle.load(f)
    return model_used


def predict_from_fitted_model(model, X_test, threshold: float = .5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = np.where(y_proba > threshold, 1, 0)
    return y_proba, y_pred


def get_metrics(y_test, y_pred, y_proba, threshold: float = .5):
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy Score", "ROC AUC Score", "F1 Score"],
        "Value": [f"{accuracy:.2f}", f"{roc_auc:.2f}", f"{f1:.2f}"],
    })
    classif_report = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True)
    ).transpose().iloc[:2, :]
    return metrics_df, classif_report


def get_roc_auc(y_test, y_proba):
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc="lower right")
    return fig


def get_feature_importance_names(specific_model, model, n: int = 10):
    importances = specific_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    feature_names_transformed = get_feature_names(model.named_steps['preprocessor'])

    top_indices = indices[:n]
    top_importances = importances[top_indices]
    top_feature_names = [feature_names_transformed[i] for i in top_indices]
    return top_importances, top_feature_names


def plot_feature_importance(top_importances, top_feature_names):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Top 10 Feature Importances")
    ax.bar(range(len(top_importances)), top_importances, color="r", align="center")
    ax.set_xticks(range(len(top_importances)))
    ax.set_xticklabels(top_feature_names, rotation=45, ha="right")
    ax.set_xlim([-1, len(top_importances)])
    plt.tight_layout()
    return fig


def plot_model_tree(specific_model, model, max_depth=3):
    estimator = specific_model.estimators_[0]
    feature_names_transformed = get_feature_names(model.named_steps['preprocessor'])
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(estimator, feature_names=feature_names_transformed, class_names=["Class0", "Class1"],
              filled=True, impurity=True, rounded=True, max_depth=max_depth, ax=ax)
    return fig
