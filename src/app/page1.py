import src.models.evaluation as eval
from src.app.utils import display_classif_metrics


def display_numerical_results(model_name, method, X_test, y_test):
    model = eval.get_model(model_name, method)
    y_proba, y_pred = eval.predict_from_fitted_model(model, X_test)
    metrics_df, classif_report = eval.get_metrics(y_test, y_pred, y_proba)
    return metrics_df
    #display_classif_metrics(metrics_df, classif_report)
