"""Script to train a model"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from models.preprocessing import create_preprocessor
from models.training import get_model_configs, train_model_gscv, save_model_pickle
from etl import get_params
from app.utils import get_names_by_type

parser = argparse.ArgumentParser()
parser.add_argument(
    "--methods",
    choices=["SMOTE", "OVER", ["SMOTE", "OVER", None]],
    default=["SMOTE", "OVER", None],
    help="Resampling methods to apply"
    )
parser.add_argument(
    "--model",
    choices=['Random Forest', 'Logistic Regression'],
    default='Random Forest',
    help="Machine Learning model to train"
)
args = parser.parse_args()
methods = args.methods
model = args.model
SEED = 42

df = pd.read_csv("https://minio.lab.sspcloud.fr/jbrablx/ai_insurance/raw/train.csv")

df.drop(['id'], axis=1, inplace=True)
df.drop_duplicates(inplace=True)

X = df.drop('Response', axis=1)
y = df['Response']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_features, cat_features = get_names_by_type()

params_grid = get_params('grid', model)
params_skf = get_params('skf')
params_gscv = get_params('gscv')

preprocessor = create_preprocessor(num_features, cat_features)
model_instance = get_model_configs()[model]['model'](random_state=SEED)

for method in list(methods):
    grid_search_fitted = train_model_gscv(
        model=model_instance,
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        param_grid=params_grid,
        params_skf=params_skf,
        params_gscv=params_gscv,
        resampling_method=method
    )

    best_model = grid_search_fitted.best_estimator_
    save_model_pickle(model, method, best_model)
