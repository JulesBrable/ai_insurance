from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.etl import load_data


def split_data(df, test_size, x_features, y_feature="Response", SEED=42):
    X = df[x_features]
    y = df[y_feature]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED
        )
    return X_train, X_test, y_train, y_test


def prepare_data(test_size: float = 0.2):
    df = load_data()
    x_features = list(df.drop(columns="Response", axis=1).columns)
    X_train, X_test, y_train, y_test = split_data(df, test_size, x_features)
    return X_train, X_test, y_train, y_test


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
            # Si le transfo ne modifie pas les noms des caractéristiques, les renvoyer tels quels
            output_features.extend(features)

    return output_features
