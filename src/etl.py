"""ETL functions"""
from typing import Union
import streamlit as st
import pandas as pd
import yaml


def get_params(
    params_type: str,
    model: Union[str, None] = None,
    filepath: str = '../conf/params.yaml'
        ):
    """
    Import parameters grid
    """
    with open(filepath, 'rb') as f:
        conf = yaml.safe_load(f.read())
        params = conf[params_type]
        if model:
            params = params[model]

            def replace_none_strings_with_none_type(d):
                for key, value in d.items():
                    if value == 'None':
                        d[key] = None
                    elif isinstance(value, list):
                        d[key] = [None if v == 'None' else v for v in value]
                    elif isinstance(value, dict):
                        replace_none_strings_with_none_type(value)
                return d

            params = replace_none_strings_with_none_type(params)
    return params


@st.cache_data
def load_data(url="https://minio.lab.sspcloud.fr/jbrablx/ai_insurance/raw/train.csv"):
    """
    Load data from a specified URL into a pandas DataFrame.

    Parameters:
    - url (str): The URL pointing to the CSV file to be loaded.

    Returns:
    - DataFrame: A pandas DataFrame containing the data from the CSV file.

    Decorators:
    - @st.cache_data: Caches the output to avoid reloading data from the same URL multiple times.
    """
    df = pd.read_csv(url)
    df.drop(['id'], axis=1, inplace=True)
    return df


def get_features_by_type(df, selected_features):
    """
    Identify and separate numerical and categorical features from a given DataFrame.

    Parameters:
    - df (DataFrame): The pandas DataFrame containing the dataset.
    - selected_features (list of str): List of strings containing the features to be categorized.

    Returns:
    - tuple of lists: A tuple containing two lists:
        - num_features (list): List of names of the numerical features.
        - cat_features (list): List of names of the categorical features.
    """
    num_features = (
        df[selected_features]
        .select_dtypes(include=['int64', 'float64'])
        .columns.tolist()
    )
    cat_features = (
        df[selected_features]
        .select_dtypes(exclude=['int64', 'float64'])
        .columns.tolist()
    )
    return num_features, cat_features
