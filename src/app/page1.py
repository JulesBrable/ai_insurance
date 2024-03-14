import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.preprocessing import MinMaxScaler

def plot_kde_after_normalization(df):
    # Copie du DataFrame pour ne pas modifier l'original
    df_normalized = df.copy()

    # Initialisation du StandardScaler
    scaler = MinMaxScaler()

    # Liste des colonnes à normaliser
    columns_to_normalize = ['Age', 'Annual_Premium', 'Vintage']

    # Application de la normalisation pour les colonnes spécifiées et mise à jour du DataFrame
    df_normalized[columns_to_normalize] = scaler.fit_transform(df_normalized[columns_to_normalize])

    # Création des graphiques
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sns.kdeplot(data=df_normalized, x="Age", ax=axes[0], palette="coolwarm", common_norm=False)
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Distribution of Age by Response')

    sns.kdeplot(data=df_normalized, x="Annual_Premium", ax=axes[1], palette="coolwarm", common_norm=False)
    axes[1].set_xlabel('Annual Premium')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Distribution of Annual Premium by Response')

    sns.kdeplot(data=df_normalized, x="Vintage", ax=axes[2], palette="coolwarm", common_norm=False)
    axes[2].set_xlabel('Vintage')
    axes[2].set_ylabel('Density')
    axes[2].set_title('Distribution of Vintage by Response')

    plt.tight_layout()
    return st.pyplot(fig)



lr_data = {
    'Metric': ['Accuracy Score', 'ROC AUC Score', 'Recall Class 1', 'Recall Class 0'],
    'Value': [0.87, 0.85, 0.00, 1.00]
}
lr_df = pd.DataFrame(lr_data)

lr_smote_data = {
    'Metric': ['Accuracy Score', 'ROC AUC Score', 'Recall Class 1', 'Recall Class 0'],
    'Value': [0.75, 0.83, 0.76, 0.75]
}
lr_smote_df = pd.DataFrame(lr_smote_data)

lr_over_data = {
    'Metric': ['Accuracy Score', 'ROC AUC Score', 'Recall Class 1', 'Recall Class 0'],
    'Value': [0.70, 0.85, 0.94, 0.66]
}
lr_over_df = pd.DataFrame(lr_over_data)

rf_data = {
    'Metric': ['Accuracy Score', 'ROC AUC Score', 'Recall Class 1', 'Recall Class 0'],
    'Value': [0.88, 0.86, 0.00, 1.00]
}
rf_df = pd.DataFrame(rf_data)

rf_smote_data = {
    'Metric': ['Accuracy Score', 'ROC AUC Score', 'Recall Class 1', 'Recall Class 0'],
    'Value': [0.079, 0.84, 0.62, 0.81]
}
rf_smote_df = pd.DataFrame(rf_smote_data)

rf_over_data = {
    'Metric': ['Accuracy Score', 'ROC AUC Score', 'Recall Class 1', 'Recall Class 0'],
    'Value': [0.84, 0.83, 0.31, 0.92]
}
rf_over_df = pd.DataFrame(rf_over_data)