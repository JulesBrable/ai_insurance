import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as ss

sns.set_style("whitegrid")

def plot_continuous(df, num_features):
    st.markdown("<h1 style='font-family:Lucida Caligraphy;font-size:30px;color:DarkSlateBlue;text-align: center;'> KDE Plot for numerical variables </h1>", unsafe_allow_html=True)
    cols = st.columns([4/5, 1/5], gap="medium")
    
    with cols[1]:
        plot_type = st.radio(
            "Choose the type of plot you want to display:",
            ('Histogram', 'Boxplot')
        )

    n = len(num_features)
    ncols = 3
    nrows = n // ncols + (n % ncols > 0)
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
    if plot_type == 'Histogram':
        fig.suptitle('Distribution of Numerical Features (Histograms)')
    else:
        fig.suptitle('Distribution of Numerical Features (Boxplots)')

    axs = axs.flatten()

    for i, col in enumerate(num_features):
        if plot_type == 'Histogram':
            sns.histplot(df[col], kde=True, ax=axs[i])
        else:
            sns.boxplot(y=df[col], ax=axs[i])
        axs[i].set_title(f'{col}')

    for i in range(n, len(axs)):
        axs[i].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    with cols[0]:
        st.pyplot(fig)

def plot_pie(df: pd.DataFrame, target: str = "Response"):
    st.markdown("<h1 style='font-family:Lucida Caligraphy;font-size:30px;color:DarkSlateBlue;text-align: center;'> Pie Plot for the target variable </h1>", unsafe_allow_html=True)
    fig, ax = plt.subplots()
    ax.pie(df[target].value_counts(), labels=[0, 1], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)
    
def plot_heatmap(df, num_features):
    st.markdown("<h1 style='font-family:Lucida Caligraphy;font-size:30px;color:DarkSlateBlue;text-align: center;'> Correlation between numerical variables </h1>", unsafe_allow_html=True)
    corr_matrix = df[num_features].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)


def plot_categorical_distributions(df, cat_features):
    st.markdown("<h1 style='font-family:Lucida Caligraphy;font-size:30px;color:DarkSlateBlue;text-align: center;'> Barplot for categorical variables </h1>", unsafe_allow_html=True)

    num_rows = 2
    num_cols = [3, 3]

    fig, axes = plt.subplots(num_rows, max(num_cols), figsize=(14, 6 * num_rows))
    axes = axes.flatten()

    for idx, col in enumerate(cat_features):
        proportions = df[col].value_counts(normalize=True)
        ax = proportions.plot(kind='bar', ax=axes[idx], color='skyblue')
        for p in ax.patches:
            ax.annotate(f"{p.get_height() * 100:.2f}%", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', xytext=(0, 10), 
                        textcoords='offset points')

    for i in range(len(cat_features), num_rows * max(num_cols)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    st.pyplot(fig)


def plot_kde_by_response(df):
    st.markdown("<h1 style='font-family:Lucida Caligraphy;font-size:30px;color:DarkSlateBlue;text-align: center;'> KDE Plots by Response </h1>", unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sns.kdeplot(data=df, x="Age", hue="Response", ax=axes[0], palette="coolwarm", common_norm=False)
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Distribution of Age by Response')

    sns.kdeplot(data=df, x="Annual_Premium", hue="Response", ax=axes[1], palette="coolwarm", common_norm=False)
    axes[1].set_xlabel('Annual Premium')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Distribution of Annual Premium by Response')

    sns.kdeplot(data=df, x="Vintage", hue="Response", ax=axes[2], palette="coolwarm", common_norm=False)
    axes[2].set_xlabel('Vintage')
    axes[2].set_ylabel('Density')
    axes[2].set_title('Distribution of Vintage by Response')

    plt.tight_layout()
    st.pyplot(fig)


def cramers_v(x, y):
    """Function to calculate Cramer's V"""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    r_corr = r - ((r-1)**2)/(n-1)
    k_corr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2_corr / min((k_corr-1), (r_corr-1)))


def plot_cramers_v_heatmap(df, cat_variables):
    """Plotting function for Cramer's V heatmap"""
    st.markdown("""<h1 style='font-family:Lucida Caligraphy;font-size:30px;color:DarkSlateBlue;text-align: center;'> 
                Correlation between categorical variables (Cramer's V) </h1>""",
                unsafe_allow_html=True)
    
    cramers_results = pd.DataFrame(index=cat_variables, columns=cat_variables)
    for i in cat_variables:
        for j in cat_variables:
            if i == j:
                cramers_results.loc[i, j] = 1.0
            else:
                cramers_results.loc[i, j] = cramers_v(df[i], df[j])
    cramers_results = cramers_results.astype(float)
    fig, ax = plt.subplots()
    sns.heatmap(cramers_results, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Cramer\'s V'}, ax=ax)
    st.pyplot(fig)
