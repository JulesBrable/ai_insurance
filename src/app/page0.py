import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def plot_continuous(df, num_features):
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


def plot_heatmap(df, num_features):
    st.write("Correlation Matrix")
    corr_matrix = df[num_features].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)


def plot_numerical_features(df, num_features):
    plot_continuous(df, num_features)
    
    plot_heatmap(df, num_features)