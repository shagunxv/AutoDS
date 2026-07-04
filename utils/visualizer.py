import matplotlib 
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import os


os.makedirs(
    "static/plots",
    exist_ok=True
)
def generate_heatmap(df):

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm"
    )

    plt.title(
        "Correlation Heatmap"
    )

    plt.tight_layout()

    plt.savefig(
        "static/plots/heatmap.png"
    )

    plt.close()
def generate_missing_plot(df):

    missing = df.isnull().sum()

    plt.figure(figsize=(10, 5))

    missing.plot(
        kind="bar"
    )

    plt.title(
        "Missing Values"
    )

    plt.ylabel(
        "Count"
    )

    plt.tight_layout()

    plt.savefig(
        "static/plots/missing.png"
    )

    plt.close()
def generate_target_distribution(
        df,
        target):

    plt.figure(figsize=(8, 5))

    if df[target].dtype == "object":

        sns.countplot(
            x=df[target]
        )

    else:

        sns.histplot(
            df[target],
            kde=True
        )

    plt.title(
        f"{target} Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        "static/plots/target.png"
    )

    plt.close()
