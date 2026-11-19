
import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


PLOT_FOLDER = "static/plots"
os.makedirs(PLOT_FOLDER, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")


def generate_actual_vs_predicted(y_test, predictions):

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(y_test, predictions, alpha=0.7)

    minimum = min(y_test.min(), predictions.min())
    maximum = max(y_test.max(), predictions.max())

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    ax.set_title("Actual vs Predicted")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")

    fig.tight_layout()
    fig.savefig("static/plots/actual_vs_predicted.png", dpi=150)
    plt.close(fig)


def generate_heatmap(df):

    numeric_df = df.select_dtypes(include=["number"])

    fig, ax = plt.subplots(figsize=(11, 7))

    if numeric_df.shape[1] < 2:
        ax.text(
            0.5, 0.5,
            "Not enough numerical columns for correlation analysis",
            ha="center", va="center",
            fontsize=12
        )
        ax.axis("off")

    else:
        correlation = numeric_df.corr()

        sns.heatmap(
            correlation,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            square=True,
            cbar_kws={"label": "Correlation"},
            ax=ax
        )

        ax.set_title(
            "Feature Correlation Heatmap",
            fontsize=15,
            fontweight="bold",
            pad=15
        )

        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    fig.savefig(
        f"{PLOT_FOLDER}/heatmap.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.close(fig)



def generate_missing_plot(df):

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    if missing.empty:
        ax.text(
            0.5, 0.5,
            "No missing values found!",
            ha="center", va="center",
            fontsize=15, fontweight="bold",
            color="green"
        )
        ax.axis("off")

    else:
        percentages = (missing / len(df)) * 100

        bars = ax.barh(
            missing.index,
            missing.values,
            color="#4285F4"
        )

        for bar, count, percentage in zip(
            bars, missing.values, percentages.values
        ):
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f"  {int(count)} ({percentage:.1f}%)",
                va="center",
                fontsize=9
            )

        ax.set_xlabel("Number of Missing Values")
        ax.set_ylabel("Features")
        ax.set_title("Missing Values by Feature")
        ax.set_xlim(0, max(missing.values) * 1.35)

    fig.tight_layout()
    fig.savefig(f"{PLOT_FOLDER}/missing.png", dpi=150)
    plt.close(fig)


def generate_target_distribution(df, target):

    series = df[target].dropna()

    fig, ax = plt.subplots(figsize=(9, 5))

    if series.nunique() <= 20:

        counts = series.value_counts().sort_index()
        labels = [str(value).removesuffix(".0")
                  if isinstance(value, float) and value.is_integer()
                  else str(value) for value in counts.index]

        bars = ax.bar(
            labels,
            counts.values,
            color=sns.color_palette("viridis", len(counts))
        )

        ax.bar_label(bars, padding=4)

        ax.set_xlabel("Class")
        ax.set_ylabel("Number of Records")

    else:

        sns.histplot(
            series,
            bins=30,
            kde=True,
            color="#4285F4",
            ax=ax
        )

        ax.set_xlabel(target)
        ax.set_ylabel("Frequency")

    ax.set_title(f"Distribution of {target}")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(f"{PLOT_FOLDER}/target.png", dpi=150)
    plt.close(fig)
