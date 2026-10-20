"""Exploratory data analysis: draws charts and saves them to reports/figures/."""
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from src import config
from src.data import load_clean

matplotlib.use("Agg")                       # draw to files, no window needed
sns.set_theme(style="whitegrid", palette="Set2")
LABELS = {0: "No disease", 1: "Disease"}


def save(fig, filename):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def class_balance(df):
    """Bar chart: how many patients have / do not have heart disease."""
    counts = df[config.TARGET].map(LABELS).value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index, counts.values, color=["#2a9d8f", "#e76f51"])
    for i, value in enumerate(counts.values):
        ax.text(i, value + 2, f"{value} ({value / counts.sum():.0%})", ha="center")
    ax.set_title("Class balance")
    ax.set_ylabel("Patients")
    return save(fig, "01_class_balance.png")


def histograms(df):
    """Distribution of each numeric feature, split by diagnosis."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, column in zip(axes.ravel(), config.NUMERIC):
        sns.histplot(data=df, x=column, hue=df[config.TARGET].map(LABELS), kde=True, element="step",
                     stat="density", common_norm=False, ax=ax)
        ax.set_title(f"Distribution of {column}")
    return save(fig, "02_histograms.png")


def correlation_heatmap(df):
    """Which features move together with each other and with the target?"""
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title("Correlation heatmap")
    return save(fig, "03_correlation_heatmap.png")


def categorical_vs_target(df):
    """For each category feature: how many healthy / sick patients per value."""
    columns = config.BINARY + config.CATEGORICAL
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, column in zip(axes.ravel(), columns):
        sns.countplot(data=df, x=column, hue=df[config.TARGET].map(LABELS), ax=ax)
        ax.set_title(f"{column} vs target")
        ax.legend(fontsize=7)
    axes.ravel()[-1].axis("off")
    return save(fig, "04_categorical_vs_target.png")


def boxplots(df):
    """Box plots: spread of four key numbers for healthy vs sick patients."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, column in zip(axes, ["age", "thalach", "oldpeak", "chol"]):
        groups = [df.loc[df[config.TARGET] == k, column].dropna() for k in (0, 1)]
        boxes = ax.boxplot(groups, patch_artist=True)
        ax.set_xticks([1, 2], [LABELS[0], LABELS[1]])
        for patch, colour in zip(boxes["boxes"], ["#2a9d8f", "#e76f51"]):
            patch.set_facecolor(colour)
        ax.set_title(f"{column} by diagnosis")
    return save(fig, "05_boxplots.png")


def run_eda():
    """Draw all charts and return their file paths."""
    df = load_clean()
    return [class_balance(df), histograms(df), correlation_heatmap(df),
            categorical_vs_target(df), boxplots(df)]


if __name__ == "__main__":
    for figure_path in run_eda():
        print("saved", figure_path)
