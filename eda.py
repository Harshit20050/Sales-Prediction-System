# eda.py
# Exploratory Data Analysis for the Sales Prediction project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "white"})

OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path="advertising.csv"):
    df = pd.read_csv(path)
    return df


def basic_info(df):
    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    print(f"Shape      : {df.shape}")
    print(f"Columns    : {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nDescriptive Statistics:\n{df.describe().round(2)}")


def plot_distributions(df):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold", y=1.02)
    for ax, col in zip(axes, df.columns):
        sns.histplot(df[col], kde=True, ax=ax, color="#3b82f6")
        ax.set_title(col, fontsize=12)
        ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/distributions.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/distributions.png")


def plot_scatter(df):
    features = ["TV", "Radio", "Newspaper"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Advertising Spend vs Sales", fontsize=14, fontweight="bold")
    colors = ["#3b82f6", "#10b981", "#f59e0b"]
    for ax, feat, color in zip(axes, features, colors):
        ax.scatter(df[feat], df["Sales"], alpha=0.6, color=color, edgecolors="white", linewidth=0.5)
        m, b = np.polyfit(df[feat], df["Sales"], 1)
        x_line = np.linspace(df[feat].min(), df[feat].max(), 100)
        ax.plot(x_line, m * x_line + b, color="red", linewidth=1.5, linestyle="--", label="Trend")
        ax.set_xlabel(feat, fontsize=11)
        ax.set_ylabel("Sales", fontsize=11)
        ax.set_title(f"{feat} vs Sales", fontsize=12)
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/scatter_plots.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/scatter_plots.png")


def plot_correlation(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        mask=mask, ax=ax, linewidths=0.5,
        annot_kws={"size": 12}
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/correlation_heatmap.png")


def plot_boxplots(df):
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Outlier Detection (Box Plots)", fontsize=14, fontweight="bold")
    for ax, col in zip(axes, df.columns):
        sns.boxplot(y=df[col], ax=ax, color="#a5b4fc")
        ax.set_title(col, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/boxplots.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/boxplots.png")


if __name__ == "__main__":
    df = load_data()
    basic_info(df)
    plot_distributions(df)
    plot_scatter(df)
    plot_correlation(df)
    plot_boxplots(df)
    print("\nEDA complete. All plots saved to:", OUTPUT_DIR)
