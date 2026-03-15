# train_model.py
# Trains, evaluates, and saves the Linear Regression model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "white"})

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(path="advertising.csv"):
    return pd.read_csv(path)


def train_and_evaluate(df):
    X = df[["TV", "Radio", "Newspaper"]]
    y = df["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)

    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring="r2")

    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print(f"R² Score        : {r2:.4f}")
    print(f"RMSE            : {rmse:.4f}")
    print(f"MAE             : {mae:.4f}")
    print(f"CV R² (5-fold)  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\nFeature Coefficients:")
    for feat, coef in zip(X.columns, model.coef_):
        print(f"  {feat:12s}: {coef:.4f}")
    print(f"  {'Intercept':12s}: {model.intercept_:.4f}")

    return model, scaler, X_test, y_test, y_pred, r2, rmse, mae


def plot_results(y_test, y_pred):
    os.makedirs("eda_outputs", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Evaluation", fontsize=14, fontweight="bold")

    # Actual vs Predicted
    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.65, color="#3b82f6", edgecolors="white", linewidth=0.5)
    mn, mx = y_test.min(), y_test.max()
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Sales", fontsize=11)
    ax.set_ylabel("Predicted Sales", fontsize=11)
    ax.set_title("Actual vs Predicted Sales", fontsize=12)
    ax.legend()

    # Residuals
    ax = axes[1]
    residuals = y_test.values - y_pred
    ax.scatter(y_pred, residuals, alpha=0.65, color="#10b981", edgecolors="white", linewidth=0.5)
    ax.axhline(0, color="red", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Predicted Sales", fontsize=11)
    ax.set_ylabel("Residuals", fontsize=11)
    ax.set_title("Residual Plot", fontsize=12)

    plt.tight_layout()
    plt.savefig("eda_outputs/model_evaluation.png", bbox_inches="tight")
    plt.close()
    print("Saved: eda_outputs/model_evaluation.png")


def save_artifacts(model, scaler):
    with open(f"{MODEL_DIR}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"\nModel saved  : {MODEL_DIR}/model.pkl")
    print(f"Scaler saved : {MODEL_DIR}/scaler.pkl")


if __name__ == "__main__":
    df = load_data()
    model, scaler, X_test, y_test, y_pred, r2, rmse, mae = train_and_evaluate(df)
    plot_results(y_test, y_pred)
    save_artifacts(model, scaler)
    print("\nTraining complete!")
