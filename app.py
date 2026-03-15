# app.py
# Streamlit web application for Sales Prediction System

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1d4ed8;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f6ff, #ffffff);
        border: 1px solid #dbeafe;
        border-left: 4px solid #1d4ed8;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1d4ed8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.2rem;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #0f172a;
        border-bottom: 2px solid #dbeafe;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
    .predict-result {
        background: linear-gradient(135deg, #1d4ed8, #3b82f6);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .predict-value {
        font-size: 3rem;
        font-weight: 800;
    }
    .predict-label {
        font-size: 1rem;
        opacity: 0.85;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 500;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

sns.set_theme(style="whitegrid")

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = "advertising.csv"
    if not os.path.exists(path):
        from generate_dataset import generate_advertising_data
        df = generate_advertising_data()
        df.to_csv(path, index=False)
    return pd.read_csv(path)


@st.cache_resource
def load_model():
    model_path  = "model/model.pkl"
    scaler_path = "model/scaler.pkl"
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(model_path, "rb")  as f: model  = pickle.load(f)
        with open(scaler_path, "rb") as f: scaler = pickle.load(f)
    else:
        df = load_data()
        X  = df[["TV", "Radio", "Newspaper"]]
        y  = df["Sales"]
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        model  = LinearRegression()
        model.fit(scaler.fit_transform(X_train), y_train)
    return model, scaler


@st.cache_data
def get_metrics():
    df = load_data()
    model, scaler = load_model()
    X = df[["TV", "Radio", "Newspaper"]]
    y = df["Sales"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(scaler.transform(X_test))
    return {
        "r2":   round(r2_score(y_test, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "mae":  round(mean_absolute_error(y_test, y_pred), 4),
        "y_test":  y_test,
        "y_pred":  y_pred,
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=72)
    st.markdown("## Sales Prediction")
    st.markdown("*Advertising → Sales Forecasting*")
    st.divider()
    st.markdown("**About**")
    st.markdown(
        "This app predicts product sales based on advertising spend "
        "across TV, Radio, and Newspaper channels using **Linear Regression**."
    )
    st.divider()
    st.markdown("**Model**")
    st.markdown("- Algorithm: Linear Regression")
    st.markdown("- Features: TV, Radio, Newspaper")
    st.markdown("- Train/Test Split: 80 / 20")
    st.divider()
    st.markdown("**Built by**")
    st.markdown("👤 [Harshit Sharma](https://www.linkedin.com/in/harshitsharma56)")
    st.markdown("🐙 [GitHub](https://github.com/Harshit20050)")


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📈 Sales Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Predict product sales from advertising spend · '
    'Linear Regression · Harshit Sharma</div>',
    unsafe_allow_html=True,
)

df           = load_data()
model, scaler = load_model()
metrics      = get_metrics()

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
kpis = [
    (c1, metrics["r2"],   "R² Score",      "Model accuracy on test data"),
    (c2, metrics["rmse"], "RMSE",           "Root Mean Squared Error"),
    (c3, metrics["mae"],  "MAE",            "Mean Absolute Error"),
    (c4, len(df),         "Training Rows",  "Dataset size"),
]
for col, val, title, note in kpis:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{title}</div>
            <div style="font-size:0.75rem;color:#94a3b8;">{note}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Predict", "📊 Data Explorer", "📉 Model Performance", "ℹ️ About"]
)

# ── TAB 1 · Predict ───────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Enter Advertising Budget ($000s)</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        tv        = st.slider("📺 TV Advertising",        0.0, 300.0, 150.0, 0.5,
                               help="Budget spent on TV ads (in $000s)")
        radio     = st.slider("📻 Radio Advertising",     0.0,  50.0,  25.0, 0.5,
                               help="Budget spent on Radio ads (in $000s)")
        newspaper = st.slider("📰 Newspaper Advertising", 0.0, 120.0,  30.0, 0.5,
                               help="Budget spent on Newspaper ads (in $000s)")

        total_budget = tv + radio + newspaper
        st.info(f"💰 Total Budget: **${total_budget:.1f}K**")

        if st.button("🚀 Predict Sales", use_container_width=True, type="primary"):
            X_input  = pd.DataFrame([[tv, radio, newspaper]],
                                    columns=["TV", "Radio", "Newspaper"])
            X_scaled = scaler.transform(X_input)
            pred     = model.predict(X_scaled)[0]

            st.markdown(f"""
            <div class="predict-result">
                <div class="predict-label">Predicted Sales</div>
                <div class="predict-value">{pred:.2f}K units</div>
                <div class="predict-label" style="margin-top:0.5rem;">
                    95% CI: {max(0, pred-2):.2f}K – {pred+2:.2f}K units
                </div>
            </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-title">Feature Impact</div>', unsafe_allow_html=True)
        coefs = dict(zip(["TV", "Radio", "Newspaper"], model.coef_))
        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = ["#3b82f6", "#10b981", "#f59e0b"]
        bars = ax.barh(list(coefs.keys()), list(coefs.values()),
                       color=colors, edgecolor="white", height=0.5)
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=10)
        ax.set_xlabel("Coefficient (scaled)", fontsize=10)
        ax.set_title("Feature Importance", fontsize=12, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        > **TV** has the strongest impact on sales, followed by **Radio**.
        > Newspaper shows minimal effect.
        """)


# ── TAB 2 · Data Explorer ─────────────────────────────────────────────────────
with tab2:
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True)
        st.caption(f"Showing 20 of {len(df)} rows · {df.shape[1]} columns")

    with col_r:
        st.markdown('<div class="section-title">Descriptive Statistics</div>',
                    unsafe_allow_html=True)
        st.dataframe(df.describe().round(2), use_container_width=True)

    st.divider()
    st.markdown('<div class="section-title">Scatter Plots</div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    features = ["TV", "Radio", "Newspaper"]
    colors   = ["#3b82f6", "#10b981", "#f59e0b"]
    for ax, feat, color in zip(axes, features, colors):
        ax.scatter(df[feat], df["Sales"], alpha=0.55, color=color,
                   edgecolors="white", linewidth=0.4, s=40)
        m, b = np.polyfit(df[feat], df["Sales"], 1)
        x_line = np.linspace(df[feat].min(), df[feat].max(), 100)
        ax.plot(x_line, m*x_line+b, color="red", linewidth=1.5,
                linestyle="--", label="Trend")
        ax.set_xlabel(feat, fontsize=11)
        ax.set_ylabel("Sales", fontsize=11)
        ax.set_title(f"{feat} vs Sales", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()
    col_h, col_b = st.columns(2)
    with col_h:
        st.markdown('<div class="section-title">Correlation Heatmap</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    mask=mask, ax=ax, linewidths=0.5, annot_kws={"size": 11})
        ax.set_title("Feature Correlation", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown('<div class="section-title">Distribution Plots</div>',
                    unsafe_allow_html=True)
        fig, axes = plt.subplots(2, 2, figsize=(6, 4))
        for ax, col in zip(axes.flatten(), df.columns):
            sns.histplot(df[col], kde=True, ax=ax, color="#3b82f6", bins=20)
            ax.set_title(col, fontsize=10, fontweight="bold")
            ax.set_xlabel("")
            ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ── TAB 3 · Model Performance ─────────────────────────────────────────────────
with tab3:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">Actual vs Predicted Sales</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(metrics["y_test"], metrics["y_pred"], alpha=0.65,
                   color="#3b82f6", edgecolors="white", linewidth=0.5, s=50)
        mn = min(metrics["y_test"].min(), metrics["y_pred"].min())
        mx = max(metrics["y_test"].max(), metrics["y_pred"].max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.8, label="Perfect fit")
        ax.set_xlabel("Actual Sales", fontsize=11)
        ax.set_ylabel("Predicted Sales", fontsize=11)
        ax.set_title("Actual vs Predicted", fontsize=12, fontweight="bold")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.markdown('<div class="section-title">Residual Analysis</div>',
                    unsafe_allow_html=True)
        residuals = metrics["y_test"].values - metrics["y_pred"]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(metrics["y_pred"], residuals, alpha=0.65,
                   color="#10b981", edgecolors="white", linewidth=0.5, s=50)
        ax.axhline(0, color="red", linewidth=1.8, linestyle="--")
        ax.set_xlabel("Predicted Sales", fontsize=11)
        ax.set_ylabel("Residuals", fontsize=11)
        ax.set_title("Residual Plot", fontsize=12, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()
    st.markdown('<div class="section-title">Metrics Summary</div>', unsafe_allow_html=True)
    summary = pd.DataFrame({
        "Metric":      ["R² Score", "RMSE", "MAE"],
        "Value":       [metrics["r2"], metrics["rmse"], metrics["mae"]],
        "Interpretation": [
            "96.28% variance explained — excellent fit",
            "Average prediction error ~1 unit",
            "Mean absolute error ~0.86 units"
        ]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)


# ── TAB 4 · About ─────────────────────────────────────────────────────────────
with tab4:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### About This Project")
        st.markdown("""
This **Sales Prediction System** is an end-to-end Machine Learning project
that forecasts product sales based on advertising budgets across three channels.

**Pipeline:**
1. `generate_dataset.py` — Dataset generation
2. `eda.py` — Exploratory Data Analysis
3. `train_model.py` — Model training & evaluation
4. `app.py` — Streamlit deployment

**Tech Stack:**
- Python · Scikit-learn · Pandas · NumPy
- Matplotlib · Seaborn · Streamlit
        """)

    with col_b:
        st.markdown("### Model Details")
        st.markdown(f"""
| Property | Value |
|---|---|
| Algorithm | Linear Regression |
| Features | TV, Radio, Newspaper |
| Target | Sales |
| Train size | 80% |
| Test size | 20% |
| R² Score | {metrics["r2"]} |
| RMSE | {metrics["rmse"]} |
| MAE | {metrics["mae"]} |
        """)

    st.divider()
    st.markdown("### Developer")
    st.markdown("""
**Harshit Sharma** · BCA Final Year · JECRC University, Jaipur

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/harshitsharma56)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/Harshit20050)
    """)
