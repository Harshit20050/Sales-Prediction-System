# 📈 Sales Prediction System

> Predict product sales from advertising spend using Linear Regression · Deployed on Streamlit

<p align="center">
  <a href="https://sales-prediction-system-gphxzacs8cwsd5fiik3wot.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit" height="35"/>
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3670A0?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge"/>
</p>

---

## 🎯 Overview

This end-to-end Machine Learning project forecasts product **Sales** based on advertising budgets
across **TV**, **Radio**, and **Newspaper** channels using **Linear Regression**.

| Metric | Score | Interpretation |
|--------|-------|----------------|
| R² Score | **0.9628** | 96.28% variance explained |
| RMSE | **1.0039** | Avg prediction error ~1 unit |
| MAE | **0.8593** | Mean absolute error ~0.86 units |
| Algorithm | **Linear Regression** | Scikit-learn |

---

## 🌐 Live Demo

👉 **[Click here to open the live app](https://sales-prediction-system-gphxzacs8cwsd5fiik3wot.streamlit.app/)**

<a href="https://sales-prediction-system-gphxzacs8cwsd5fiik3wot.streamlit.app/">
  <img src="https://img.shields.io/badge/🚀 Open Live Demo-1d4ed8?style=for-the-badge&logoColor=white" />
</a>

**App Features:**
- 🔮 **Predict** — real-time sales prediction using interactive sliders
- 📊 **Data Explorer** — scatter plots, correlation heatmap, distributions
- 📉 **Model Performance** — actual vs predicted, residual analysis
- ℹ️ **About** — project pipeline and tech stack

---

## 🗂️ Project Structure

```
Sales-Prediction-System/
│
├── generate_dataset.py     # Step 1 — Generate advertising dataset
├── eda.py                  # Step 2 — Exploratory Data Analysis + plots
├── train_model.py          # Step 3 — Train, evaluate & save model
├── app.py                  # Step 4 — Streamlit web application
│
├── advertising.csv         # Dataset (auto-generated if missing)
├── requirements.txt        # Python dependencies
│
├── .streamlit/
│   └── config.toml         # Streamlit theme config
│
├── model/
│   ├── model.pkl           # Saved Linear Regression model
│   └── scaler.pkl          # Saved StandardScaler
│
└── eda_outputs/            # EDA and evaluation plots
    ├── distributions.png
    ├── scatter_plots.png
    ├── correlation_heatmap.png
    ├── boxplots.png
    └── model_evaluation.png
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Harshit20050/Sales-Prediction-System
cd Sales-Prediction-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset + train model
python generate_dataset.py
python eda.py
python train_model.py

# 4. Launch the app
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 🌐 Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to **[share.streamlit.io](https://share.streamlit.io)**
3. Click **New app** → select your repo
4. Set **Main file**: `app.py`
5. Click **Deploy** ✅

---

## 🛠️ Tech Stack

<p>
  <img src="https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"/>
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-ffffff?style=for-the-badge&logo=Matplotlib&logoColor=black"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
</p>

---

## 👨‍💻 Author

**Harshit Sharma** · BCA Final Year · JECRC University, Jaipur

<p>
  <a href="https://www.linkedin.com/in/harshitsharma56">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white"/>
  </a>
  <a href="https://github.com/Harshit20050">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/>
  </a>
  <a href="https://sales-prediction-system-gphxzacs8cwsd5fiik3wot.streamlit.app/">
    <img src="https://img.shields.io/badge/Live Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  </a>
</p>