# 📈 Sales Prediction System

> Predict product sales from advertising spend using Linear Regression · Deployed on Streamlit

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.9+-3670A0?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Overview

This end-to-end ML project forecasts product **Sales** based on advertising budgets
across **TV**, **Radio**, and **Newspaper** channels.

| Metric | Score |
|--------|-------|
| R² Score | **0.9628** |
| RMSE | **1.0039** |
| MAE | **0.8593** |
| Algorithm | Linear Regression |

---

## 🗂️ Project Structure

```
Sales-Prediction-System/
│
├── generate_dataset.py   # Step 1 — Generate/load advertising dataset
├── eda.py                # Step 2 — Exploratory Data Analysis + plots
├── train_model.py        # Step 3 — Train, evaluate & save model
├── app.py                # Step 4 — Streamlit web application
│
├── advertising.csv       # Dataset (auto-generated if missing)
├── requirements.txt      # Python dependencies
├── .streamlit/
│   └── config.toml       # Streamlit theme config
│
├── model/
│   ├── model.pkl         # Saved Linear Regression model
│   └── scaler.pkl        # Saved StandardScaler
│
└── eda_outputs/          # EDA and evaluation plots
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

---

## 🌐 Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo
4. Set **Main file**: `app.py`
5. Click **Deploy** ✅

---

## 📊 Features

- 🔮 **Predict** — real-time sales prediction with sliders
- 📊 **Data Explorer** — scatter plots, correlation heatmap, distributions
- 📉 **Model Performance** — actual vs predicted, residual analysis
- ℹ️ **About** — project details and tech stack

---

## 🛠️ Tech Stack

`Python` · `Scikit-learn` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Streamlit`

---

## 👨‍💻 Author

**Harshit Sharma** · BCA Final Year · JECRC University, Jaipur

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/harshitsharma56)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/Harshit20050)
