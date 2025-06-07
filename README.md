# 📈 Stock Price Analysis, Forecasating & Prediction Dashboard (End-to-End Project)

This project is an end-to-end data analytics and machine learning pipeline focused on understanding and forecasting stock price movements using technical indicators and real-time data. It combines data engineering, business intelligence, and predictive modeling in one integrated workflow.

## 🔧 Project Scope
- **Objective**: Predict short-term stock return movements (Up, Neutral, Down) using technical indicators and visualize key performance metrics across companies and time.
- **Tools Used**: Python (Pandas, NumPy, Scikit-learn, XGBoost, MatlabPlot, Seabron), SQL (SQLite), Excel, Tableau, Streamlit

## 📊 Data Processing & Exploration
- Cleaned and merged 5 years of stock price data across multiple tickers.
- Conducted exploratory data analysis (EDA) in **Excel** and **Python**, analyzing trends in volume, volatility, and price movement.
- Built pivot tables and used Excel formulas for summary statistics, volatility levels, and return distributions.
- Loaded cleaned data into **SQLite** and used **SQL** to run complex queries for slicing time periods and comparing company performance.

## 📈 Visual Analytics
- Built **interactive dashboards in Tableau** showing:
  - Price trends, volume spikes, and moving averages.
  - High vs. low volatility stocks.
  - Monthly return patterns and risk/return ratios.
- Used **Tableau** and python visualization libraries to visualize company-wise volatility and compare KPIs.

## 🧠 Predictive Modeling
- Engineered features including RSI, MACD, Bollinger Bands, EMA, and daily returns using the `ta` library.
- Trained an **XGBoost classifier** with `GridSearchCV` for tuning.
- Evaluated model with classification report and confusion matrix on test data.
- Serialized model with `joblib` for deployment.

## 🚀 Deployment & Automation
- Developed two interfaces for predictions:
  - **Command-Line Interface (CLI)**: Lightweight terminal tool (`predict_cli.py`).
  - **Web App**: Interactive Streamlit app (`app.py`) that:
    - Fetches real-time data from `yfinance`.
    - Auto-computes technical indicators.
    - Displays prediction with confidence score and visual feedback.
- Streamlit app designed with custom styling and support for both manual and automatic input modes.

## 🎯 Business Impact
- Enables users (investors, analysts, students) to interactively explore historical trends and forecast stock movement.
- Demonstrates the full data-to-deployment workflow for real-world financial decision-making.

---

Feel free to clone this repository, run the scripts, and interact with the app locally or deploy it on Streamlit Cloud!
