import streamlit as st
import joblib
import pandas as pd
import yfinance as yf
import ta

# ✅ Load model
model = joblib.load("xgb_pipeline_model.pkl")
label_map = {0: "Down", 1: "Neutral", 2: "Up"}

# ✅ Custom Styling
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {
            background-color: #ffffff;
            color: #000000;
            border: 1px solid #ccc;
            padding: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Title and About
st.title("📈 Stock Movement Predictor")
with st.expander("📘 About this model"):
    st.markdown("""
    This app uses a trained **XGBoost Classifier** to predict stock return movement based on technical indicators such as:

    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Band Mean
    - EMA (Exponential Moving Average)
    - Daily Return
    - And price/volume metrics

    The model classifies returns into:
    - **0 = Down**
    - **1 = Neutral**
    - **2 = Up**
    """)

# ✅ Get stock ticker
ticker = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT, TSLA)", "AAPL")

# ✅ Fetch data and calculate features
def get_latest_technical_indicators(ticker):
    df = yf.download(ticker, period="3mo")
    df = df.rename(columns={
        "Open": "Open", "High": "High", "Low": "Low",
        "Close": "Close", "Volume": "Volume"
    })
    df = ta.add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
    ).copy()
    df = df.apply(lambda col: col.squeeze() if hasattr(col, "squeeze") else col)
    df["Daily_Return"] = df["Close"].pct_change().fillna(0)
    return df.iloc[-1]  # Last row with latest indicators

if st.button("Predict"):
    try:
        stock_data = get_latest_technical_indicators(ticker)
        features = [
            stock_data["Open"], stock_data["High"], stock_data["Low"], stock_data["Close"],
            stock_data["Volume"], stock_data["momentum_rsi"], stock_data["trend_macd"],
            stock_data["volatility_bbm"], stock_data["trend_ema_fast"], stock_data["Daily_Return"]
        ]

        prediction = model.predict([features])[0]
        confidence = max(model.predict_proba([features])[0]) * 100

        if prediction == 0:
            st.error(f"📉 Predicted Return: Down ({confidence:.2f}%)")
        elif prediction == 1:
            st.warning(f"➖ Predicted Return: Neutral ({confidence:.2f}%)")
        elif prediction == 2:
            st.success(f"📈 Predicted Return: Up ({confidence:.2f}%)")

    except Exception as e:
        st.error(f"Error fetching data: {e}")
