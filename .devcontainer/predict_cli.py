import streamlit as st
import joblib
import pandas as pd
import yfinance as yf
import ta

# 🎨 Custom Styling
st.markdown("""
    <style>
        .main {
            background-color: #f9f9fc;
        }
        h1 {
            color: #2c3e50;
        }
        .stTextInput>div>div>input {
            background-color: #ffffff;
            border: 1px solid #ddd;
            padding: 8px;
        }
        .stock-symbol {
            font-size: 20px;
            color: #2c3e50;
            background-color: #e6ecf0;
            padding: 6px 12px;
            border-radius: 5px;
            font-weight: bold;
            display: inline-block;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# 🔄 Load model
model = joblib.load("xgb_pipeline_model.pkl")
label_map = {0: "Down", 1: "Neutral", 2: "Up"}

st.title("📊 Stock Movement Predictor")
st.caption("🚀 Real-time stock return prediction using technical indicators and XGBoost")

# ℹ️ Description
with st.expander("📘 What This App Does"):
    st.markdown("""
    This tool predicts the **next-day return direction** of a stock using:

    - RSI, MACD, Bollinger Bands, EMA
    - Daily Return & Volume data
    - Cleaned using TA-Lib, modeled with XGBoost

    **Outputs:**
    - `0` = 📉 Down  
    - `1` = ➖ Neutral  
    - `2` = 📈 Up  
    """)
import numpy as np
import pandas as pd
import yfinance as yf
import ta
import streamlit as st

def get_latest_technical_indicators(symbol):
    data = yf.download(symbol, period="3mo", interval="1d", auto_adjust=True)
    if data.empty or len(data) < 20:
        return None

    # ✅ Drop multi-index if exists
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(0)

    # ✅ Rename columns based on position
    col_count = len(data.columns)
    if col_count >= 6:
        data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    elif col_count == 5:
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    else:
        st.error(f"Unexpected column count: {col_count}. Can't map properly.")
        return None

    # Flatten, convert, and sanitize
    def force_flat(col):
        values = col.values
        if values.ndim == 2:
            values = values.flatten()
        return pd.Series(values, index=col.index)

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        data[col] = force_flat(data[col])
        data[col] = data[col].apply(lambda x: x[0] if isinstance(x, (np.ndarray, list)) else x)
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data['daily_return'] = data['Close'].pct_change()

    try:
        data = ta.add_all_ta_features(
            data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
        )
    except Exception as e:
        st.error(f"❌ Error computing technical indicators: {e}")
        return None

    latest = data.iloc[-1]
    return {
        "Open": latest["Open"],
        "High": latest["High"],
        "Low": latest["Low"],
        "Close": latest["Close"],
        "Volume": latest["Volume"],
        "RSI": latest.get("momentum_rsi", 0),
        "MACD": latest.get("trend_macd", 0),
        "Bollinger_Band_Mean": latest.get("volatility_bbm", 0),
        "EMA_Fast": latest.get("trend_ema_fast", 0),
        "Daily_Return": latest["daily_return"]
    }



# 🔍 App UI
ticker = st.text_input("🔍 Enter Stock Symbol (e.g., AAPL)", value="AAPL")

if ticker:
    with st.spinner("📡 Fetching and computing indicators..."):
        stock_data = get_latest_technical_indicators(ticker)

    if stock_data:
        st.markdown(f'<div class="stock-symbol">{ticker.upper()}</div>', unsafe_allow_html=True)
        st.success("📌 Indicators auto-filled. You may adjust before prediction.")

        open_price = st.number_input("Open Price", value=stock_data["Open"])
        high_price = st.number_input("High Price", value=stock_data["High"])
        low_price = st.number_input("Low Price", value=stock_data["Low"])
        close_price = st.number_input("Close Price", value=stock_data["Close"])
        volume = st.number_input("Volume", value=stock_data["Volume"])
        rsi = st.number_input("RSI", value=stock_data["RSI"])
        macd = st.number_input("MACD", value=stock_data["MACD"])
        bbm = st.number_input("Bollinger Band Mean", value=stock_data["Bollinger_Band_Mean"])
        ema_fast = st.number_input("EMA Fast", value=stock_data["EMA_Fast"])
        daily_return = st.number_input("Daily Return", value=stock_data["Daily_Return"])

        if st.button("🔮 Predict Next-Day Movement"):
            features = [[open_price, high_price, low_price, close_price, volume, rsi, macd, bbm, ema_fast, daily_return]]
            prediction = model.predict(features)[0]
            confidence = max(model.predict_proba(features)[0]) * 100

            if prediction == 0:
                st.error(f"📉 Predicted Return: Down ({confidence:.2f}% confidence)")
            elif prediction == 1:
                st.warning(f"➖ Predicted Return: Neutral ({confidence:.2f}% confidence)")
            elif prediction == 2:
                st.success(f"📈 Predicted Return: Up ({confidence:.2f}% confidence)")
    else:
        st.warning("⚠️ Not enough recent data to generate indicators. Try another ticker.")
