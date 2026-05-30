import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Stock Market Trend Prediction",
    page_icon="📈",
    layout="wide"
)

# LOAD MODEL

try:
    rf_model = joblib.load('models/model.pkl')

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# CUSTOM CSS

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3, h4 {
    color: white;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
    background-color: #4CAF50;
    color: white;
}

.metric-card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# TITLE

st.title("📈 Stock Market Trend Signal Prediction")

st.markdown("""
This Machine Learning application predicts whether the stock market trend is:

- 📈 **Bullish Trend**
- 📉 **Bearish Trend**

using technical indicators and Machine Learning models.
""")

# SIDEBAR INPUTS

st.sidebar.header("📊 Input Stock Features")

open_price = st.sidebar.number_input(
    "Open Price",
    min_value=0.0,
    value=200.0
)

high_price = st.sidebar.number_input(
    "High Price",
    min_value=0.0,
    value=210.0
)

low_price = st.sidebar.number_input(
    "Low Price",
    min_value=0.0,
    value=195.0
)

close_price = st.sidebar.number_input(
    "Close Price",
    min_value=0.0,
    value=205.0
)

volume = st.sidebar.number_input(
    "Volume",
    min_value=0.0,
    value=1000000.0
)

daily_return = st.sidebar.number_input(
    "Daily Return",
    value=0.01,
    format="%.4f"
)

volatility = st.sidebar.number_input(
    "Volatility",
    value=0.02,
    format="%.4f"
)

rsi = st.sidebar.slider(
    "RSI",
    min_value=0,
    max_value=100,
    value=50
)

ema_10 = st.sidebar.number_input(
    "EMA 10",
    min_value=0.0,
    value=200.0
)

ema_30 = st.sidebar.number_input(
    "EMA 30",
    min_value=0.0,
    value=198.0
)

macd = st.sidebar.number_input(
    "MACD",
    value=2.0,
    format="%.4f"
)

# CREATE INPUT DATAFRAME

input_data = pd.DataFrame({
    'Open': [open_price],
    'High': [high_price],
    'Low': [low_price],
    'Close': [close_price],
    'Volume': [volume],
    'Daily_Return': [daily_return],
    'Volatility': [volatility],
    'RSI': [rsi],
    'EMA_10': [ema_10],
    'EMA_30': [ema_30],
    'MACD': [macd]
})

# PREDICTION BUTTON

if st.button("🚀 Predict Market Trend"):

    try:

        prediction = rf_model.predict(input_data)[0]

        # PREDICTION RESULT

        st.subheader("📌 Prediction Result")

        if prediction == 1:

            st.success("📈 Bullish Trend Predicted")

            st.markdown("""
            <div class="metric-card">
                <h2>📈 Market Sentiment</h2>
                <h1>BULLISH</h1>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.error("📉 Bearish Trend Predicted")

            st.markdown("""
            <div class="metric-card">
                <h2>📉 Market Sentiment</h2>
                <h1>BEARISH</h1>
            </div>
            """, unsafe_allow_html=True)

        # INPUT DATA DISPLAY

        st.subheader("📋 Input Feature Values")

        st.dataframe(input_data)

        # FEATURE IMPORTANCE GRAPH

        st.subheader("📊 Feature Importance")

        feature_names = [
            'Open',
            'High',
            'Low',
            'Close',
            'Volume',
            'Daily_Return',
            'Volatility',
            'RSI',
            'EMA_10',
            'EMA_30',
            'MACD'
        ]

        importance_values = rf_model.feature_importances_

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.barh(feature_names, importance_values)

        ax.set_xlabel("Importance")
        ax.set_ylabel("Features")
        ax.set_title("Random Forest Feature Importance")

        ax.invert_yaxis()

        st.pyplot(fig)

        # RSI VISUALIZATION

        st.subheader("📈 RSI Indicator")

        fig2, ax2 = plt.subplots(figsize=(10, 2))

        ax2.barh(['RSI'], [rsi])

        ax2.set_xlim(0, 100)

        ax2.set_title("Relative Strength Index")

        st.pyplot(fig2)

        # MACD VISUALIZATION

        st.subheader("📉 MACD Indicator")

        fig3, ax3 = plt.subplots(figsize=(10, 2))

        ax3.barh(['MACD'], [macd])

        ax3.set_title("MACD Momentum")

        st.pyplot(fig3)

        # MARKET INSIGHTS

        st.subheader("🧠 Market Insights")

        # RSI Insights

        if rsi > 70:
            st.warning("RSI indicates the stock may be overbought.")

        elif rsi < 30:
            st.warning("RSI indicates the stock may be oversold.")

        else:
            st.info("RSI indicates neutral market conditions.")

        # MACD Insights

        if macd > 0:
            st.success("MACD indicates bullish momentum.")

        else:
            st.error("MACD indicates bearish momentum.")

        # EMA Insights

        if ema_10 > ema_30:
            st.success("Short-term EMA is above long-term EMA indicating bullish trend.")

        else:
            st.warning("Short-term EMA is below long-term EMA indicating bearish trend.")

    except Exception as e:

        st.error(f"Prediction Error: {e}")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")