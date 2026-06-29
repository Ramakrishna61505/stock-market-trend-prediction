# 🌌 Trend Analysis AI Terminal

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green)
![Data](https://img.shields.io/badge/Data-yfinance-yellow)

## 📌 Project Overview

The **Trend Analysis AI Terminal** is a professional-grade, multi-asset trading dashboard powered by Machine Learning. Designed with a Bloomberg-terminal aesthetic, the application provides real-time market data, interactive charting, and an AI-driven predictive engine that forecasts short-term market trends across Equities, Cryptocurrencies, and Commodities.

Unlike static models, this architecture dynamically trains a **Random Forest Classifier** on-the-fly using the latest historical data and user-selected technical indicators, ensuring the neural synthesis is strictly tailored to the specific asset's current market conditions.

---

## 🚀 Key Features

- **Global Multi-Asset Support**: Seamlessly fetch and analyze data for major Crypto (BTC, ETH, SOL), Metals (Gold, Silver), and Equities (AAPL, TSLA, SPY).
- **Intraday Terminal**: High-fidelity, real-time interactive `Plotly` candlestick charts with intraday 15-minute intervals, Volume subplots, and Bollinger Bands.
- **Dynamic AI Prediction Engine**: Toggle technical features (RSI, MACD, EMA 9, EMA 15) to dynamically re-train the classification model in the background and generate confidence-weighted predictions.
- **Narrative Synthesis Agent**: An algorithmic parser that translates complex oscillator states and moving average crossovers into human-readable trading intelligence.
- **Liquid Glass Architecture**: A highly responsive, dark-mode CSS implementation utilizing backdrop filters and modular tabs for a premium UX.

---

## 📂 System Architecture & File Structure

```text
stock-market-prediction/
│
├── app/
│   └── app.py                  # Core application entry point and Streamlit frontend
│
├── data/
│   └── tesla_stock_data.csv    # Legacy V1 training dataset (Deprecated in V3)
│
├── main/
│   └── main.ipynb              # Jupyter notebook outlining V1 Data Science pipeline
│
├── models/
│   ├── model.pkl               # Pickled V1 static model (Deprecated in V3)
│   └── scaler.pkl              # Pickled scaler parameters
│
├── requirements.txt            # Environment dependencies
└── README.md                   # System documentation
```

---

## 🛠️ Installation & Setup

To run this application locally, ensure you have Python 3.10+ installed.

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd stock-market-prediction
   ```

2. **Install Dependencies**:
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize the Terminal**:
   Run the Streamlit server using the Python module format:
   ```bash
   python -m streamlit run app/app.py
   ```

4. **Access the Dashboard**:
   Open your browser and navigate to `http://localhost:8501`.

---

## 🧠 Technical Implementation Notes

### Machine Learning Pipeline
The predictive engine utilizes `scikit-learn`'s `RandomForestClassifier`. The target variable is engineered as a binary classification:
- `1` (Bullish): The subsequent trading session's close price is strictly greater than the current session.
- `0` (Bearish): The subsequent trading session's close price is less than or equal to the current session.

The model is retrained continuously whenever the user modifies the `Ticker` or the `Feature Space` configurations, preventing historical drift and optimizing for localized volatility.

### Data Engineering
Real-time pricing and historical datasets are streamed directly via the Yahoo Finance API (`yfinance`). To optimize UI latency, heavy data ingestion functions are aggressively cached utilizing Streamlit's `@st.cache_data` decorators.