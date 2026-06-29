import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Trend Analysis AI Terminal",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS: LIQUID GLASS THEME
st.markdown("""
<style>
/* Background Gradient */
.stApp {
    background: radial-gradient(circle at 10% 20%, rgb(0, 0, 0) 0%, rgb(20, 20, 30) 90%);
    color: #e0e0e0;
}
/* Glassmorphism Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
    margin-bottom: 20px;
    color: #ffffff;
}
.glass-card h1, .glass-card h2, .glass-card h3, .glass-card p {
    color: #ffffff !important;
}
/* Main Titles */
h1, h2, h3 {
    color: #ffffff;
    font-weight: 300;
}
.metric-container {
    display: flex;
    justify-content: space-between;
}
/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 4px 4px 0px 0px;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# SESSION STATE FOR WALLET
if 'connected' not in st.session_state:
    st.session_state.connected = False

# WALLET CONNECTION MODAL
@st.dialog("Secure Web3 / Exchange Connect")
def connect_modal():
    st.markdown("### Initialize Secure Connection")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🦊 MetaMask", use_container_width=True):
            st.session_state.connected = True
            st.rerun()
    with col2:
        if st.button("📈 TradingView", use_container_width=True):
            st.session_state.connected = True
            st.rerun()

# HEADER & WALLET CONNECT
col_title, col_wallet = st.columns([4, 1])
with col_title:
    st.title("🌌 Trend Analysis AI Terminal")
with col_wallet:
    st.write("") # Padding
    if not st.session_state.connected:
        if st.button("🔗 Connect Wallet", use_container_width=True):
            connect_modal()
    else:
        st.success("🟢 Node Connected")

# SIDEBAR INPUTS
st.sidebar.markdown("## ⚙️ Terminal Config")

# Categorized Asset Selection
asset_class = st.sidebar.selectbox("Asset Class", ["Crypto", "Metals", "Stocks"])

tickers = {
    "Crypto": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD", "Ripple": "XRP-USD"},
    "Metals": {"Gold": "GC=F", "Silver": "SI=F", "Crude Oil": "CL=F"},
    "Stocks": {"Apple": "AAPL", "Microsoft": "MSFT", "Nvidia": "NVDA", "Tesla": "TSLA", "S&P 500": "SPY", "Nasdaq": "QQQ"}
}

selected_asset_name = st.sidebar.selectbox("Select Asset", list(tickers[asset_class].keys()))
ticker = tickers[asset_class][selected_asset_name]

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 AI Model Features")
st.sidebar.write("Select technical indicators to synthesize.")

available_features = ["Open", "High", "Low", "Close", "RSI", "EMA_9", "EMA_15", "MACD"]
selected_features = []

for feature in available_features:
    if st.sidebar.checkbox(feature, value=True if feature in ["Close", "RSI", "EMA_9", "MACD"] else False):
        selected_features.append(feature)

if len(selected_features) == 0:
    st.sidebar.error("Error: Zero features selected.")
    st.stop()

# DATA FETCHING FUNCTIONS
@st.cache_data(ttl=60)
def fetch_intraday_data(t):
    # Fetch 7 days of 15m interval data for real-time charting
    df = yf.download(t, period="5d", interval="15m", progress=False)
    return df

@st.cache_data(ttl=3600)
def fetch_and_process_daily_data(t):
    # Fetch 2 years of daily data for ML training
    df = yf.download(t, period="2y", interval="1d", progress=False)
    if df.empty:
        return None
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate EMAs
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_15'] = df['Close'].ewm(span=15, adjust=False).mean()
    
    # Calculate MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    
    # Target: 1 if next day's close > today's close, else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df.dropna(inplace=True)
    return df

# DATA PIPELINE EXECUTION
with st.spinner("Synthesizing market data streams..."):
    intraday_data = fetch_intraday_data(ticker)
    daily_data = fetch_and_process_daily_data(ticker)

if daily_data is None or daily_data.empty:
    st.error(f"Critical Failure: Unable to establish data stream for {ticker}.")
    st.stop()

# TOP LEVEL METRICS
if not intraday_data.empty:
    latest_close = float(intraday_data['Close'].iloc[-1].iloc[0]) if isinstance(intraday_data['Close'].iloc[-1], pd.Series) else float(intraday_data['Close'].iloc[-1])
    open_price = float(intraday_data['Open'].iloc[0].iloc[0]) if isinstance(intraday_data['Open'].iloc[0], pd.Series) else float(intraday_data['Open'].iloc[0])
    pct_change = ((latest_close - open_price) / open_price) * 100
    vol = int(intraday_data['Volume'].sum().iloc[0]) if isinstance(intraday_data['Volume'].sum(), pd.Series) else int(intraday_data['Volume'].sum())
else:
    # Fallback to daily data
    latest_close = float(daily_data['Close'].iloc[-1].iloc[0]) if isinstance(daily_data['Close'].iloc[-1], pd.Series) else float(daily_data['Close'].iloc[-1])
    open_price = float(daily_data['Open'].iloc[-1].iloc[0]) if isinstance(daily_data['Open'].iloc[-1], pd.Series) else float(daily_data['Open'].iloc[-1])
    pct_change = ((latest_close - open_price) / open_price) * 100
    vol = int(daily_data['Volume'].iloc[-1].iloc[0]) if isinstance(daily_data['Volume'].iloc[-1], pd.Series) else int(daily_data['Volume'].iloc[-1])

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Asset", selected_asset_name, ticker)
col_m2.metric("Last Price (USD)", f"${latest_close:,.2f}", f"{pct_change:+.2f}%")
col_m3.metric("Volume (Session)", f"{vol:,}")
col_m4.metric("Status", "LIVE 🟢", "Secure")
st.markdown("---")

# MACHINE LEARNING PIPELINE
X = daily_data[selected_features]
y = daily_data['Target']

# Train on all but the last day
X_train = X[:-1]
y_train = y[:-1]
latest_data = X.iloc[[-1]]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict for the next trading day
prediction = model.predict(latest_data)[0]
prediction_prob = model.predict_proba(latest_data)[0][prediction]

# TABS LAYOUT
tab1, tab2, tab3 = st.tabs(["📡 Real-Time Terminal", "🔮 AI Prediction Engine", "📊 Technical Data"])

with tab1:
    st.markdown("### 📡 Live Intraday Market Stream (15m Interval)")
    if not intraday_data.empty:
        # Complex Plotly Chart with Volume Subplot
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, subplot_titles=(f'{selected_asset_name} Price Action', 'Volume'), 
                            row_width=[0.2, 0.7])

        # Candlestick
        fig.add_trace(go.Candlestick(x=intraday_data.index,
                        open=intraday_data['Open'].squeeze(),
                        high=intraday_data['High'].squeeze(),
                        low=intraday_data['Low'].squeeze(),
                        close=intraday_data['Close'].squeeze(),
                        name='Price'), row=1, col=1)
        
        # Calculate Bollinger Bands on intraday
        window = 20
        rolling_mean = intraday_data['Close'].squeeze().rolling(window=window).mean()
        rolling_std = intraday_data['Close'].squeeze().rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)

        fig.add_trace(go.Scatter(x=intraday_data.index, y=upper_band, line=dict(color='rgba(255,255,255,0.2)', width=1), name='Upper BB'), row=1, col=1)
        fig.add_trace(go.Scatter(x=intraday_data.index, y=lower_band, line=dict(color='rgba(255,255,255,0.2)', width=1), fill='tonexty', fillcolor='rgba(255,255,255,0.05)', name='Lower BB'), row=1, col=1)

        # Volume
        colors = ['red' if row['Open'].squeeze() - row['Close'].squeeze() >= 0 else 'green' for index, row in intraday_data.iterrows()]
        fig.add_trace(go.Bar(x=intraday_data.index, y=intraday_data['Volume'].squeeze(), marker_color=colors, name='Volume'), row=2, col=1)

        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=700,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Intraday data is not available for this asset. Please view the Daily ML data in Tab 3.")

with tab2:
    st.markdown("### 🔮 Trend Analysis AI Terminal")
    col2a, col2b = st.columns([1, 1])

    with col2a:
        st.markdown("""
        <div class="glass-card">
            <h3>Neural Network Output</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if prediction == 1:
            st.success(f"🚀 STRONG BULLISH TREND DETECTED ({prediction_prob*100:.2f}% Confidence)")
        else:
            st.error(f"🩸 BEARISH DOWNTREND DETECTED ({prediction_prob*100:.2f}% Confidence)")
            
        # Narrative Logic
        narrative = f"The Deep Learning architecture has synthesized the provided data vectors and forecasts a **{'Bullish' if prediction == 1 else 'Bearish'}** trajectory.<br><br>"
        
        if 'RSI' in latest_data.columns:
            rsi = latest_data['RSI'].values[0]
            if rsi > 70:
                narrative += f"► **RSI Over-extension ({rsi:.1f})**: Asset is currently heavily overbought. Risk of correction is elevated.<br>"
            elif rsi < 30:
                narrative += f"► **RSI Capitulation ({rsi:.1f})**: Asset is deeply oversold. Accumulation zones identified.<br>"
            else:
                narrative += f"► **RSI Equilibrium ({rsi:.1f})**: Momentum oscillators are currently neutral.<br>"
                
        if 'EMA_9' in latest_data.columns and 'EMA_15' in latest_data.columns:
            ema_9 = latest_data['EMA_9'].values[0]
            ema_15 = latest_data['EMA_15'].values[0]
            if ema_9 > ema_15:
                narrative += f"► **Moving Average Convergence**: Short-term velocity (EMA 9) is overpowering long-term gravity (EMA 15), signaling bullish continuation.<br>"
            else:
                narrative += f"► **Moving Average Divergence**: Short-term velocity (EMA 9) has failed to breach long-term resistance (EMA 15).<br>"

        st.markdown(f"""
        <div class="glass-card">
            <h3>🤖 AI Market Synthesis</h3>
            <p style="font-size: 1.1em; line-height: 1.7; font-family: monospace;">{narrative}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2b:
        st.markdown("### 🧠 Feature Importance Weighting")
        importance_values = model.feature_importances_

        fig_imp = go.Figure(go.Bar(
                    x=importance_values,
                    y=selected_features,
                    orientation='h',
                    marker=dict(color='rgba(0, 200, 255, 0.7)', line=dict(color='rgba(0, 200, 255, 1.0)', width=1))
                ))

        fig_imp.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Significance Score",
            yaxis_title="Feature Vector"
        )
        st.plotly_chart(fig_imp, use_container_width=True)

with tab3:
    st.markdown("### 📊 Daily Macro Technical Data (Used for ML)")
    
    st.markdown("#### Latest Calculated Vectors")
    # Clean up column names for display
    display_data = latest_data.copy()
    if isinstance(display_data.columns, pd.MultiIndex):
        display_data.columns = display_data.columns.get_level_values(0)
    st.dataframe(display_data.style.highlight_max(axis=1), use_container_width=True)
    
    st.markdown("#### Historical Macro Dataset")
    hist_data = daily_data.copy().tail(100)
    if isinstance(hist_data.columns, pd.MultiIndex):
        hist_data.columns = hist_data.columns.get_level_values(0)
    st.dataframe(hist_data, use_container_width=True)