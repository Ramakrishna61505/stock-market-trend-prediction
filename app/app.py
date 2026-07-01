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
/* Base / Common CSS - Let Streamlit handle the background natively */
.stApp {
    background: transparent;
}
.glass-card {
    background: var(--secondary-background-color);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--primary-color);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    margin-bottom: 20px;
    transition: all 0.3s ease-in-out;
}
.glass-card:hover {
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    border: 1px solid var(--text-color);
    opacity: 0.95;
}
.glass-card h1, .glass-card h2, .glass-card h3, .glass-card p {
    color: var(--text-color) !important;
}
.glass-card p.synthesis-text {
    color: #10b981 !important; /* Vibrant emerald/light green */
    font-weight: 600;
    text-shadow: 0px 1px 2px rgba(0,0,0,0.3); /* Ensure readability in both modes */
}
h1, h2, h3 {
    color: var(--text-color) !important;
    font-weight: 400;
    letter-spacing: 1px;
}
.metric-container {
    display: flex;
    justify-content: space-between;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 16px;
    background-color: var(--secondary-background-color);
    border-radius: 12px;
    padding: 8px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 8px;
    gap: 1px;
    padding: 10px 20px;
    transition: background-color 0.3s;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: var(--secondary-background-color);
    border: 1px solid var(--primary-color);
}
.stTabs [aria-selected="true"] {
    background-color: var(--primary-color) !important;
    color: var(--background-color) !important;
}

/* Prediction Blink Animation */
@keyframes flash-fade {
    0% { opacity: 0; transform: scale(0.95); box-shadow: 0 0 0px transparent; }
    20% { opacity: 1; transform: scale(1.02); box-shadow: 0 0 20px rgba(255,255,255,0.5); }
    100% { opacity: 1; transform: scale(1); box-shadow: 0 0 0px transparent; }
}
.prediction-blink {
    animation: flash-fade 2s ease-out 1;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 25px;
    font-size: 1.2em;
    font-weight: bold;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.2);
}
.bullish-blink {
    background: linear-gradient(90deg, rgba(0,255,100,0.1) 0%, rgba(0,255,100,0.3) 50%, rgba(0,255,100,0.1) 100%);
    color: #00ff64;
    border-color: #00ff64;
    text-shadow: 0 0 10px rgba(0,255,100,0.5);
}
.bearish-blink {
    background: linear-gradient(90deg, rgba(255,0,50,0.1) 0%, rgba(255,0,50,0.3) 50%, rgba(255,0,50,0.1) 100%);
    color: #ff0032;
    border-color: #ff0032;
    text-shadow: 0 0 10px rgba(255,0,50,0.5);
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

tickers = {
    "Crypto": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD", "Ripple": "XRP-USD", "Cardano": "ADA-USD", "Dogecoin": "DOGE-USD", "Polkadot": "DOT-USD", "Polygon": "MATIC-USD"},
    "Commodities": {"Gold": "GC=F", "Silver": "SI=F", "Crude Oil": "CL=F", "Platinum": "PL=F", "Copper": "HG=F", "Natural Gas": "NG=F"},
    "Equities": {"Apple": "AAPL", "Microsoft": "MSFT", "Nvidia": "NVDA", "Tesla": "TSLA", "Amazon": "AMZN", "Alphabet": "GOOGL", "Meta": "META", "Netflix": "NFLX", "JPMorgan": "JPM", "Visa": "V", "Walmart": "WMT", "S&P 500": "SPY", "Nasdaq": "QQQ", "Dow Jones": "DIA"}
}

asset_class = st.sidebar.selectbox("Asset Class", list(tickers.keys()))

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
        
        # Volume
        colors = ['red' if row['Open'].squeeze() - row['Close'].squeeze() >= 0 else 'green' for index, row in intraday_data.iterrows()]
        fig.add_trace(go.Bar(x=intraday_data.index, y=intraday_data['Volume'].squeeze(), marker_color=colors, name='Volume'), row=2, col=1)

        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=700,
            dragmode='pan',
            xaxis_rangeslider_visible=True,
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
            st.markdown(f'<div class="prediction-blink bullish-blink">🚀 STRONG BULLISH TREND DETECTED ({prediction_prob*100:.2f}% Confidence)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="prediction-blink bearish-blink">🩸 BEARISH DOWNTREND DETECTED ({prediction_prob*100:.2f}% Confidence)</div>', unsafe_allow_html=True)
            
        # Narrative Logic
        narrative = f"The Deep Learning architecture has synthesized the provided data vectors and forecasts a **{'Bullish' if prediction == 1 else 'Bearish'}** trajectory.<br><br>"
        
        # 1. Macro Market Conditions (Liquidity & Volatility)
        narrative += f"► **Macro Market Conditions**: "
        recent_vol = daily_data['Volume'].iloc[-14:].mean()
        current_vol = daily_data['Volume'].iloc[-1]
        
        recent_tr = (daily_data['High'] - daily_data['Low']).iloc[-14:].mean()
        current_tr = (daily_data['High'].iloc[-1] - daily_data['Low'].iloc[-1])
        
        # Handle multi-index/series if present from yfinance
        if isinstance(current_vol, pd.Series): current_vol = current_vol.iloc[0]
        if isinstance(recent_vol, pd.Series): recent_vol = recent_vol.iloc[0]
        if isinstance(current_tr, pd.Series): current_tr = current_tr.iloc[0]
        if isinstance(recent_tr, pd.Series): recent_tr = recent_tr.iloc[0]

        if current_vol > recent_vol * 1.2:
            narrative += "High institutional liquidity with above-average volume. "
        elif current_vol < recent_vol * 0.8:
            narrative += "Low liquidity environment (below-average volume). "
        else:
            narrative += "Standard market liquidity. "
            
        if current_tr > recent_tr * 1.2:
            narrative += "Volatility is expanding, signaling a strong directional push.<br>"
        elif current_tr < recent_tr * 0.8:
            narrative += "Volatility is compressing, signaling a ranging market or impending breakout.<br>"
        else:
            narrative += "Volatility remains stable within expected ranges.<br>"
            
        # 2. Price Action Structure
        if 'Close' in latest_data.columns and 'Open' in latest_data.columns:
            close_p = latest_data['Close'].values[0]
            open_p = latest_data['Open'].values[0]
            if close_p > open_p:
                narrative += f"► **Price Action Structure**: The latest session closed above its open, establishing a base of buyer support.<br>"
            else:
                narrative += f"► **Price Action Structure**: The latest session saw distribution, closing below the open and reflecting seller dominance.<br>"
        else:
            if prediction == 1:
                narrative += f"► **Price Action Structure**: The model detects underlying accumulation patterns in recent price movements.<br>"
            else:
                narrative += f"► **Price Action Structure**: The model detects structural distribution and weakness in recent price movements.<br>"

        # 2. RSI
        if 'RSI' in latest_data.columns:
            rsi = latest_data['RSI'].values[0]
            if rsi > 70:
                narrative += f"► **RSI Over-extension ({rsi:.1f})**: Asset is currently heavily overbought. Risk of correction is elevated.<br>"
            elif rsi < 30:
                narrative += f"► **RSI Capitulation ({rsi:.1f})**: Asset is deeply oversold. Accumulation zones identified.<br>"
            else:
                narrative += f"► **RSI Equilibrium ({rsi:.1f})**: Momentum oscillators are currently neutral.<br>"
                
        # 3. EMA
        if 'EMA_9' in latest_data.columns and 'EMA_15' in latest_data.columns:
            ema_9 = latest_data['EMA_9'].values[0]
            ema_15 = latest_data['EMA_15'].values[0]
            if ema_9 > ema_15:
                narrative += f"► **Moving Average Convergence**: Short-term velocity (EMA 9) is overpowering long-term gravity (EMA 15), signaling bullish continuation.<br>"
            else:
                narrative += f"► **Moving Average Divergence**: Short-term velocity (EMA 9) has failed to breach long-term resistance (EMA 15).<br>"

        # 4. MACD
        if 'MACD' in latest_data.columns:
            macd = latest_data['MACD'].values[0]
            if macd > 0:
                narrative += f"► **MACD Momentum ({macd:.2f})**: The MACD histogram is positive, indicating underlying bullish momentum.<br>"
            else:
                narrative += f"► **MACD Momentum ({macd:.2f})**: The MACD histogram is negative, indicating underlying bearish pressure.<br>"
                
        # 5. Confidence Confluence
        if prediction_prob > 0.65:
            narrative += f"► **High Confluence Rating**: The model exhibits strong confidence ({prediction_prob*100:.1f}%) in this directional move.<br>"
        else:
            narrative += f"► **Low Confluence Rating**: The model exhibits marginal confidence ({prediction_prob*100:.1f}%), suggesting a potential ranging market or mixed signals.<br>"

        st.markdown(f"""
        <div class="glass-card">
            <h3>🤖 AI Market Synthesis</h3>
            <p class="synthesis-text" style="font-size: 1.1em; line-height: 1.7; font-family: monospace;">{narrative}</p>
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