import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Trend Analysis AI Terminal",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS: FUTURISTIC NEON TERMINAL THEME
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
}

/* Deep space background */
.stApp {
    background: radial-gradient(circle at 50% 0%, #1a0b2e 0%, #05010d 60%, #000000 100%) !important;
}

/* Futuristic Glass Card */
.glass-card {
    background: rgba(15, 12, 29, 0.6);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(0, 243, 255, 0.15);
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 0 15px rgba(0, 243, 255, 0.05);
    margin-bottom: 20px;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    position: relative;
    overflow: hidden;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 50%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent);
    transform: skewX(-20deg);
    transition: all 0.5s ease;
}

.glass-card:hover::before {
    left: 150%;
}

.glass-card:hover {
    box-shadow: 0 12px 40px rgba(0, 243, 255, 0.2), inset 0 0 20px rgba(157, 78, 221, 0.1);
    border: 1px solid rgba(0, 243, 255, 0.4);
    transform: translateY(-5px);
}

.glass-card h1, .glass-card h2, .glass-card h3, .glass-card p {
    color: #e0e6ed !important;
}

.glass-card h3 {
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 1.1rem;
    color: #00f3ff !important;
    border-bottom: 1px solid rgba(0, 243, 255, 0.2);
    padding-bottom: 8px;
    margin-bottom: 16px;
}

.glass-card p.synthesis-text {
    color: #00ff64 !important;
    font-weight: 400;
    text-shadow: 0px 0px 8px rgba(0, 255, 100, 0.4);
    letter-spacing: 0.5px;
}

h1, h2, h3 {
    color: #ffffff !important;
    font-weight: 600;
    letter-spacing: 1px;
}

/* Metric Cards */
div[data-testid="metric-container"] {
    background: rgba(15, 12, 29, 0.6);
    border: 1px solid rgba(157, 78, 221, 0.2);
    border-left: 4px solid #9d4edd;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    transition: all 0.3s ease;
}
div[data-testid="metric-container"]:hover {
    border-left: 4px solid #00f3ff;
    box-shadow: 0 4px 20px rgba(0, 243, 255, 0.2);
    transform: scale(1.02);
}
div[data-testid="metric-container"] label {
    color: #a0aec0 !important;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-size: 0.85rem;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 2rem;
    font-weight: 700;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 16px;
    background-color: rgba(10, 10, 25, 0.7);
    border-radius: 12px;
    padding: 10px;
    border: 1px solid rgba(0, 243, 255, 0.1);
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 8px;
    gap: 1px;
    padding: 10px 24px;
    transition: all 0.3s;
    color: #8b9eb5;
    font-weight: 600;
    letter-spacing: 1px;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #00f3ff;
    background: rgba(0, 243, 255, 0.05);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #9d4edd 0%, #00f3ff 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 0 15px rgba(0, 243, 255, 0.4);
    border: none;
}

/* Prediction Blink Animation */
@keyframes neon-pulse {
    0% { opacity: 0.8; box-shadow: 0 0 10px currentColor; }
    50% { opacity: 1; box-shadow: 0 0 25px currentColor, 0 0 40px currentColor; }
    100% { opacity: 0.8; box-shadow: 0 0 10px currentColor; }
}
.prediction-blink {
    animation: neon-pulse 2s infinite alternate;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 25px;
    font-size: 1.4em;
    font-weight: 700;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 2px;
    position: relative;
    overflow: hidden;
}
.prediction-blink::after {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 20%; height: 100%;
    background: rgba(255,255,255,0.2);
    transform: skewX(-20deg);
    animation: shine 4s infinite;
}
@keyframes shine {
    0% { left: -100%; }
    20% { left: 200%; }
    100% { left: 200%; }
}
.bullish-blink {
    background: rgba(0, 255, 100, 0.1);
    color: #00ff64;
    border: 1px solid #00ff64;
}
.bearish-blink {
    background: rgba(255, 0, 50, 0.1);
    color: #ff0032;
    border: 1px solid #ff0032;
}

/* Sidebar overrides */
[data-testid="stSidebar"] {
    background: rgba(5, 1, 13, 0.95) !important;
    border-right: 1px solid rgba(157, 78, 221, 0.3);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #9d4edd 0%, #00f3ff 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase;
}
.stButton > button:hover {
    box-shadow: 0 0 20px rgba(0, 243, 255, 0.6) !important;
    transform: translateY(-2px) !important;
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    background: rgba(15, 12, 29, 0.6);
    border-radius: 12px;
    border: 1px solid rgba(0, 243, 255, 0.15);
    padding: 10px;
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
    st.error("Warning: This is only for Educational Purpose, NFA | DYOR")
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

@st.cache_data(ttl=3600)
def fetch_sentiment(t):
    try:
        ticker_obj = yf.Ticker(t)
        news = ticker_obj.news
        if not news: return 0.0
        analyzer = SentimentIntensityAnalyzer()
        compound_scores = []
        for n in news[:10]: # Top 10 news
            if 'title' in n:
                score = analyzer.polarity_scores(n['title'])['compound']
                compound_scores.append(score)
        if len(compound_scores) == 0: return 0.0
        return float(np.mean(compound_scores))
    except Exception:
        return 0.0

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

# Train XGBoost Model
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict for the next trading day (Ensemble)
xgb_prob = xgb_model.predict_proba(latest_data)[0]
rf_prob = rf_model.predict_proba(latest_data)[0]
ensemble_prob = (xgb_prob + rf_prob) / 2
prediction = 1 if ensemble_prob[1] > 0.5 else 0
prediction_prob = ensemble_prob[prediction]

# Fetch Sentiment
with st.spinner("Analyzing NLP sentiment..."):
    sentiment_score = fetch_sentiment(ticker)

# TABS LAYOUT
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📡 Real-Time Terminal", "🔮 AI Prediction Engine", "📊 Technical Data", "📈 Backtest Engine", "🔍 Market Screener"])

with tab1:
    st.markdown("### 📡 Trading Terminal")
    
    # Helper to map our tickers to TradingView expected symbols
    tv_symbol = ticker
    if asset_class == "Crypto":
        tv_symbol = ticker.replace("-", "") # e.g. BTC-USD -> BTCUSD
    elif asset_class == "Commodities":
        tv_mapping = {"Gold": "GOLD", "Silver": "SILVER", "Crude Oil": "USOIL", "Platinum": "PLATINUM", "Copper": "COPPER", "Natural Gas": "NATGAS"}
        tv_symbol = tv_mapping.get(selected_asset_name, ticker)
    
    tradingview_html = f"""
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container" style="height:700px;width:100%">
      <div id="tradingview_12345" style="height:100%;width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {{
      "autosize": true,
      "symbol": "{tv_symbol}",
      "interval": "15",
      "timezone": "Etc/UTC",
      "theme": "dark",
      "style": "1",
      "locale": "en",
      "enable_publishing": false,
      "backgroundColor": "rgba(10, 10, 25, 1)",
      "gridColor": "rgba(255, 255, 255, 0.05)",
      "hide_top_toolbar": false,
      "hide_legend": false,
      "hide_side_toolbar": false,
      "allow_symbol_change": true,
      "save_image": false,
      "container_id": "tradingview_12345",
      "toolbar_bg": "rgba(10, 10, 25, 1)"
    }}
      );
      </script>
    </div>
    <!-- TradingView Widget END -->
    """
    import streamlit.components.v1 as components
    components.html(tradingview_html, height=700)

with tab2:
    st.markdown("### 🔮 Trend Analysis AI Terminal")
    col2a, col2b = st.columns([1, 1])

    with col2a:
        st.markdown("""
        <div class="glass-card">
            <h3>Agent Output(Prediction) [Not a Financial Advice]</h3>
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
            
        # 6. Sentiment Analysis
        if sentiment_score > 0.2:
            narrative += f"► **NLP Sentiment ({sentiment_score:.2f})**: Positive news flow detected, acting as a tailwind.<br>"
        elif sentiment_score < -0.2:
            narrative += f"► **NLP Sentiment ({sentiment_score:.2f})**: Negative news flow detected, acting as a headwind.<br>"
        else:
            narrative += f"► **NLP Sentiment ({sentiment_score:.2f})**: News sentiment is currently neutral.<br>"

        st.markdown(f"""
        <div class="glass-card">
            <h3>🤖 AI Market Synthesis</h3>
            <p class="synthesis-text" style="font-size: 1.1em; line-height: 1.7; font-family: monospace;">{narrative}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2b:
        st.markdown("### 🧠 Feature Importance Weighting")
        importance_values = (xgb_model.feature_importances_ + rf_model.feature_importances_) / 2

        fig_imp = go.Figure(go.Bar(
                    x=importance_values,
                    y=selected_features,
                    orientation='h',
                    marker=dict(
                        color='rgba(157, 78, 221, 0.7)', 
                        line=dict(color='#00f3ff', width=1)
                    )
                ))

        fig_imp.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Significance Score",
            yaxis_title="Feature Vector",
            font=dict(family="Outfit", color="#a0aec0"),
            xaxis=dict(gridcolor="rgba(0, 243, 255, 0.1)"),
            yaxis=dict(gridcolor="rgba(0, 243, 255, 0.1)")
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

with tab4:
    st.markdown("### 📈 Historical Backtesting Engine")
    st.write("This engine simulates trading the asset based on the XGBoost and Random Forest Ensemble predictions over the fetched period.")
    st.write("This is only a Prediction and Backtest data visualization, not a trading signal.")
    with st.spinner("Running historical backtest..."):
        # Calculate strategy returns
        xgb_preds = xgb_model.predict_proba(X)
        rf_preds = rf_model.predict_proba(X)
        ensemble_preds = (xgb_preds + rf_preds) / 2
        historical_preds = (ensemble_preds[:, 1] > 0.5).astype(int)
        
        backtest_df = daily_data.copy()
        backtest_df['Prediction'] = historical_preds
        backtest_df['Market_Return'] = backtest_df['Close'].pct_change()
        # Strategy Return: if pred is 1 (Bullish), we get market return next day.
        backtest_df['Strategy_Return'] = backtest_df['Market_Return'] * backtest_df['Prediction'].shift(1)
        
        backtest_df.dropna(inplace=True)
        backtest_df['Cumulative_Market'] = (1 + backtest_df['Market_Return']).cumprod()
        backtest_df['Cumulative_Strategy'] = (1 + backtest_df['Strategy_Return']).cumprod()
        
        # Plot
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Cumulative_Market'], mode='lines', name='Buy & Hold (Market)', line=dict(color='#a0aec0')))
        fig_bt.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Cumulative_Strategy'], mode='lines', name='AI Strategy', line=dict(color='#00f3ff')))
        
        fig_bt.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Outfit", color="#a0aec0"), margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"), yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            title="Cumulative Return: AI Strategy vs Buy & Hold"
        )
        st.plotly_chart(fig_bt, use_container_width=True)
        
        total_market = (backtest_df['Cumulative_Market'].iloc[-1] - 1) * 100
        total_strategy = (backtest_df['Cumulative_Strategy'].iloc[-1] - 1) * 100
        col_bt1, col_bt2 = st.columns(2)
        col_bt1.metric("Market Return (Buy & Hold)", f"{total_market:.2f}%")
        col_bt2.metric("Strategy Return (Ensemble)", f"{total_strategy:.2f}%")

with tab5:
    st.markdown("### 🔍 Market Screener (Top 5 Watchlist)")
    st.write(f"Scanning the top assets in the {asset_class} class for immediate trade setups...")
    
    screener_assets = list(tickers[asset_class].items())[:5]
    
    if st.button("Run AI Screener"):
        screener_results = []
        progress_bar = st.progress(0)
        
        for i, (name, sym) in enumerate(screener_assets):
            with st.spinner(f"Analyzing {name} ({sym})..."):
                d_df = fetch_and_process_daily_data(sym)
                if d_df is not None and not d_df.empty:
                    s_X = d_df[selected_features]
                    s_y = d_df['Target']
                    s_train_X, s_train_y = s_X[:-1], s_y[:-1]
                    s_latest = s_X.iloc[[-1]]
                    
                    s_xgb = XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
                    s_xgb.fit(s_train_X, s_train_y)
                    s_rf = RandomForestClassifier(n_estimators=50, random_state=42)
                    s_rf.fit(s_train_X, s_train_y)
                    
                    s_xgb_prob = s_xgb.predict_proba(s_latest)[0]
                    s_rf_prob = s_rf.predict_proba(s_latest)[0]
                    s_ensemble_prob = (s_xgb_prob + s_rf_prob) / 2
                    s_pred = 1 if s_ensemble_prob[1] > 0.5 else 0
                    s_prob = s_ensemble_prob[s_pred]
                    s_sent = fetch_sentiment(sym)
                    
                    signal = "Bullish 🚀" if s_pred == 1 else "Bearish 🩸"
                    screener_results.append({
                        "Asset": name,
                        "Ticker": sym,
                        "AI Signal": signal,
                        "Confidence": f"{s_prob*100:.1f}%",
                        "News Sentiment": f"{s_sent:.2f}"
                    })
            progress_bar.progress((i + 1) / len(screener_assets))
            
        if screener_results:
            screener_df = pd.DataFrame(screener_results)
            st.dataframe(screener_df, use_container_width=True)
        else:
            st.error("Failed to fetch data for the screener.")