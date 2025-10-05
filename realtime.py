# app.py
import os
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Suppress Streamlit warnings
import logging
logging.getLogger('streamlit').setLevel(logging.ERROR)

# Page configuration
st.set_page_config(
    page_title="AI Stock Predictor Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        padding: 20px;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 20px;
        color: white;
        margin: 15px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        border: none;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
    }
    .stock-price {
        font-size: 3rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .price-up {
        color: #00ff88;
    }
    .price-down {
        color: #ff4444;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin: 25px 0 15px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid #667eea;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 10px;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

class RealTimeStockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
        self.current_data = None
        
    def get_stock_info(self, ticker):
        """Get real-time stock information from Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get recent data (last 7 days to ensure we have current data)
            history = stock.history(period="7d", interval="1d")
            
            if history.empty:
                return None
                
            current_price = history['Close'].iloc[-1]
            previous_close = history['Close'].iloc[-2] if len(history) > 1 else current_price
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100
            
            return {
                'current_price': current_price,
                'previous_close': previous_close,
                'change': change,
                'change_percent': change_percent,
                'company_name': info.get('longName', ticker),
                'volume': history['Volume'].iloc[-1],
                'market_cap': info.get('marketCap', 0),
                'day_high': history['High'].iloc[-1],
                'day_low': history['Low'].iloc[-1],
                'open_price': history['Open'].iloc[-1]
            }
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def get_historical_data(self, ticker, period="1y"):
        """Get historical stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            return data
        except Exception as e:
            st.error(f"Error fetching historical data: {str(e)}")
            return None
    
    def prepare_features(self, data):
        """Create technical indicators and features for machine learning"""
        df = data.copy()
        
        # Price-based features
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Price changes and momentum
        df['Price_Change'] = df['Close'].pct_change()
        df['Momentum'] = df['Close'] - df['Close'].shift(5)
        
        # Volume features
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Support and Resistance (simplified)
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Support'] = df['Low'].rolling(window=20).min()
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def train_model(self, data, model_type='random_forest'):
        """Train the prediction model with historical data"""
        df = self.prepare_features(data)
        
        if df.empty:
            return df, np.array([]), 0, 0
        
        # Features for prediction
        feature_columns = ['SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'Volatility', 
                          'RSI', 'MACD', 'MACD_Signal', 'Volume', 'Volume_SMA',
                          'Momentum', 'Volume_Ratio']
        
        X = df[feature_columns]
        y = df['Close']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        
        self.model.fit(X_scaled, y)
        
        # Make predictions on training data for evaluation
        predictions = self.model.predict(X_scaled)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        
        return df, predictions, mae, rmse
    
    def predict_next_day(self, current_data):
        """Predict next day's price using trained model"""
        if self.model is None:
            return None, None
        
        df = self.prepare_features(current_data)
        if df.empty:
            return None, None
        
        feature_columns = ['SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'Volatility', 
                          'RSI', 'MACD', 'MACD_Signal', 'Volume', 'Volume_SMA',
                          'Momentum', 'Volume_Ratio']
        
        # Get the latest features
        latest_features = df[feature_columns].iloc[-1:].values
        latest_features_scaled = self.scaler.transform(latest_features)
        
        # Make prediction
        prediction = self.model.predict(latest_features_scaled)[0]
        current_price = df['Close'].iloc[-1]
        
        return prediction, current_price

def format_currency(value):
    """Format large numbers as currency strings"""
    if value >= 1e12:
        return f"${value/1e12:.2f}T"
    elif value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    else:
        return f"${value:,.0f}"

def format_number(value):
    """Format large numbers with commas"""
    return f"{value:,.0f}"

def main():
    # Header with modern design
    st.markdown('<div class="main-header">🚀 AI Stock Predictor Pro</div>', unsafe_allow_html=True)
    
    # Initialize predictor
    predictor = RealTimeStockPredictor()
    
    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown("### 🔧 Control Panel")
        st.markdown("---")
        
        # Stock selection
        popular_stocks = {
            "Apple (AAPL)": "AAPL",
            "Microsoft (MSFT)": "MSFT", 
            "Google (GOOGL)": "GOOGL",
            "Amazon (AMZN)": "AMZN",
            "Tesla (TSLA)": "TSLA",
            "NVIDIA (NVDA)": "NVDA",
            "Meta (META)": "META",
            "Netflix (NFLX)": "NFLX",
            "AMD (AMD)": "AMD",
            "Intel (INTC)": "INTC"
        }
        
        selected_stock_name = st.selectbox(
            "📊 Select Stock:",
            list(popular_stocks.keys()),
            index=0
        )
        ticker = popular_stocks[selected_stock_name]
        
        # Model selection
        model_type = st.selectbox(
            "🤖 Prediction Model:",
            ["random_forest", "linear"],
            format_func=lambda x: "Random Forest" if x == "random_forest" else "Linear Regression"
        )
        
        # Data period
        period = st.selectbox(
            "📅 Data Period:",
            ["3mo", "6mo", "1y", "2y", "5y"],
            index=2
        )
        
        # Auto-refresh option
        auto_refresh = st.checkbox("🔄 Enable Auto-Refresh (30s)", value=True)
        
        st.markdown("---")
        st.markdown("### 📈 Real-Time Data Source")
        st.markdown("""
        **Data Provider:** Yahoo Finance
        **Update Frequency:** Real-time during market hours
        **Data Includes:** Price, Volume, Technical Indicators
        **Coverage:** Global stocks, ETFs, indices
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ About This App")
        st.markdown("""
        This AI-powered stock predictor uses machine learning to analyze historical patterns and technical indicators to forecast price movements.
        
        **Disclaimer:** Predictions are for educational purposes only. Always do your own research before investing.
        """)

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown('<div class="section-header">💰 Real-Time Data</div>', unsafe_allow_html=True)
        
        # Get real-time data with progress indicator
        with st.spinner('🔄 Fetching real-time data...'):
            stock_info = predictor.get_stock_info(ticker)
        
        if stock_info:
            # Current price card with enhanced design
            price_color = "price-up" if stock_info['change'] >= 0 else "price-down"
            change_icon = "📈" if stock_info['change'] >= 0 else "📉"
            
            st.markdown(f"""
            <div class="prediction-card">
                <h3>{stock_info['company_name']} ({ticker})</h3>
                <div class="stock-price {price_color}">${stock_info['current_price']:.2f}</div>
                <div style="font-size: 1.2rem; margin-top: 10px;">
                    {change_icon} {stock_info['change']:+.2f} ({stock_info['change_percent']:+.2f}%)
                </div>
                <div style="margin-top: 15px; font-size: 0.9rem;">
                    Previous Close: ${stock_info['previous_close']:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Key metrics in cards
            st.markdown("#### 📊 Key Metrics")
            
            col_met1, col_met2 = st.columns(2)
            with col_met1:
                st.markdown(f"""
                <div class="metric-card">
                    <strong>Open</strong><br>
                    ${stock_info['open_price']:.2f}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <strong>Day High</strong><br>
                    ${stock_info['day_high']:.2f}
                </div>
                """, unsafe_allow_html=True)
                
            with col_met2:
                st.markdown(f"""
                <div class="metric-card">
                    <strong>Day Low</strong><br>
                    ${stock_info['day_low']:.2f}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <strong>Volume</strong><br>
                    {format_number(stock_info['volume'])}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <strong>Market Cap</strong><br>
                {format_currency(stock_info['market_cap'])}
            </div>
            """, unsafe_allow_html=True)
    
    with col1:
        st.markdown('<div class="section-header">📈 Price Analysis & Predictions</div>', unsafe_allow_html=True)
        
        # Get historical data
        with st.spinner('📊 Loading historical data and training model...'):
            historical_data = predictor.get_historical_data(ticker, period)
        
        if historical_data is not None and not historical_data.empty:
            # Train model and get predictions
            trained_data, predictions, mae, rmse = predictor.train_model(historical_data, model_type)
            
            # Create interactive price chart with more features
            fig = go.Figure()
            
            # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=historical_data.index,
                open=historical_data['Open'],
                high=historical_data['High'],
                low=historical_data['Low'],
                close=historical_data['Close'],
                name='Price',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ))
            
            # Moving averages
            if 'SMA_20' in trained_data.columns:
                fig.add_trace(go.Scatter(
                    x=trained_data.index,
                    y=trained_data['SMA_20'],
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                ))
            
            if 'SMA_50' in trained_data.columns:
                fig.add_trace(go.Scatter(
                    x=trained_data.index,
                    y=trained_data['SMA_50'],
                    name='SMA 50',
                    line=dict(color='blue', width=1)
                ))
            
            fig.update_layout(
                title=f"{selected_stock_name} - Price Chart with Technical Indicators",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=500,
                showlegend=True,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Next day prediction
            next_day_pred, current_price = predictor.predict_next_day(historical_data)
            
            if next_day_pred is not None:
                pred_change = next_day_pred - current_price
                pred_change_percent = (pred_change / current_price) * 100
                
                st.markdown("#### 🔮 Next Day Prediction")
                
                col_pred1, col_pred2, col_pred3 = st.columns(3)
                
                with col_pred1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>Current Price</strong><br>
                        <span style="font-size: 1.5rem;">${current_price:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_pred2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>Predicted Price</strong><br>
                        <span style="font-size: 1.5rem;">${next_day_pred:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_pred3:
                    direction = "Bullish 📈" if pred_change >= 0 else "Bearish 📉"
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>Prediction</strong><br>
                        <span style="font-size: 1.2rem; color: {'#00ff88' if pred_change >= 0 else '#ff4444'}">
                            {pred_change_percent:+.2f}%
                        </span><br>
                        <small>{direction}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Prediction confidence based on model performance
                confidence = max(0, min(100, 100 - (rmse / current_price * 100)))
                st.markdown(f"**Model Confidence: {confidence:.1f}%**")
                st.progress(int(confidence))
            
            # Technical indicators in expandable section
            with st.expander("🔍 Detailed Technical Analysis", expanded=False):
                if not trained_data.empty:
                    col_tech1, col_tech2, col_tech3, col_tech4 = st.columns(4)
                    
                    with col_tech1:
                        current_rsi = trained_data['RSI'].iloc[-1]
                        rsi_status = "🔴 Overbought" if current_rsi > 70 else "🟢 Oversold" if current_rsi < 30 else "🟡 Neutral"
                        st.metric("RSI", f"{current_rsi:.1f}", rsi_status)
                    
                    with col_tech2:
                        macd = trained_data['MACD'].iloc[-1]
                        macd_signal = trained_data['MACD_Signal'].iloc[-1]
                        macd_status = "🟢 Bullish" if macd > macd_signal else "🔴 Bearish"
                        st.metric("MACD", f"{macd:.3f}", macd_status)
                    
                    with col_tech3:
                        sma_20 = trained_data['SMA_20'].iloc[-1]
                        price_vs_sma = ((current_price - sma_20) / sma_20) * 100
                        st.metric("VS SMA(20)", f"{price_vs_sma:+.1f}%")
                    
                    with col_tech4:
                        volatility = trained_data['Volatility'].iloc[-1]
                        vol_status = "High" if volatility > current_price * 0.02 else "Low"
                        st.metric("Volatility", f"{volatility:.2f}", vol_status)
            
            # Model performance
            st.markdown("#### 🎯 Model Performance")
            col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
            
            with col_perf1:
                st.metric("Mean Absolute Error", f"${mae:.2f}")
            
            with col_perf2:
                st.metric("Root Mean Square Error", f"${rmse:.2f}")
            
            with col_perf3:
                accuracy = max(0, 100 - (mae / historical_data['Close'].mean() * 100))
                st.metric("Accuracy", f"{accuracy:.1f}%")
            
            with col_perf4:
                st.metric("Data Points", f"{len(trained_data):,}")
        
        else:
            st.error("❌ Unable to fetch historical data. Please check your internet connection and try again.")
    
    # Refresh logic
    if auto_refresh:
        refresh_time = 30  # seconds
        time.sleep(refresh_time)
        st.rerun()

if __name__ == "__main__":
    main()
