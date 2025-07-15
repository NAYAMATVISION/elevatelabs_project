import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Load trained model and scaler
model = load_model('model.h5')
scaler = joblib.load('scaler.pkl')

# Set page config
st.set_page_config(
    page_title="AI Stock Price Predictor",
    layout="wide",
    page_icon="📊"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .ticker-input {
        font-size: 18px !important;
    }
    .prediction-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .header {
        color: #2c3e50;
    }
    .positive {
        color: #27ae60;
    }
    .negative {
        color: #e74c3c;
    }
    </style>
    """, unsafe_allow_html=True)

# App title and header
st.markdown("<h1 class='header'>📈 AI Stock Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("""
    <p style='color: #7f8c8d; font-size: 16px;'>
    This advanced LSTM model predicts the next day's closing price using historical price data and technical indicators.
    </p>
    <hr style='border: 1px solid #eee; margin-bottom: 30px;'>
    """, unsafe_allow_html=True)

# Create columns for layout
col1, col2 = st.columns([1, 3])

with col1:
    # Ticker input in a card
    with st.container():
        st.markdown("<h3 style='color: #2c3e50;'>Stock Selection</h3>", unsafe_allow_html=True)
        ticker = st.text_input("Enter stock ticker (e.g., AAPL, MSFT, TSLA):", "AAPL", key="ticker_input").upper()
        
        # Additional parameters
        st.markdown("<h3 style='color: #2c3e50; margin-top: 20px;'>Model Parameters</h3>", unsafe_allow_html=True)
        lookback_days = st.slider("Lookback Period (days)", 30, 90, 60)
        
        # Prediction button with icon
        predict_btn = st.button("🚀 Predict Next Day Price", use_container_width=True)

with col2:
    # Display stock info card when ticker is entered
    if ticker:
        try:
            stock_info = yf.Ticker(ticker).info
            with st.container():
                st.markdown(f"""
                    <div class='prediction-card'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <h2 style='margin-bottom: 5px;'>{stock_info.get('shortName', ticker)} ({ticker})</h2>
                            <span style='font-size: 14px; color: #7f8c8d;'>{stock_info.get('sector', 'N/A')}</span>
                        </div>
                        <div style='display: flex; align-items: baseline; margin-bottom: 10px;'>
                            <h1 style='margin-right: 10px;'>${stock_info.get('currentPrice', 'N/A')}</h1>
                            <span style='color: {'#27ae60' if stock_info.get('regularMarketChange', 0) >= 0 else '#e74c3c'}; font-weight: bold;'>
                                {stock_info.get('regularMarketChange', 'N/A')} ({stock_info.get('regularMarketChangePercent', 'N/A')}%)
                            </span>
                        </div>
                        <p style='color: #7f8c8d;'>{stock_info.get('longBusinessSummary', '')[:200]}...</p>
                    </div>
                """, unsafe_allow_html=True)
        except:
            st.warning(f"Could not fetch information for {ticker}")

# Prediction logic
if predict_btn and ticker:
    with st.spinner(f"Analyzing {ticker} and predicting next day's closing price..."):
        try:
            # Step 1: Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            data = yf.download(ticker, start=start_date, end=end_date)
            
            if data.empty:
                st.error("No data available for this ticker.")
                st.stop()
                
            data = data[['Close']].copy()
            data['SMA'] = data['Close'].rolling(window=14).mean()
            data.dropna(inplace=True)

            if len(data) < lookback_days:
                st.error(f"Not enough data available (need at least {lookback_days} days).")
                st.stop()

            # Step 2: Prepare input data
            last_n_days = data[['Close', 'SMA']].values[-lookback_days:]
            scaled_input = scaler.transform(last_n_days)
            X_input = np.reshape(scaled_input, (1, lookback_days, 2))

            # Step 3: Predict
            pred_scaled = model.predict(X_input)
            
            # Step 4: Inverse scale
            pred_full = np.hstack((pred_scaled, np.zeros((pred_scaled.shape[0], 1))))
            pred_price = scaler.inverse_transform(pred_full)[0][0]
            
            # Get current price
            current_price = data['Close'].iloc[-1]
            price_change = pred_price - current_price
            percent_change = (price_change / current_price) * 100
            
            # Step 5: Display prediction in a card
            with st.container():
                st.markdown(f"""
                    <div class='prediction-card'>
                        <h2 style='margin-bottom: 10px;'>Prediction Result</h2>
                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>
                            <div>
                                <h3 style='margin-bottom: 5px;'>Current Price</h3>
                                <h1 style='margin-top: 0;'>${current_price:.2f}</h1>
                            </div>
                            <div style='text-align: right;'>
                                <h3 style='margin-bottom: 5px;'>Predicted Next Close</h3>
                                <h1 style='margin-top: 0; color: {'#27ae60' if price_change >= 0 else '#e74c3c'};'>
                                    ${pred_price:.2f} 
                                    <span style='font-size: 20px;'>({'↑' if price_change >= 0 else '↓'} {abs(percent_change):.2f}%)</span>
                                </h1>
                            </div>
                        </div>
                        <p style='color: #7f8c8d; font-size: 14px;'>
                            Prediction based on last {lookback_days} trading days ending {data.index[-1].strftime('%Y-%m-%d')}.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Step 6: Interactive Plotly chart
                st.markdown("<h3 style='color: #2c3e50;'>Price History & Prediction</h3>", unsafe_allow_html=True)
                
                # Create figure
                fig = go.Figure()
                
                # Add historical price trace
                fig.add_trace(go.Scatter(
                    x=data.index[-lookback_days:],
                    y=data['Close'].values[-lookback_days:],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#3498db', width=2)
                ))
                
                # Add SMA trace
                fig.add_trace(go.Scatter(
                    x=data.index[-lookback_days:],
                    y=data['SMA'].values[-lookback_days:],
                    mode='lines',
                    name='14-Day SMA',
                    line=dict(color='#f39c12', width=2, dash='dot')
                ))
                
                # Add prediction point
                next_day = data.index[-1] + timedelta(days=1)
                fig.add_trace(go.Scatter(
                    x=[next_day],
                    y=[pred_price],
                    mode='markers',
                    name='Prediction',
                    marker=dict(color='#27ae60' if price_change >= 0 else '#e74c3c', size=10)
                ))
                
                # Add vertical line at prediction point
                fig.add_vline(
                    x=data.index[-1], 
                    line_dash="dash", 
                    line_color="gray"
                )
                
                # Update layout
                fig.update_layout(
                    height=500,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional metrics section
                st.markdown("<h3 style='color: #2c3e50;'>Technical Metrics</h3>", unsafe_allow_html=True)
                
                # Calculate some basic technical indicators
                latest = data.iloc[-1]
                prev_close = data['Close'].iloc[-2] if len(data) > 1 else latest['Close']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Daily Change", 
                            f"${latest['Close'] - prev_close:.2f}", 
                            f"{(latest['Close'] - prev_close)/prev_close*100:.2f}%")
                
                with col2:
                    st.metric("14-Day SMA", f"${latest['SMA']:.2f}")
                
                with col3:
                    st.metric("Price vs SMA", 
                            f"{(latest['Close'] - latest['SMA'])/latest['SMA']*100:.2f}%",
                            f"${latest['Close'] - latest['SMA']:.2f}")
                
                with col4:
                    st.metric("Predicted Change", 
                            f"${price_change:.2f}", 
                            f"{percent_change:.2f}%")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("""
    <hr style='border: 1px solid #eee; margin-top: 30px;'>
    <div style='text-align: center; color: #7f8c8d; font-size: 14px;'>
        <p>Disclaimer: This prediction is for educational purposes only and should not be considered financial advice.</p>
        <p>Stock market investments are subject to risks. Past performance is not indicative of future results.</p>
    </div>
""", unsafe_allow_html=True)