import os
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"


from xml.parsers.expat import model
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import pandas_datareader as data
from keras.models import load_model
import yfinance as yf

# Page configuration (should be first Streamlit command)
st.set_page_config(
    page_title="TrendLoom",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .stApp {
            background-image: url('https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-blend-mode: overlay;
            background-color: rgba(255, 255, 255, 0.9);
        }
        
        .main .block-container {
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        
        /* Rest of your existing styles */
        .stTextInput>div>div>input {
            background-color: #f0f2f6;
        }
        
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        
        .stButton>button:hover {
            background-color: #45a049;
        }
        
        .stAlert {
            border-radius: 10px;
        }
        
        .css-1aumxhk {
            background-color: #f0f2f6;
            background-image: none;
        }
        
        .plot-container {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 1rem;
            margin-bottom: 2rem;
            background-color: white;
        }
        
        .header {
            color: #2c3e50;
            border-bottom: 2px solid #2c3e50;
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

start = '2010-01-01'
end = '2020-01-01'

# Main title with better styling
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>📈 TrendLoom</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; margin-bottom: 2rem;'>Predict stock prices using LSTM deep learning model</div>", unsafe_allow_html=True)

# Sidebar for user inputs
with st.sidebar:
    st.markdown("<h2 style='color: #2c3e50;'>Settings</h2>", unsafe_allow_html=True)
    user_input = st.text_input('Enter Stock Ticker', 'AAPL', help="Enter the stock symbol (e.g., AAPL for Apple)")
    st.markdown("---")
    st.markdown("<h3 style='color: #2c3e50;'>About</h3>", unsafe_allow_html=True)
    st.info("""
        This app predicts stock prices using:
        - Historical data from Yahoo Finance
        - LSTM neural network model
        - Technical indicators (100MA, 200MA)
    """)

# Data loading with progress indicator
with st.spinner('Loading stock data...'):
    df = yf.download(user_input, start, end)

if df.empty:
    st.error("No data found for this stock ticker. Please try another one.")
    st.stop()

# Main content in tabs for better organization
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Technical Analysis", "🤖 AI Prediction"])

with tab1:
    st.markdown("<h2 class='header'>Data Overview </h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("<h4>Closing Price History</h4>", unsafe_allow_html=True)
        fig = plt.figure(figsize=(12,6))
        plt.plot(df['Close'], color='#3498db', linewidth=2)
        plt.title(f'{user_input} Closing Price', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Price ($)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
    
    with col2:
        st.markdown("<h4>Statistics</h4>", unsafe_allow_html=True)
        st.dataframe(df.describe().style.format("{:.2f}").highlight_max(axis=0, color='#2ecc71').highlight_min(axis=0, color='#e74c3c'))
        
        st.markdown("---")
        st.markdown("<h4>Recent Data</h4>", unsafe_allow_html=True)
        st.dataframe(df.tail(10).style.format({"Close": "{:.2f}"}))

with tab2:
    st.markdown("<h2 class='header'>Technical Analysis</h2>", unsafe_allow_html=True)
    
    st.markdown("<h4>Moving Averages</h4>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h5>100-Day Moving Average</h5>", unsafe_allow_html=True)
        maa100 = df['Close'].rolling(100).mean()
        fig = plt.figure(figsize=(12,6))
        plt.plot(maa100, 'r', label='100MA', linewidth=2)
        plt.plot(df['Close'], 'b', label='Closing Price', linewidth=1, alpha=0.7)   
        plt.title('100-Day Moving Average', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
    
    with col2:
        st.markdown("<h5>200-Day Moving Average</h5>", unsafe_allow_html=True)
        maa200 = df['Close'].rolling(200).mean()
        fig = plt.figure(figsize=(12,6))
        plt.plot(maa200, 'g', label='200MA', linewidth=2)
        plt.plot(df['Close'], 'b', label='Closing Price', linewidth=1, alpha=0.7)   
        plt.title('200-Day Moving Average', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

with tab3:
    st.markdown("<h2 class='header'>AI Price Prediction</h2>", unsafe_allow_html=True)
    
    with st.spinner('Making predictions...'):
        # Splitting the data into training and testing data
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0,1))
        data_training_array = scaler.fit_transform(data_training)

        # Load model
        try:
            model = load_model('keras_model.h5')
        except:
            st.error("Model file not found. Please ensure 'keras_model.h5' is in the correct directory.")
            st.stop()

        past_100_days = data_training.tail(100)
        final_df = pd.concat([data_testing], ignore_index=True)
        input_data = scaler.fit_transform(final_df)

        x_test = []
        y_test = []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i, 0])
            y_test.append(input_data[i, 0]) 

        x_test, y_test = np.array(x_test), np.array(y_test)
        y_predicted = model.predict(x_test)

        scaler.scale_
        scale_factor = 1/scaler.scale_[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

    st.markdown("<h4>Model Performance</h4>", unsafe_allow_html=True)
    fig2 = plt.figure(figsize=(12,6))
    plt.plot(y_test, 'b', label='Actual Price', linewidth=2)
    plt.plot(y_predicted, 'r', label='Predicted Price', linewidth=2, alpha=0.7)
    plt.title('Actual vs Predicted Price', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Price ($)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig2)

    # Add some metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(y_test, y_predicted)
    rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Absolute Error", f"${mae:.2f}")
    with col2:
        st.metric("Root Mean Squared Error", f"${rmse:.2f}")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
    <div style='max-width: 800px; margin: 0 auto; text-align: left;'>
        <p style='font-size: 15px; margin-bottom: 0.5rem;'>
            <strong>Model Details:</strong> This app uses a <strong>Long Short-Term Memory (LSTM)</strong> neural network, 
            a type of deep learning model designed to analyze time-series data like stock prices.
        </p>
        <ul style='font-size: 14px; margin-top: 0.5rem; margin-bottom: 0.5rem; padding-left: 1.5rem;'>
            <li>The model is trained on past closing prices</li>
            <li>It tries to <strong>predict future stock prices</strong> based on learned patterns</li>
            <li>The chart shows the <strong>predicted trend</strong> vs actual historical data</li>
        </ul>
        <p style='font-size: 14px; color: #e74c3c; margin-top: 0.5rem; margin-bottom: 0.5rem;'>
            ⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only. The predictions are not guaranteed 
            and should not be used for real financial decisions.
        </p>
    </div>
    <div style='margin-top: 1rem; font-size: 14px;'>
        Stock Prediction App © 2025 | Powered by Streamlit, Keras, and Yahoo Finance
    </div>
</div>
""", unsafe_allow_html=True)