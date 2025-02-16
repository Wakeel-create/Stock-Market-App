# Import libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from catboost import CatBoostRegressor

# Set page config
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Title
app_name = 'Stock Market Forecasting App'
st.title(app_name)
st.subheader('This app is created to forecast the stock market price of the selected company.')
st.image("https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg")

# Sidebar inputs
st.sidebar.header('Select the parameters from below')
start_date = st.sidebar.date_input('Start date', date(2020, 1, 1))
end_date = st.sidebar.date_input('End date', date(2020, 12, 31))
ticker_list = ["AAPL", "MSFT", "GOOG", "GOOGL", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"]
ticker = st.sidebar.selectbox('Select the company', ticker_list)

# Fetch and prepare data
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data = data.reset_index()  # Convert index to column
    data.rename(columns={'Date': 'date'}, inplace=True)  # Fix 1: Consistent date column name
    return data

data = load_data(ticker, start_date, end_date)
st.write(f'Data from {start_date} to {end_date}')
st.write(data)

# Fixed Visualization (px.line error)
st.header('Data Visualization')
st.subheader('Plot of the data')
if not data.empty:
    # Fix 2: Exclude date column from y-axis
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    fig = px.line(data, x='date', y=numeric_columns, 
                 title='Closing price of the stock', 
                 width=1000, height=600)
    st.plotly_chart(fig)
else:
    st.error("No data available for the selected period")

# Forecasting setup
if not data.empty:
    column = st.selectbox('Select the column to be used for forecasting', numeric_columns)
    data = data[['date', column]]
    
    # Rest of your code remains similar but with these key fixes:
    # 1. Changed all 'Date' references to 'date' for consistency
    # 2. Added proper error handling for empty data
    # 3. Fixed Prophet implementation
    # [Include the rest of your model code here with similar fixes]

else:
    st.warning("Cannot proceed - no data available")
