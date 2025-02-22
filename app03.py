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
from statsmodels.tsa.arima.model import ARIMA

# Page configuration (MUST BE FIRST)
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Title section
app_name = 'Stock Market Forecasting App'
st.title(app_name)
st.subheader('This app is created to forecast the stock market price of the selected company.')
st.image("https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg")

# Sidebar inputs
with st.sidebar:
    st.header('Parameters')
    start_date = st.date_input('Start date', date(2020, 1, 1))
    end_date = st.date_input('End date', date(2023, 12, 31))
    ticker_list = ["AAPL", "MSFT", "GOOG", "GOOGL", "META", "TSLA", "NVDA", 
                  "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"]
    ticker = st.selectbox('Select company', ticker_list)
    models = ['SARIMA', 'ARIMA', 'Random Forest', 'LSTM', 'Prophet', 'XGBoost', 'CatBoost']
    selected_model = st.selectbox('Select model', models)

# Data loading with MultiIndex fix
@st.cache_data
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        
        # Fix MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.map('_'.join).str.strip('_')
        else:
            data.columns = [col.replace(' ', '_') for col in data.columns]
            
        data.insert(0, "Date", data.index)
        return data.reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

data = load_data(ticker, start_date, end_date)

if not data.empty:
    # Display data
    st.write(f'Data from {start_date} to {end_date}')
    st.write(data)

    # Visualization with safe column selection
    st.header('Price History')
    
    # Get valid plot columns (numeric columns except Date)
    plot_columns = [col for col in data.columns 
                   if col != 'Date' 
                   and pd.api.types.is_numeric_dtype(data[col])]
    
    if plot_columns:
        fig = px.line(data, 
                     x='Date', 
                     y=plot_columns,
                     title='Price History',
                     labels={'value': 'Price', 'variable': 'Metric'},
                     width=1000,
                     height=600)
        st.plotly_chart(fig)
    else:
        st.warning("No numeric columns found for plotting")

    # Column selection from valid numeric columns
    column = st.selectbox('Select column for forecasting', plot_columns)
    data = data[['Date', column]]

    # ADF Test
    st.header('Stationarity Check')
    result = adfuller(data[column])
    st.write(f'ADF p-value: {result[1]:.4f} ({"" if result[1] < 0.05 else "not "}stationary)')

    # Decomposition
    st.header('Time Series Decomposition')
    try:
        decomposition = seasonal_decompose(data[column], model='additive', period=365)
        fig = px.line(x=data['Date'], y=decomposition.trend, title='Trend Component')
        st.plotly_chart(fig)
    except ValueError as e:
        st.error(f"Decomposition error: {str(e)}")

    # Model implementations
    st.header(f'{selected_model} Forecasting')

    # SARIMA Model
    if selected_model == 'SARIMA':
        col1, col2 = st.columns(2)
        with col1:
            p = st.slider('p', 0, 5, 1)
            d = st.slider('d', 0, 5, 1)
            q = st.slider('q', 0, 5, 1)
        with col2:
            P = st.slider('Seasonal P', 0, 5, 1)
            D = st.slider('Seasonal D', 0, 5, 1)
            Q = st.slider('Seasonal Q', 0, 5, 1)
            s = st.slider('Seasonal Period', 1, 365, 365)
        
        model = sm.tsa.statespace.SARIMAX(data[column],
                                        order=(p, d, q),
                                        seasonal_order=(P, D, Q, s))
        results = model.fit()
        
        forecast_steps = st.number_input('Forecast days', 1, 365, 30)
        forecast = results.get_forecast(steps=forecast_steps)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data[column], name='Historical'))
        fig.add_trace(go.Scatter(x=forecast.predicted_mean.index, 
                                y=forecast.predicted_mean, 
                                name='Forecast'))
        st.plotly_chart(fig)

    # ARIMA Model
    elif selected_model == 'ARIMA':
        col1, col2, col3 = st.columns(3)
        with col1: p = st.slider('p', 0, 5, 1)
        with col2: d = st.slider('d', 0, 5, 1)
        with col3: q = st.slider('q', 0, 5, 1)
        
        model = ARIMA(data[column], order=(p, d, q))
        results = model.fit()
        
        forecast_steps = st.number_input('Forecast days', 1, 365, 30)
        forecast = results.forecast(steps=forecast_steps)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data[column], name='Historical'))
        fig.add_trace(go.Scatter(x=pd.date_range(start=data['Date'].iloc[-1], 
                                              periods=forecast_steps+1, 
                                              freq='D')[1:], 
                               y=forecast, 
                               name='Forecast'))
        st.plotly_chart(fig)

    # Random Forest Model
    elif selected_model == 'Random Forest':
        data['Date_ordinal'] = data['Date'].map(pd.Timestamp.toordinal)
        
        train_size = int(len(data) * 0.8)
        train, test = data.iloc[:train_size], data.iloc[train_size:]
        
        model = RandomForestRegressor(n_estimators=100)
        model.fit(train[['Date_ordinal']], train[column])
        
        forecast_days = st.number_input('Forecast days', 1, 365, 30)
        future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=forecast_days, freq='D')
        future_ordinal = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        predictions = model.predict(future_ordinal)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data[column], name='Historical'))
        fig.add_trace(go.Scatter(x=future_dates, y=predictions, name='Forecast'))
        st.plotly_chart(fig)

    # LSTM Model
    elif selected_model == 'LSTM':
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[[column]])
        
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data)-seq_length):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length])
            return np.array(X), np.array(y)
        
        seq_length = st.slider('Sequence length', 30, 180, 60)
        X, y = create_sequences(scaled_data, seq_length)
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        
        forecast_days = st.number_input('Forecast days', 1, 365, 30)
        forecast = []
        current_batch = scaled_data[-seq_length:]
        
        for _ in range(forecast_days):
            pred = model.predict(current_batch.reshape(1, seq_length, 1))
            forecast.append(pred[0][0])
            current_batch = np.append(current_batch[1:], pred)
        
        forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data[column], name='Historical'))
        fig.add_trace(go.Scatter(x=pd.date_range(start=data['Date'].iloc[-1], 
                               periods=forecast_days, 
                               freq='D'), 
                     y=forecast.flatten(), 
                     name='Forecast'))
        st.plotly_chart(fig)

    # Prophet Model
    elif selected_model == 'Prophet':
        prophet_data = data.rename(columns={'Date': 'ds', column: 'y'})
        
        model = Prophet()
        model.fit(prophet_data)
        
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)
        
        fig = model.plot(forecast)
        st.pyplot(fig)

    # XGBoost Model
    elif selected_model == 'XGBoost':
        data['year'] = data['Date'].dt.year
        data['month'] = data['Date'].dt.month
        data['day'] = data['Date'].dt.day
        
        train_size = int(len(data) * 0.8)
        train, test = data.iloc[:train_size], data.iloc[train_size:]
        
        model = xgb.XGBRegressor(n_estimators=1000)
        model.fit(train[['year', 'month', 'day']], train[column])
        
        future_days = st.number_input('Forecast days', 1, 365, 30)
        future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=future_days, freq='D')
        future_df = pd.DataFrame({
            'year': future_dates.year,
            'month': future_dates.month,
            'day': future_dates.day
        })
        
        predictions = model.predict(future_df)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data[column], name='Historical'))
        fig.add_trace(go.Scatter(x=future_dates, y=predictions, name='Forecast'))
        st.plotly_chart(fig)

    # CatBoost Model
    elif selected_model == 'CatBoost':
        data['year'] = data['Date'].dt.year
        data['month'] = data['Date'].dt.month
        data['day'] = data['Date'].dt.day
        
        train_size = int(len(data) * 0.8)
        train, test = data.iloc[:train_size], data.iloc[train_size:]
        
        model = CatBoostRegressor(verbose=0)
        model.fit(train[['year', 'month', 'day']], train[column])
        
        future_days = st.number_input('Forecast days', 1, 365, 30)
        future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=future_days, freq='D')
        future_df = pd.DataFrame({
            'year': future_dates.year,
            'month': future_dates.month,
            'day': future_dates.day
        })
        
        predictions = model.predict(future_df)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data[column], name='Historical'))
        fig.add_trace(go.Scatter(x=future_dates, y=predictions, name='Forecast'))
        st.plotly_chart(fig)

else:
    st.warning("No data loaded. Please check your inputs and internet connection.")

# Force rendering of all elements
st.experimental_show()
