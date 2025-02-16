# Import libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

# setting the side bar to collapsed taa k footer jo ha wo sahi dikhay
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Title
app_name = 'Stock Market Forecasting App'
st.title(app_name)
st.subheader('This app is created to forecast the stock market price of the selected company.')
# Add an image from an online resource
st.image("https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg")

# Take input from the user of the app about the start and end date

# Sidebar
st.sidebar.header('Select the parameters from below')

start_date = st.sidebar.date_input('Start date', date(2020, 1, 1))
end_date = st.sidebar.date_input('End date', date(2020, 12, 31))
# Add ticker symbol list
ticker_list = ["AAPL", "MSFT", "GOOG", "GOOGL", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"]
ticker = st.sidebar.selectbox('Select the company', ticker_list)

# Fetch data from user inputs using yfinance library
data = yf.download(ticker, start=start_date, end=end_date)
# Add Date as a column to the dataframe
data.insert(0, "Date", data.index, True)
data.reset_index(drop=True, inplace=True)
st.write('Data from', start_date, 'to', end_date)
st.write(data)

# Plot the data
st.header('Data Visualization')
st.subheader('Plot of the data')
st.write("**Note:** Select your specific date range on the sidebar, or zoom in on the plot and select your specific column")
fig = px.line(data, x='Date', y=data.columns, title='Closing price of the stock', width=1000, height=600)
st.plotly_chart(fig)

# Add a select box to choose the column for forecasting
column = st.selectbox('Select the column to be used for forecasting', data.columns[1:])

# Subsetting the data
data = data[['Date', column]]
st.write("Selected Data")
st.write(data)

# ADF test to check stationarity
st.header('Is data Stationary?')
st.write(adfuller(data[column])[1] < 0.05)

# Decompose the data
st.header('Decomposition of the data')
decomposition = seasonal_decompose(data[column], model='additive', period=12)
st.write(decomposition.plot())
# Make same plot in Plotly
st.write("## Plotting the decomposition in Plotly")
st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend, title='Trend', width=1000, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Blue'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.seasonal, title='Seasonality', width=1000, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='green'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.resid, title='Residuals', width=1000, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Red', line_dash='dot'))

# Model selection
models = ['SARIMA', 'ARIMA', 'Random Forest', 'LSTM', 'Prophet', 'XGBoost', 'CatBoost']
selected_model = st.sidebar.selectbox('Select the model for forecasting', models)

# SARIMA Model
if selected_model == 'SARIMA':
    # User input for SARIMA parameters
    p = st.slider('Select the value of p', 0, 5, 2)
    d = st.slider('Select the value of d', 0, 5, 1)
    q = st.slider('Select the value of q', 0, 5, 2)

    # Dynamically set the seasonal period based on the data frequency (252 trading days for daily data)
    freq = pd.infer_freq(data['Date'])
    seasonal_period = 252 if freq == 'D' else 12

    seasonal_order = st.slider('Select the value of seasonal p', 0, 24, seasonal_period)

    model = sm.tsa.statespace.SARIMAX(data[column], order=(p, d, q), seasonal_order=(p, d, q, seasonal_order))
    model = model.fit()

    # Print model summary
    st.header('Model Summary')
    st.write(model.summary())
    st.write("---")

    # Forecasting using SARIMA
    st.write("<p style='color:green; font-size: 50px; font-weight: bold;'>Forecasting the data with SARIMA</p>",
             unsafe_allow_html=True)

    forecast_period = st.number_input('Select the number of days to forecast', 1, 365, 10)
    # Predict the future values
    predictions = model.get_prediction(start=len(data), end=len(data) + forecast_period)
    predictions = predictions.predicted_mean
    # Add index to the predictions
    predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
    predictions = pd.DataFrame(predictions)
    predictions.insert(0, "Date", predictions.index, True)
    predictions.reset_index(drop=True, inplace=True)
    st.write("Predictions", predictions)
    st.write("Actual Data", data)
    st.write("---")

    # Plot the data
    fig = go.Figure()
    # Add actual data to the plot
    fig.add_trace(go.Scatter(x=data["Date"], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
    # Add predicted data to the plot
    fig.add_trace(
        go.Scatter(x=predictions["Date"], y=predictions["predicted_mean"], mode='lines', name='Predicted',
                   line=dict(color='red')))
    # Set the title and axis labels
    fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=1000, height=400)
    # Display the plot
    st.plotly_chart(fig)

# ARIMA Model
elif selected_model == 'ARIMA':
    # ARIMA Model
    st.header('AutoRegressive Integrated Moving Average (ARIMA)')

    # User input for ARIMA parameters
    p = st.slider('Select the value of p', 0, 5, 1)
    d = st.slider('Select the value of d', 0, 5, 1)
    q = st.slider('Select the value of q', 0, 5, 1)

    # Fit the ARIMA model
    model = sm.tsa.ARIMA(data[column], order=(p, d, q))
    model = model.fit()

    # Print model summary
    st.write(model.summary())

    # Forecasting the future values
    forecast_period = st.number_input('Select the number of days to forecast', 1, 365, 10)
    forecast = model.forecast(steps=forecast_period)
    forecast_dates = pd.date_range(start=end_date, periods=forecast_period, freq='D')

    # Create a DataFrame with the forecast data
    forecast_df = pd.DataFrame(forecast, columns=[column])
    forecast_df.insert(0, "Date", forecast_dates)

    st.write("Forecast with ARIMA:")
    st.write(forecast_df)

    # Plot the data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df[column], mode='lines', name='Predicted', line=dict(color='red')))
    fig.update_layout(title='Actual vs Predicted (ARIMA)', xaxis_title='Date', yaxis_title='Price', width=1000, height=400)
    st.plotly_chart(fig)

# Random Forest Model
elif selected_model == 'Random Forest':
    st.header('Random Forest Regression')

    # Splitting data into training and testing sets
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    # Feature engineering
    train_X, train_y = train_data['Date'], train_data[column]
    test_X, test_y = test_data['Date'], test_data[column]

    # Initialize RandomForestRegressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(train_X, train_y)

    # Forecast the future
    rf_predictions = rf_model.predict(test_X)

    # Plot the data
    st.write("Random Forest Model Forecast")
    st.write(rf_predictions)

    # Plot results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_data["Date"], y=train_data[column], mode='lines', name='Train Data'))
    fig.add_trace(go.Scatter(x=test_data["Date"], y=rf_predictions, mode='lines', name='Predictions'))
    st.plotly_chart(fig)

# LSTM Model
elif selected_model == 'LSTM':
    st.header('LSTM Neural Network')
    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[column].values.reshape(-1, 1))

    # Prepare training and testing data
    train_size = int(len(data) * 0.8)
    train_data, test_data = data_scaled[:train_size], data_scaled[train_size:]

    # Create LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(train_data.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(train_data, epochs=10, batch_size=32)

    # Make predictions
    predictions = model.predict(test_data)
    st.write(predictions)

    # Plot results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data[column], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=data["Date"], y=predictions, mode='lines', name='Predictions'))
    st.plotly_chart(fig)

# Prophet Model
elif selected_model == 'Prophet':
    st.header('Facebook Prophet Model')
    prophet_data = data.rename(columns={'Date': 'ds', column: 'y'})
    model = Prophet()
    model.fit(prophet_data)

    # Forecasting
    future = model.make_future_dataframe(prophet_data, periods=365)
    forecast = model.predict(future)

    # Plot the forecast
    fig = model.plot(forecast)
    st.plotly_chart(fig)

# XGBoost Model
elif selected_model == 'XGBoost':
    st.header('XGBoost Model')

    # Feature engineering for XGBoost
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day

    # Adding cyclical features
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
    data['Day_sin'] = np.sin(2 * np.pi * data['Day'] / 31)
    data['Day_cos'] = np.cos(2 * np.pi * data['Day'] / 31)

    # Prepare training data
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    X_train = train_data[['Year', 'Month', 'Day', 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos']]
    y_train = train_data[column]
    X_test = test_data[['Year', 'Month', 'Day', 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos']]
    y_test = test_data[column]

    # Initialize XGBoost model
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=1000)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    st.write(predictions)

    # Plot the results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_data['Date'], y=y_test, mode='lines', name='True Data'))
    fig.add_trace(go.Scatter(x=test_data['Date'], y=predictions, mode='lines', name='Predictions'))
    st.plotly_chart(fig)

# CatBoost Model
elif selected_model == 'CatBoost':
    st.header('CatBoost Model')

    # Feature engineering for CatBoost
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day

    # Adding cyclical features
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
    data['Day_sin'] = np.sin(2 * np.pi * data['Day'] / 31)
    data['Day_cos'] = np.cos(2 * np.pi * data['Day'] / 31)

    # Prepare training data
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    X_train = train_data[['Year', 'Month', 'Day', 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos']]
    y_train = train_data[column]
    X_test = test_data[['Year', 'Month', 'Day', 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos']]
    y_test = test_data[column]

    # Initialize CatBoost model
    model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, loss_function='RMSE')
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    st.write(predictions)

    # Plot the results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_data['Date'], y=y_test, mode='lines', name='True Data'))
    fig.add_trace(go.Scatter(x=test_data['Date'], y=predictions, mode='lines', name='Predictions'))
    st.plotly_chart(fig)
