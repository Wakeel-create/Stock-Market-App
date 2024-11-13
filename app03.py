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
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

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
fig = px.line(data, x='Date', y=data['Close'], title='Closing price of the stock', width=1000, height=600)
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
st.plotly_chart(px.line(x=data["Date"], y=decomposition.seasonal, title='Seasonality', width=1000, height=400,
labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='green'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.resid, title='Residuals', width=1000, height=400,
labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Red', line_dash='dot'))

# Model selection
models = ['SARIMA', 'Random Forest', 'LSTM', 'Prophet', 'XGBoost', 'ARIMA']
selected_model = st.sidebar.selectbox('Select the model for forecasting', models)

# **SARIMA Model**

if selected_model == 'SARIMA':
    # SARIMA Model
    # User input for SARIMA parameters
    p = st.slider('Select the value of p', 0, 5, 2)
    d = st.slider('Select the value of d', 0, 5, 1)
    q = st.slider('Select the value of q', 0, 5, 2)
    seasonal_order = st.number_input('Select the value of seasonal p', 0, 24, 12)

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

# Similar adjustments for other models like Random Forest, LSTM, Prophet, XGBoost, and ARIMA
# Make sure that for each plot, the y-axis is assigned a single column instead of `data.columns`
# If the error persists, I'll need to address that in other places where plotly is being used.
