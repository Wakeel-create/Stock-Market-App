import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fbprophet import Prophet
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load Data
@st.cache
def load_data(file):
    data = pd.read_csv(file, parse_dates=['Date'])
    data.sort_values('Date', inplace=True)
    return data

# Feature Engineering
def feature_engineering(data):
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['Date_ordinal'] = data['Date'].map(datetime.datetime.toordinal)
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
    data['Day_sin'] = np.sin(2 * np.pi * data['Day'] / 31)
    data['Day_cos'] = np.cos(2 * np.pi * data['Day'] / 31)
    return data

# Train-Test Split
def split_data(data, column):
    X = data[['Date_ordinal', 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos']]
    y = data[column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Forecasting Models

def train_xgboost(X_train, y_train, X_test):
    model = XGBRegressor(n_estimators=500, learning_rate=0.1)
    model.fit(X_train, y_train)
    return model.predict(X_test)

def train_catboost(X_train, y_train, X_test):
    model = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, loss_function='RMSE', verbose=100)
    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    return model.predict(X_test)

def train_random_forest(X_train, y_train, X_test):
    model = RandomForestRegressor(n_estimators=500, random_state=42)
    model.fit(X_train, y_train)
    return model.predict(X_test)

def train_prophet(data, column):
    df = data[['Date', column]].rename(columns={'Date': 'ds', column: 'y'})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']]

def train_arima(data, column):
    model = ARIMA(data[column], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    return forecast

def train_sarima(data, column):
    model = SARIMAX(data[column], order=(1,1,1), seasonal_order=(1,1,1,12))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    return forecast

# Streamlit App
st.title("Stock Market Forecasting")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    data = load_data(uploaded_file)
    data = feature_engineering(data)
    st.write("Data Preview:", data.head())
    
    column = st.selectbox("Select column to forecast", data.columns[1:])
    model_choice = st.selectbox("Select Forecasting Model", ["XGBoost", "CatBoost", "Random Forest", "Prophet", "ARIMA", "SARIMA"])
    
    if st.button("Run Forecasting"):
        X_train, X_test, y_train, y_test = split_data(data, column)
        
        if model_choice == "XGBoost":
            preds = train_xgboost(X_train, y_train, X_test)
        elif model_choice == "CatBoost":
            preds = train_catboost(X_train, y_train, X_test)
        elif model_choice == "Random Forest":
            preds = train_random_forest(X_train, y_train, X_test)
        elif model_choice == "Prophet":
            preds = train_prophet(data, column)
        elif model_choice == "ARIMA":
            preds = train_arima(data, column)
        elif model_choice == "SARIMA":
            preds = train_sarima(data, column)
        
        st.write("Forecasting Completed!")
        if model_choice in ["XGBoost", "CatBoost", "Random Forest"]:
            st.write("Mean Absolute Error:", mean_absolute_error(y_test, preds))
            st.write("Mean Squared Error:", mean_squared_error(y_test, preds))
            plt.figure(figsize=(10,5))
            plt.plot(y_test.values, label='Actual')
            plt.plot(preds, label='Predicted')
            plt.legend()
            st.pyplot(plt)
        else:
            st.write(preds)
