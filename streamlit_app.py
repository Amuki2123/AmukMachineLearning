import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import joblib
import json
import matplotlib.pyplot as plt
from prophet import Prophet
from neuralprophet import NeuralProphet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from keras.models import model_from_json
%matplotlib inline
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from neuralprophet import NeuralProphet, set_log_level
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_predict

# Title of the app
st.title("Regional Malaria Cases Forecasting App")
st.write("Forecast malaria cases for Juba, Yei, and Wau based on rainfall and temperature using various models.")

# Upload dataset
uploaded_file = st.file_uploader("https://github.com/Amuki2123/Amuk-MachineLearning/blob/master/Malaria Forecasting.zip/Malaria Forecasting/malaria_data_upd", type="csv")
if uploaded_file:
data = pd.read_csv(malaria_data_upd)
st.write(data)

# Preprocess uploaded data
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Load pre-trained models
models = {
    'Juba': {
        'ARIMA': pickle.load(open('juba_arima_model.pkl', 'rb')),
        'NeuralProphet': pickle.load(open('juba_np_model.pkl', 'rb')),
        'Prophet': Prophet().from_json(open('juba_prophet_model.json', 'r').read()),
        'Exponential Smoothing': pickle.load(open('juba_es_model.pkl', 'rb'))
    },
    'Yei': {
        'ARIMA': pickle.load(open('yei_arima_model.pkl', 'rb')),
        'NeuralProphet': pickle.load(open('yei_np_model.pkl', 'rb')),
        'Prophet': Prophet().from_json(open('yei_prophet_model.json', 'r').read()),
        'Exponential Smoothing': pickle.load(open('yei_es_model.pkl', 'rb'))
    },
    'Wau': {
        'ARIMA': pickle.load(open('wau_arima_model.pkl', 'rb')),
        'NeuralProphet': pickle.load(open('wau_np_model.pkl', 'rb')),
        'Prophet': Prophet().from_json(open('wau_prophet_model.json', 'r').read()),
        'Exponential Smoothing': pickle.load(open('wau_es_model.pkl', 'rb'))
    }
}

# Select region and model
region = st.selectbox("Select a region:", ['Juba', 'Yei', 'Wau'])
model_type = st.selectbox("Select a model:", ['ARIMA', 'NeuralProphet', 'Prophet', 'Exponential Smoothing'])

# Input daily rainfall and temperature
daily_rainfall = st.number_input("Enter daily rainfall (mm):", min_value=0, max_value=200, value=10)
daily_temp = st.number_input("Enter daily temperature (°C):", min_value=15, max_value=40, value=25)

# Forecast malaria cases
if st.button("Forecast Malaria Cases"):
    try:
        # Load the selected model
        model = models[region][model_type]

        # Prepare future DataFrame for prediction
        future_dates = pd.date_range('2023-01-01', periods=365)
        future_df = pd.DataFrame({
            'ds': future_dates,
            'daily_rainfall': [daily_rainfall] * 365,
            'daily_temp': [daily_temp] * 365
        })

        # Generate forecasts
        if model_type == "NeuralProphet":
            forecast = model.predict(future_df)
        elif model_type == "Prophet":
            forecast = model.predict(future_df)
        elif model_type == "Exponential Smoothing":
            forecast = model.forecast(365)
            future_df['yhat'] = forecast
        elif model_type == "ARIMA":
            forecast = model.forecast(steps=365)
            future_df['yhat'] = forecast

        # Calculate annual cases
        annual_cases = future_df['yhat'].sum()
        st.write(f"Forecasted annual malaria cases in {region}: {annual_cases:.2f}")

        # Plot forecast
        fig, ax = plt.subplots()
        ax.plot(future_df['ds'], future_df['yhat'], label='Forecasted Cases')
        ax.set_xlabel('Date')
        ax.set_ylabel('Malaria Cases')
        ax.set_title(f"Malaria Cases Forecast for {region} ({model_type} Model)")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error occurred during forecast: {e}")

# Option to download forecast as CSV
if 'future_df' in locals():
    csv = future_df.to_csv(index=False)
    st.download_button(label="Download Forecast as CSV",
                       data=csv,
                       file_name=f"{region}_{model_type}_forecast.csv",
                       mime="text/csv")

st.write('Hello world!')

