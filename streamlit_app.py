import os
import zipfile
import pickle
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from neuralprophet import NeuralProphet
import pmdarima as pm
from prophet.serialize import model_to_json, model_from_json
import warnings
warnings.filterwarnings("ignore")

# --- Constants ---
DATA_FILE = "malaria_data_upd.csv"
MODEL_ZIP = "Malaria_Forecasting.zip"
REGIONS = ["Juba", "Yei", "Wau"]
MODEL_TYPES = ["ARIMA", "Prophet", "NeuralProphet", "Exponential Smoothing"]

# --- Data Preparation ---
@st.cache_data
def load_data():
    """Load and preprocess malaria data with environmental factors"""
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"The data file '{DATA_FILE}' is missing. Please upload it first.")
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_region_data(df, region):
    """Prepare dataset for a specific region"""
    region_df = df[df['Region'] == region].set_index('Date').sort_index()
    return region_df[['Cases', 'Temperature', 'Rainfall']]

# --- Model Training Functions ---
def train_arima(data):
    """Train ARIMAX model with environmental factors"""
    return pm.auto_arima(
        data['Cases'],
        exogenous=data[['Temperature', 'Rainfall']],
        seasonal=False,
        stepwise=True,
        suppress_warnings=True
    )

def train_prophet(data):
    """Train Prophet model with regressors"""
    df = data.reset_index().rename(columns={'Date': 'ds', 'Cases': 'y'})
    model = Prophet()
    model.add_regressor('Temperature')
    model.add_regressor('Rainfall')
    model.fit(df)
    return model

def train_neuralprophet(data):
    """Train NeuralProphet with external regressors"""
    df = data.reset_index().rename(columns={'Date': 'ds', 'Cases': 'y'})
    model = NeuralProphet()
    model.add_future_regressor('Temperature')
    model.add_future_regressor('Rainfall')
    model.fit(df, freq='D')
    return model

def train_exponential_smoothing(data):
    """Train Exponential Smoothing with automatic seasonal period adjustment"""
    # Calculate available seasonal periods
    days_of_data = (data.index[-1] - data.index[0]).days
    seasonal_periods = min(365, days_of_data//2)  # Use half the available days or 365
    
    # Ensure we have at least 2 periods worth of data
    if days_of_data < seasonal_periods * 2:
        return ExponentialSmoothing(
            data['Cases'],
            trend='add',
            seasonal=None  # Disable seasonal component if insufficient data
        ).fit()
    
    return ExponentialSmoothing(
        data['Cases'],
        trend='add',
        seasonal='add',
        seasonal_periods=seasonal_periods
    ).fit()

# --- [Rest of the code remains exactly the same as in your original version] ---
# ... (all other functions including train_all_models, forecasting functions, etc.)
# ... (main() function and Streamlit interface remain identical)

if __name__ == "__main__":
    main()
