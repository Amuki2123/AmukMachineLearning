import asyncio
import torch
import streamlit as st
import zipfile
import pickle
import json
import os
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import model_from_json
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from keras.models import model_from_json
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
st.title("Regional Malaria Cases Forecasting Models")
st.write("Forecast malaria cases for Juba, Yei, and Wau based on rainfall and temperature using various models.")

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write(data)

def load_regional_model(zip_path: str, region: str, model_type: str) -> Optional[object]:
    """
    Load a region-specific model from a ZIP archive.
    
    Args:
        zip_path: Path to the ZIP file (e.g., "Malaria_Forecasting.zip")
        region: Target region (e.g., "juba", "yei", "wau")
        model_type: Type of model (e.g., "arima", "prophet", "neural")
    
    Returns:
        Loaded model object or None if failed
    """
    # Map model types to their file patterns
    model_patterns = {
        "arima": f"{region}_arima_model.pkl",
        "prophet": f"{region}_prophet_model.json",
        "neural": f"{region}_np_model.pkl"
    }
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Find matching file in ZIP
            model_file = next(
                (f for f in zip_ref.namelist() 
                 if f.lower() == model_patterns[model_type].lower()),
                None
            )
            
            if not model_file:
                available = [f for f in zip_ref.namelist() if region in f.lower()]
                st.error(f"Model not found. Available {region} models: {available}")
                return None
            
            # Extract to memory and load
            with zip_ref.open(model_file) as f:
                if model_file.endswith('.pkl'):
                    return pickle.load(f)
                elif model_file.endswith('.json'):
                    from keras.models import model_from_json
                    return model_from_json(f.read().decode('utf-8'))
                
    except Exception as e:
        st.error(f"Error loading {region} {model_type} model: {str(e)}")
        return None

def make_forecast(model, region: str, steps: int = 365) -> pd.DataFrame:
    """Generate forecasts from a loaded model."""
    try:
        if hasattr(model, 'forecast'):  # ARIMA-style models
            forecast = model.forecast(steps=steps)
            return pd.DataFrame({
                'date': pd.date_range(start=pd.Timestamp.today(), periods=steps),
                'cases': forecast
            })
        elif hasattr(model, 'predict'):  # Prophet-style models
            future = pd.DataFrame({
                'ds': pd.date_range(start=pd.Timestamp.today(), periods=steps)
            })
            return model.predict(future)
    except Exception as e:
        st.error(f"Forecast failed: {str(e)}")
        return pd.DataFrame()

# Streamlit UI
st.title("Regional Malaria Forecast")

# User inputs
region = st.selectbox("Select Region", ["juba", "yei", "wau"])
model_type = st.selectbox("Select Model Type", ["arima", "prophet", "neural"])
forecast_days = st.slider("Forecast Days", 30, 365, 90)

if st.button("Run Forecast"):
    with st.spinner(f"Loading {region} {model_type} model..."):
        model = load_regional_model(
            zip_path="Malaria_Forecasting.zip",
            region=region,
            model_type=model_type
        )
    
    if model:
        st.success(f"✅ {model_type.upper()} model loaded for {region.capitalize()}!")
        forecast = make_forecast(model, region, forecast_days)
        
        if not forecast.empty:
            st.line_chart(forecast.set_index('date' if 'date' in forecast.columns else 'ds'))
            st.download_button(
                "Download Forecast",
                data=forecast.to_csv(index=False),
                file_name=f"{region}_{model_type}_forecast.csv"
            )
    st.download_button(label="Download Forecast as CSV",
                       data=csv,
                       file_name=f"{region}_{model_type}_forecast.csv",
                       mime="text/csv")

