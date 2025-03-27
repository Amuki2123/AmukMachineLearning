import os
import zipfile
import pickle
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, Union
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from neuralprophet import NeuralProphet
import pmdarima as pm
from prophet.serialize import model_to_json, model_from_json
import warnings
warnings.filterwarnings("ignore")

# --- Data Preparation ---
@st.cache_data
def load_data():
    """Load and preprocess malaria data with environmental factors"""
    df = pd.read_csv("malaria_data_upd.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_region_data(df, region):
    """Prepare dataset for a specific region"""
    region_df = df[df['Region'] == region].set_index('Date').sort_index()
    return region_df[['Cases', 'Temperature', 'Rainfall']]

# --- Model Training Functions ---
def train_arima(data):
    """Train ARIMAX model with environmental factors"""
    model = pm.auto_arima(
        data['Cases'],
        exogenous=data[['Temperature', 'Rainfall']],
        seasonal=False,
        stepwise=True,
        suppress_warnings=True
    )
    return model

def train_prophet(data):
    """Train Prophet model with regressors"""
    df = data.reset_index()
    df = df.rename(columns={'Date': 'ds', 'Cases': 'y'})
    model = Prophet()
    model.add_regressor('Temperature')
    model.add_regressor('Rainfall')
    model.fit(df)
    return model

def train_neuralprophet(data):
    """Train NeuralProphet with external regressors"""
    df = data.reset_index()
    df = df.rename(columns={'Date': 'ds', 'Cases': 'y'})
    model = NeuralProphet()
    model.add_future_regressor('Temperature')
    model.add_future_regressor('Rainfall')
    model.fit(df, freq='D')
    return model

def train_exponential_smoothing(data):
    """Train Exponential Smoothing with environmental factors"""
    model = ExponentialSmoothing(
        data['Cases'],
        exogenous=data[['Temperature', 'Rainfall']],
        trend='add',
        seasonal='add',
        seasonal_periods=365
    ).fit()
    return model

# --- Model Training Interface ---
def train_all_models():
    """Train and save all models for all regions"""
    df = load_data()
    regions = df['Region'].unique()
    models = {}
    
    with st.status("Training models...", expanded=True) as status:
        for region in regions:
            st.write(f"🚀 Training models for {region}")
            data = prepare_region_data(df, region)
            
            # ARIMA
            with st.spinner(f"Training ARIMAX for {region}"):
                models[f"{region.lower()}_arima_model.pkl"] = train_arima(data)
            
            # Prophet
            with st.spinner(f"Training Prophet for {region}"):
                prophet_model = train_prophet(data)
                models[f"{region.lower()}_prophet_model.json"] = prophet_model
            
            # NeuralProphet
            with st.spinner(f"Training NeuralProphet for {region}"):
                models[f"{region.lower()}_np_model.pkl"] = train_neuralprophet(data)
            
            # Exponential Smoothing
            with st.spinner(f"Training Exponential Smoothing for {region}"):
                models[f"{region.lower()}_es_model.pkl"] = train_exponential_smoothing(data)
        
        # Save models to ZIP
        with zipfile.ZipFile("Malaria Forecasting.zip", 'w') as zipf:
            for name, model in models.items():
                if name.endswith('.pkl'):
                    with zipf.open(name, 'w') as f:
                        pickle.dump(model, f)
                elif name.endswith('.json'):
                    with zipf.open(name, 'w') as f:
                        f.write(model_to_json(model).encode('utf-8'))
        
        status.update(label="✅ All models trained successfully!", state="complete")

# --- Forecasting Functions ---
def forecast_arima(model, days, temp, rain):
    """Generate ARIMAX forecast with environmental factors"""
    future_exog = pd.DataFrame({
        'Temperature': [temp] * days,
        'Rainfall': [rain] * days
    })
    forecast = model.predict(n_periods=days, exogenous=future_exog)
    return pd.date_range(datetime.today(), periods=days), forecast

def forecast_prophet(model, days, temp, rain):
    """Generate Prophet forecast with regressors"""
    future = model.make_future_dataframe(periods=days)
    future['Temperature'] = temp
    future['Rainfall'] = rain
    forecast = model.predict(future)
    return forecast['ds'].iloc[-days:], forecast['yhat'].iloc[-days:]

def forecast_neuralprophet(model, days, temp, rain):
    """Generate NeuralProphet forecast with regressors"""
    future = model.make_future_dataframe(periods=days)
    future['Temperature'] = temp
    future['Rainfall'] = rain
    forecast = model.predict(future)
    return forecast['ds'].iloc[-days:], forecast['yhat'].iloc[-days:]

def forecast_expsmooth(model, days, temp, rain):
    """Generate Exponential Smoothing forecast"""
    future_exog = pd.DataFrame({
        'Temperature': [temp] * days,
        'Rainfall': [rain] * days
    })
    forecast = model.forecast(days, exogenous=future_exog)
    return pd.date_range(datetime.today(), periods=days), forecast

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Malaria Forecasting", layout="wide")
    st.title("🦟 Malaria Cases Forecasting with Environmental Factors")
    
    # Model Training Section
    with st.expander("⚙️ Model Training", expanded=False):
        if st.button("Train All Models"):
            train_all_models()
    
    # Forecasting Interface
    st.header("Forecast Malaria Cases")
    
    col1, col2 = st.columns(2)
    with col1:
        region = st.selectbox("Select Region", ["Juba", "Yei", "Wau"])
        model_type = st.selectbox("Select Model", 
            ["ARIMA", "Prophet", "NeuralProphet", "Exponential Smoothing"])
    
    with col2:
        temp = st.slider("Temperature (°C)", 15.0, 40.0, 25.0, 0.5)
        rain = st.slider("Rainfall (mm)", 0.0, 300.0, 50.0, 5.0)
        days = st.slider("Forecast Days", 7, 365, 30, 1)
    
    if st.button("Generate Forecast", type="primary"):
        if not os.path.exists("Malaria Forecasting.zip"):
            st.error("Please train models first!")
            return
        
        with st.spinner(f"Loading {model_type} model..."):
            model = None
            try:
                with zipfile.ZipFile("Malaria Forecasting.zip", 'r') as zipf:
                    model_file = f"{region.lower()}_{model_type.lower().replace(' ', '')}"
                    if model_type == "Prophet":
                        model_file += ".json"
                        with zipf.open(model_file) as f:
                            model = model_from_json(f.read().decode('utf-8'))
                    else:
                        model_file += ".pkl"
                        with zipf.open(model_file) as f:
                            model = pickle.load(f)
            except Exception as e:
                st.error(f"Model loading failed: {str(e)}")
                return
        
        if model:
            st.success(f"✅ {model_type} model loaded!")
            
            with st.spinner("Generating forecast..."):
                try:
                    if model_type == "ARIMA":
                        dates, values = forecast_arima(model, days, temp, rain)
                    elif model_type == "Prophet":
                        dates, values = forecast_prophet(model, days, temp, rain)
                    elif model_type == "NeuralProphet":
                        dates, values = forecast_neuralprophet(model, days, temp, rain)
                    else:
                        dates, values = forecast_es(model, days, temp, rain)
                    
                    forecast_df = pd.DataFrame({
                        'Date': dates,
                        'Cases': values,
                        'Temperature': temp,
                        'Rainfall': rain
                    })
                    
                    # Visualization
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(forecast_df['Date'], forecast_df['Cases'], 'b-')
                    ax.set_title(
                        f"{region} {model_type} Forecast\n"
                        f"Temperature: {temp}°C, Rainfall: {rain}mm",
                        pad=20
                    )
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Cases")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Data Export
                    csv = forecast_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Forecast",
                        data=csv,
                        file_name=f"{region}_forecast.csv",
                        mime="text/csv"
                    )
                
                except Exception as e:
                    st.error(f"Forecast failed: {str(e)}")

if __name__ == "__main__":
    main()
