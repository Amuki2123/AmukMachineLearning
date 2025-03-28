import os
import zipfile
import pickle
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
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
    """Train NeuralProphet with strict column validation"""
    df = data.reset_index()[['Date', 'Cases', 'Temperature', 'Rainfall']]
    df = df.rename(columns={'Date': 'ds', 'Cases': 'y'}).dropna()
    
    # Validate columns
    required_cols = ['ds', 'y', 'Temperature', 'Rainfall']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing columns. Required: {required_cols}")

    model = NeuralProphet(
        n_forecasts=1,
        n_lags=0,
        epochs=50,
        batch_size=16,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    
    model = model.add_future_regressor('Temperature')
    model = model.add_future_regressor('Rainfall')
    
    with st.spinner("Training NeuralProphet..."):
        model.fit(df, freq='D')
    return model

def train_exponential_smoothing(data):
    """Train Exponential Smoothing with robust seasonal handling"""
    try:
        return ExponentialSmoothing(
            data['Cases'],
            trend='add',
            seasonal=None,
        ).fit()
    except Exception as e:
        st.warning(f"Using simpler Exponential Smoothing: {str(e)}")
        return ExponentialSmoothing(
            data['Cases'],
            trend='add',
        ).fit()

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
    """Robust NeuralProphet forecasting with proper dataframe structure"""
    try:
        # Create future dates starting from tomorrow
        start_date = datetime.today() + timedelta(days=1)
        future_dates = pd.date_range(start=start_date, periods=days, freq='D')

        # Build future dataframe with ALL required columns
        future = pd.DataFrame({
            'ds': future_dates,
            'y': np.nan,  # Dummy y column (required but ignored)
            'Temperature': [temp] * days,
            'Rainfall': [rain] * days
        })

        # Generate forecast
        forecast = model.predict(future)
        return forecast['ds'].values, forecast['yhat1'].values

    except Exception as e:
        st.error(f"NeuralProphet prediction error: {str(e)}")
        return pd.date_range(datetime.today(), periods=days).values, np.zeros(days)

def forecast_expsmooth(model, days, temp, rain):
    """Generate Exponential Smoothing forecast"""
    forecast = model.forecast(days)
    return pd.date_range(datetime.today(), periods=days), forecast

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Malaria Forecasting", layout="wide")
    st.title("🦟 Malaria Cases Forecasting with Environmental Factors")
    
    # File Upload Section
    with st.expander("📤 Update Data File", expanded=False):
        st.write("Upload updated malaria data (CSV format with Date, Region, Cases, Temperature, Rainfall columns)")
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                required_columns = ['Date', 'Region', 'Cases', 'Temperature', 'Rainfall']
                if not all(col in df.columns for col in required_columns):
                    st.error(f"CSV file must contain these columns: {', '.join(required_columns)}")
                else:
                    df.to_csv(DATA_FILE, index=False)
                    st.success("✅ File uploaded and saved successfully!")
                    st.cache_data.clear()
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Model Training Section
    with st.expander("⚙️ Model Training", expanded=False):
        if st.button("Train All Models"):
            train_all_models()
    
    # Forecasting Interface
    st.header("Forecast Malaria Cases")
    col1, col2 = st.columns(2)
    
    with col1:
        region = st.selectbox("Select Region", REGIONS)
        model_type = st.selectbox("Select Model", MODEL_TYPES)
    
    with col2:
        temp = st.slider("Temperature (°C)", 15.0, 40.0, 25.0, 0.5)
        rain = st.slider("Rainfall (mm)", 0.0, 300.0, 50.0, 5.0)
        days = st.slider("Forecast Days", 7, 365, 30, 1)
    
    if st.button("Generate Forecast", type="primary"):
        if not os.path.exists(MODEL_ZIP):
            st.error("Please train models first!")
            return
        
        try:
            with zipfile.ZipFile(MODEL_ZIP, 'r') as zipf:
                # Correct model filename pattern
                if model_type == "Exponential Smoothing":
                    model_file = f"{region.lower()}_expsmooth_model.pkl"
                elif model_type == "Prophet":
                    model_file = f"{region.lower()}_prophet_model.json"
                else:
                    model_file = f"{region.lower()}_{model_type.lower()}_model.pkl"
                
                if model_type == "Prophet":
                    with zipf.open(model_file) as f:
                        model = model_from_json(f.read().decode('utf-8'))
                else:
                    with zipf.open(model_file) as f:
                        model = pickle.load(f)
            
            st.success(f"{model_type} model loaded for {region}!")
            
            with st.spinner("Generating forecast..."):
                if model_type == "ARIMA":
                    dates, values = forecast_arima(model, days, temp, rain)
                elif model_type == "Prophet":
                    dates, values = forecast_prophet(model, days, temp, rain)
                elif model_type == "NeuralProphet":
                    dates, values = forecast_neuralprophet(model, days, temp, rain)
                elif model_type == "Exponential Smoothing":
                    dates, values = forecast_expsmooth(model, days, temp, rain)
                
                # Handle different return types from models
                if isinstance(values, (pd.Series, np.ndarray)):
                    values = values.values if hasattr(values, 'values') else values
                elif isinstance(values, list):
                    values = np.array(values)
                
                forecast_df = pd.DataFrame({
                    'Date': pd.to_datetime(dates),
                    'Cases': np.round(values).astype(int),
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
