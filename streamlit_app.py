# --- Imports ---
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
DATA_FILE = "data/malaria_data_upd.csv"  # Changed to data directory
MODEL_ZIP = "Malaria_Forecasting.zip"
REGIONS = ["Juba", "Yei", "Wau"]
MODEL_TYPES = ["ARIMA", "Prophet", "NeuralProphet", "Exponential Smoothing"]

# Create necessary directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Data Preparation ---
@st.cache_data
def load_data():
    """Load and preprocess malaria data with environmental factors"""
    if not os.path.exists(DATA_FILE):
        # If no data file exists, create a minimal example
        example_data = {
            'Date': pd.date_range(start='2020-01-01', periods=365).strftime('%Y-%m-%d'),
            'Region': ['Juba']*365 + ['Yei']*365 + ['Wau']*365,
            'Cases': np.random.poisson(50, 365*3),
            'Temperature': np.random.normal(25, 5, 365*3),
            'Rainfall': np.random.gamma(2, 25, 365*3)
        }
        df = pd.DataFrame(example_data)
        df.to_csv(DATA_FILE, index=False)
        st.warning("No data file found. Created example data. Please upload real data.")
    
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
    """Train NeuralProphet with stability fixes"""
    df = data.reset_index()[['Date', 'Cases', 'Temperature', 'Rainfall']]
    df = df.rename(columns={'Date': 'ds', 'Cases': 'y'}).dropna()
    
    # NeuralProphet configuration with reduced complexity
    model = NeuralProphet(
        n_forecasts=1,
        n_lags=0,  # No autoregression
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        epochs=50,  # Reduced epochs for stability
        batch_size=16,
        learning_rate=0.01,
        trend_reg=0,
        trainer_config={
            'accelerator': 'cpu',
            'max_epochs': 50,
            'enable_progress_bar': True
        }
    )
    
    model.add_future_regressor('Temperature')
    model.add_future_regressor('Rainfall')
    
    # Train with progress feedback
    with st.spinner("Training NeuralProphet (this may take a minute)..."):
        try:
            metrics = model.fit(df, freq='D', progress='bar')
        except Exception as e:
            st.error(f"Training error: {str(e)}")
            return None
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

# --- Model Training Interface ---
def train_all_models():
    """Train and save all models for all regions"""
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return
    
    models = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        total_regions = len(REGIONS)
        for i, region in enumerate(REGIONS, 1):
            status_text.text(f"Training models for {region} ({i}/{total_regions})")
            data = prepare_region_data(df, region)
            
            progress = int((i-1) / total_regions * 100)
            progress_bar.progress(progress)
            
            # Train and save each model individually
            arima_model = train_arima(data)
            with open(f"{MODEL_DIR}/{region.lower()}_arima_model.pkl", 'wb') as f:
                pickle.dump(arima_model, f)
            
            prophet_model = train_prophet(data)
            with open(f"{MODEL_DIR}/{region.lower()}_prophet_model.json", 'w') as f:
                f.write(model_to_json(prophet_model))
            
            # Handle NeuralProphet training carefully
            neuralprophet_model = train_neuralprophet(data)
            if neuralprophet_model is not None:
                with open(f"{MODEL_DIR}/{region.lower()}_neuralprophet_model.pkl", 'wb') as f:
                    pickle.dump(neuralprophet_model, f)
            else:
                st.warning(f"NeuralProphet failed for {region}, skipping...")
            
            expsmooth_model = train_exponential_smoothing(data)
            with open(f"{MODEL_DIR}/{region.lower()}_expsmooth_model.pkl", 'wb') as f:
                pickle.dump(expsmooth_model, f)
        
        progress_bar.progress(100)
        status_text.success("All models trained and saved successfully!")
        st.balloons()
        
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
    finally:
        progress_bar.empty()

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
    """Robust NeuralProphet forecasting with updated API"""
    try:
        # Create future dates dataframe
        future = pd.DataFrame({
            'ds': pd.date_range(start=datetime.today(), periods=days, freq='D')
        })
        
        # Add required columns
        future['y'] = np.nan  # Dummy target column
        future['Temperature'] = temp
        future['Rainfall'] = rain
        
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
    st.title("🦟 Malaria Forecasting with Environmental Factors 🦟")
    
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
    with st.expander("⚙️ Model Training ⚙️", expanded=False):
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
        # Determine the model filename
        if model_type == "Exponential Smoothing":
            model_file = f"{MODEL_DIR}/{region.lower()}_expsmooth_model.pkl"
        elif model_type == "Prophet":
            model_file = f"{MODEL_DIR}/{region.lower()}_prophet_model.json"
        else:
            model_file = f"{MODEL_DIR}/{region.lower()}_{model_type.lower()}_model.pkl"
        
        # Check if model exists
        if not os.path.exists(model_file):
            st.error(f"Please train the {model_type} model first!")
            return
        
        try:
            # Load the appropriate model
            if model_type == "Prophet":
                with open(model_file, 'r') as f:
                    model = model_from_json(f.read())
            else:
                with open(model_file, 'rb') as f:
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
