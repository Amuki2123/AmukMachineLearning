import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"  # Add this line
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
from neuralprophet import NeuralProphet, set_log_level
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
    """Prepare dataset for a specific region with proper datetime handling"""
    # Filter and rename columns
    region_df = df[df['Region'] == region][['Date', 'Cases', 'Temperature', 'Rainfall']].copy()
    region_df = region_df.rename(columns={'Date': 'ds', 'Cases': 'y'})
    
    # Convert to datetime and set as index
    region_df['ds'] = pd.to_datetime(region_df['ds'])
    region_df = region_df.set_index('ds').sort_index()
    
    # Ensure continuous dates
    full_date_range = pd.date_range(
        start=region_df.index.min(),
        end=region_df.index.max(),
        freq='D'
    )
    region_df = region_df.reindex(full_date_range)
    
    # Handle missing values with time-based interpolation
    for col in ['y', 'Temperature', 'Rainfall']:
        if region_df[col].isnull().any():
            region_df[col] = region_df[col].interpolate(method='time').ffill().bfill()
    
    return region_df.reset_index()  # Return with ds as a column

def check_data_quality(df, region):
    """Verify data quality before training"""
    st.subheader(f"Data Quality Check for {region}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Missing Values:")
        st.write(df.isnull().sum())
        
    with col2:
        st.write("Basic Statistics:")
        st.write(df.describe())
    
    fig, ax = plt.subplots(3, 1, figsize=(12, 8))
    df['y'].plot(ax=ax[0], title='Cases')
    df['Temperature'].plot(ax=ax[1], title='Temperature')
    df['Rainfall'].plot(ax=ax[2], title='Rainfall')
    plt.tight_layout()
    st.pyplot(fig)
    
    return not df.isnull().values.any()

# --- Model Training Functions ---
def train_arima(data):
    """Train ARIMAX model with environmental factors"""
    return pm.auto_arima(
        data['y'],
        exogenous=data[['Temperature', 'Rainfall']],
        seasonal=False,
        stepwise=True,
        suppress_warnings=True
    )

def train_prophet(data):
    """Train Prophet model with regressors"""
    df = data.rename(columns={'y': 'Cases'})  # Prophet expects different column names
    df = df[['ds', 'Cases', 'Temperature', 'Rainfall']]
    model = Prophet()
    model.add_regressor('Temperature')
    model.add_regressor('Rainfall')
    model.fit(df)
    return model

def train_neuralprophet(data, forecast_horizon=5):
    """Train NeuralProphet with proper datetime handling"""
    set_log_level("ERROR")
    
    # Ensure proper datetime format
    data['ds'] = pd.to_datetime(data['ds'])
    
    # Verify required columns
    REQUIRED_COLS = {'ds', 'y'}
    if not REQUIRED_COLS.issubset(data.columns):
        missing = REQUIRED_COLS - set(data.columns)
        st.error(f"Missing required columns: {missing}")
        st.error(f"Current columns: {list(data.columns)}")
        return None
    
    # Final cleaning
    df = data.dropna(subset=['ds', 'y']).copy()
    
    # Model configuration
    model = NeuralProphet(
        n_forecasts=forecast_horizon,
        n_lags=14,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        epochs=50,
        impute_missing=True,
        normalize="soft"
    )
    
    # Add regressors
    model.add_future_regressor('Temperature')
    model.add_future_regressor('Rainfall')
    
    # Train model
    with st.spinner("Training NeuralProphet..."):
        try:
            metrics = model.fit(df, freq='D')
            return model
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            st.error("Problematic data sample:")
            st.write(df.head())
            return None

def train_exponential_smoothing(data):
    """Train Exponential Smoothing model"""
    try:
        return ExponentialSmoothing(
            data['y'],
            trend='add',
            seasonal=None,
        ).fit()
    except Exception as e:
        st.warning(f"Using simpler model: {str(e)}")
        return ExponentialSmoothing(
            data['y'],
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
        for i, region in enumerate(REGIONS, 1):
            status_text.text(f"Training models for {region} ({i}/{len(REGIONS)})")
            data = prepare_region_data(df, region)
            
            # Data quality check
            st.subheader(f"Data Quality Report for {region}")
            if not check_data_quality(data, region):
                st.warning(f"Skipping {region} due to data quality issues")
                continue
            
            # Train models
            models[f"{region.lower()}_arima_model.pkl"] = train_arima(data)
            models[f"{region.lower()}_prophet_model.json"] = train_prophet(data)
            
            neural_model = train_neuralprophet(data)
            if neural_model:
                models[f"{region.lower()}_neuralprophet_model.pkl"] = neural_model
            
            models[f"{region.lower()}_expsmooth_model.pkl"] = train_exponential_smoothing(data)
            
            progress_bar.progress(int(i/len(REGIONS)*100))
        
        # Save models to zip file
        with zipfile.ZipFile(MODEL_ZIP, 'w') as zipf:
            for name, model in models.items():
                if name.endswith('.pkl'):
                    with zipf.open(name, 'w') as f:
                        pickle.dump(model, f)
                elif name.endswith('.json'):
                    with zipf.open(name, 'w') as f:
                        f.write(model_to_json(model).encode('utf-8'))
        
        status_text.success("All models trained successfully!")
        st.balloons()
    
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
    finally:
        progress_bar.empty()

# --- Forecasting Functions ---
def forecast_arima(model, days, temp, rain):
    """Generate ARIMAX forecast"""
    future_exog = pd.DataFrame({
        'Temperature': [temp] * days,
        'Rainfall': [rain] * days
    })
    forecast = model.predict(n_periods=days, exogenous=future_exog)
    return pd.date_range(datetime.today(), periods=days), forecast

def forecast_prophet(model, days, temp, rain):
    """Generate Prophet forecast"""
    future = model.make_future_dataframe(periods=days)
    future['Temperature'] = temp
    future['Rainfall'] = rain
    forecast = model.predict(future)
    return forecast['ds'].iloc[-days:], forecast['yhat'].iloc[-days:]

def forecast_neuralprophet(model, days, temp, rain):
    """Generate NeuralProphet forecast"""
    try:
        future = model.make_future_dataframe(
            periods=days,
            regressors_df=pd.DataFrame({
                'ds': pd.date_range(start=datetime.today(), periods=days),
                'Temperature': [temp] * days,
                'Rainfall': [rain] * days
            })
        )
        forecast = model.predict(future)
        return forecast['ds'].values[-days:], forecast['yhat1'].values[-days:]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return pd.date_range(datetime.today(), periods=days).values, np.zeros(days)

def forecast_expsmooth(model, days, temp, rain):
    """Generate Exponential Smoothing forecast"""
    forecast = model.forecast(days)
    return pd.date_range(datetime.today(), periods=days), forecast

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Malaria Forecasting", layout="wide")
    st.title("🦟 Malaria Cases Forecasting with Environmental Factors 🦟")
    
    # File Upload Section
    st.header("Data Management")
    with st.expander("📤 Update Data File", expanded=False):
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                required_cols = ['Date', 'Region', 'Cases', 'Temperature', 'Rainfall']
                if all(col in df.columns for col in required_cols):
                    df.to_csv(DATA_FILE, index=False)
                    st.success("File uploaded successfully!")
                    st.cache_data.clear()
                else:
                    st.error(f"Missing required columns: {set(required_cols)-set(df.columns)}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Model Training
    st.header("Model Training")
    if st.button("Train All Models"):
        train_all_models()
    
    # Forecasting Interface
    st.header("Generate Forecasts")
    col1, col2 = st.columns(2)
    with col1:
        region = st.selectbox("Region", REGIONS)
        model_type = st.selectbox("Model", MODEL_TYPES)
    with col2:
        temp = st.slider("Temperature (°C)", 15.0, 40.0, 25.0)
        rain = st.slider("Rainfall (mm)", 0.0, 300.0, 50.0)
        days = st.slider("Forecast Days", 7, 365, 30)
    
    if st.button("Generate Forecast"):
        if not os.path.exists(MODEL_ZIP):
            st.error("Please train models first")
            return
        
        try:
            with zipfile.ZipFile(MODEL_ZIP, 'r') as zipf:
                model_file = f"{region.lower()}_{model_type.lower().replace(' ', '_')}_model.{'json' if model_type == 'Prophet' else 'pkl'}"
                
                if model_file not in zipf.namelist():
                    st.error(f"Model not found: {model_file}")
                    return
                
                with zipf.open(model_file) as f:
                    model = model_from_json(f.read().decode('utf-8')) if model_type == 'Prophet' else pickle.load(f)
                
                dates, values = {
                    'ARIMA': forecast_arima,
                    'Prophet': forecast_prophet,
                    'NeuralProphet': forecast_neuralprophet,
                    'Exponential Smoothing': forecast_expsmooth
                }[model_type](model, days, temp, rain)
                
                forecast_df = pd.DataFrame({
                    'Date': pd.to_datetime(dates),
                    'Cases': np.round(values).astype(int),
                    'Temperature': temp,
                    'Rainfall': rain
                })
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(forecast_df['Date'], forecast_df['Cases'])
                ax.set_title(f"{region} {model_type} Forecast")
                st.pyplot(fig)
                
                st.download_button(
                    "Download Forecast",
                    forecast_df.to_csv(index=False),
                    f"{region}_forecast.csv",
                    "text/csv"
                )
                
        except Exception as e:
            st.error(f"Forecasting failed: {str(e)}")

if __name__ == "__main__":
    main()
