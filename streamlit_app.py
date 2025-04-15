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
    """Prepare dataset for a specific region with proper column names for NeuralProphet"""
    region_df = df[df['Region'] == region][['Date', 'Cases', 'Temperature', 'Rainfall']]
    region_df = region_df.rename(columns={'Date': 'ds', 'Cases': 'y'})  # Critical rename
    
    # Ensure continuous dates
    full_date_range = pd.date_range(
        start=region_df['ds'].min(),
        end=region_df['ds'].max(),
        freq='D'
    )
    region_df = region_df.set_index('ds').reindex(full_date_range).reset_index()
    
    # Handle missing values
    for col in ['y', 'Temperature', 'Rainfall']:
        region_df[col] = region_df[col].interpolate(method='time').ffill().bfill()
    
    return region_df

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
    model = Prophet()
    model.add_regressor('Temperature')
    model.add_regressor('Rainfall')
    model.fit(data)
    return model

def train_neuralprophet(data, forecast_horizon=5):
    """Train NeuralProphet with strict column requirements"""
    set_log_level("ERROR")
    
    # Verify required columns
    REQUIRED_COLS = {'ds', 'y'}
    if not REQUIRED_COLS.issubset(data.columns):
        missing = REQUIRED_COLS - set(data.columns)
        st.error(f"Missing required columns: {missing}")
        st.error(f"Current columns: {list(data.columns)}")
        return None
    
    # Final cleaning
    df = data.copy().dropna(subset=['ds', 'y'])
    
    # Model config
    model = NeuralProphet(
        n_forecasts=forecast_horizon,
        n_lags=14,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        epochs=50,
        impute_missing=True
    )
    
    # Add regressors
    model.add_future_regressor('Temperature')
    model.add_future_regressor('Rainfall')
    
    # Train
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
    """Train Exponential Smoothing"""
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
            
            # Prophet needs renamed columns too
            prophet_data = data.rename(columns={'y': 'Cases'})  # Prophet expects 'y'
            models[f"{region.lower()}_prophet_model.json"] = train_prophet(prophet_data)
            
            neural_model = train_neuralprophet(data)
            if neural_model:
                models[f"{region.lower()}_neuralprophet_model.pkl"] = neural_model
            
            models[f"{region.lower()}_expsmooth_model.pkl"] = train_exponential_smoothing(data)
            
            progress_bar.progress(int(i/len(REGIONS)*100)
        
        # Save models
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
    """Generate NeuralProphet forecast with proper future regressors"""
    try:
        # Create future dataframe with proper regressors
        future = model.make_future_dataframe(
            periods=days,
            regressors_df=pd.DataFrame({
                'ds': pd.date_range(start=datetime.today(), periods=days),
                'Temperature': [temp] * days,
                'Rainfall': [rain] * days
            })
        )
        
        forecast = model.predict(future)
        
        # Handle potential missing forecasts
        forecast = forecast.ffill().bfill()
        
        forecast_dates = forecast['ds'].values[-days:]
        forecast_values = forecast['yhat1'].values[-days:]
        
        return forecast_dates, forecast_values
        
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
    st.title("🦟 Malaria Cases Forecasting with Environmental Factors 🦟")
    
    # File Upload Section
    st.header("Data Management")
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
    st.header("Model Training")
    with st.expander("⚙️ Model Training Options", expanded=False):
        if st.button("Train All Models"):
            train_all_models()
    
    # Forecasting Interface
    st.header("Generate Forecasts")
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
                if model_type == "Exponential Smoothing":
                    model_file = f"{region.lower()}_expsmooth_model.pkl"
                elif model_type == "Prophet":
                    model_file = f"{region.lower()}_prophet_model.json"
                else:
                    model_file = f"{region.lower()}_{model_type.lower()}_model.pkl"
                
                if model_type == "NeuralProphet" and f"{region.lower()}_neuralprophet_model.pkl" not in zipf.namelist():
                    st.error("NeuralProphet model not available for this region (training failed)")
                    return
                
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
                
                forecast_df = pd.DataFrame({
                    'Date': pd.to_datetime(dates),
                    'Cases': np.round(values).astype(int),
                    'Temperature': temp,
                    'Rainfall': rain
                })
                
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
