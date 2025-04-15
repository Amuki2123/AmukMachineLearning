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
    """Prepare dataset for a specific region with continuous dates"""
    region_df = df[df['Region'] == region].set_index('Date').sort_index()
    
    # Ensure continuous daily data
    full_date_range = pd.date_range(
        start=region_df.index.min(), 
        end=region_df.index.max(), 
        freq='D'
    )
    region_df = region_df.reindex(full_date_range)
    
    # Interpolate missing values
    region_df['Cases'] = region_df['Cases'].interpolate(method='time').ffill().bfill()
    region_df['Temperature'] = region_df['Temperature'].interpolate(method='linear').ffill().bfill()
    region_df['Rainfall'] = region_df['Rainfall'].interpolate(method='linear').ffill().bfill()
    
    return region_df[['Cases', 'Temperature', 'Rainfall']]

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
    df['Cases'].plot(ax=ax[0], title='Cases')
    df['Temperature'].plot(ax=ax[1], title='Temperature')
    df['Rainfall'].plot(ax=ax[2], title='Rainfall')
    plt.tight_layout()
    st.pyplot(fig)
    
    return not df.isnull().values.any()

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

def train_neuralprophet(data, forecast_horizon=5):
    """Train NeuralProphet with enhanced missing value handling"""
    # Reduce logging verbosity
    set_log_level("ERROR")
    
    # Prepare dataframe with required column names
    df = data.reset_index()[['Date', 'Cases', 'Temperature', 'Rainfall']]
    df = df.rename(columns={
        'Date': 'ds', 
        'Cases': 'y',
        'Temperature': 'Temperature',
        'Rainfall': 'Rainfall'
    })
    
    # Ensure we have the required columns
    if 'ds' not in df.columns or 'y' not in df.columns:
        st.error("Dataframe must contain 'ds' (dates) and 'y' (values) columns")
        st.error(f"Current columns: {df.columns.tolist()}")
        return None
    
    # Enhanced missing value handling
    df = df.interpolate(method='time').ffill().bfill()
    
    # Verify no missing values remain
    if df.isnull().values.any():
        st.warning("Warning: Some missing values remain after imputation")
        st.write("Missing values per column:")
        st.write(df.isnull().sum())
        df = df.dropna()  # Final fallback
        
    # NeuralProphet configuration
    model = NeuralProphet(
        n_forecasts=forecast_horizon,
        n_lags=14,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        epochs=50,
        batch_size=16,
        learning_rate=0.01,
        trend_reg=0.5,
        impute_missing=True,
        drop_missing=True,
        normalize="soft",
        trainer_config={
            'accelerator': 'cpu',
            'max_epochs': 50,
            'enable_progress_bar': True
        }
    )
    
    # Add regressors
    model.add_future_regressor('Temperature', normalize=True)
    model.add_future_regressor('Rainfall', normalize=True)
    
    with st.spinner(f"Training NeuralProphet for {forecast_horizon} forecasts..."):
        try:
            # Print debug info before training
            st.write("Training data sample:")
            st.write(df.head())
            st.write(f"Data shape: {df.shape}")
            
            metrics = model.fit(df, freq='D', progress='bar')
            return model
        except Exception as e:
            st.error(f"Training error: {str(e)}")
            st.error(f"Data columns: {df.columns.tolist()}")
            st.error(f"Missing values:\n{df.isnull().sum()}")
            st.error(f"First 5 rows:\n{df.head()}")
            return None

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
            
            # Data quality check
            st.subheader(f"Data Quality Report for {region}")
            if not check_data_quality(data, region):
                st.warning(f"Skipping {region} due to data quality issues")
                continue
            
            progress = int((i-1) / total_regions * 100)
            progress_bar.progress(progress)
            
            models[f"{region.lower()}_arima_model.pkl"] = train_arima(data)
            models[f"{region.lower()}_prophet_model.json"] = train_prophet(data)
            
            neuralprophet_model = train_neuralprophet(data, forecast_horizon=5)
            if neuralprophet_model is not None:
                models[f"{region.lower()}_neuralprophet_model.pkl"] = neuralprophet_model
            else:
                st.warning(f"NeuralProphet failed for {region}, skipping...")
            
            models[f"{region.lower()}_expsmooth_model.pkl"] = train_exponential_smoothing(data)
        
        with zipfile.ZipFile(MODEL_ZIP, 'w') as zipf:
            for name, model in models.items():
                if name.endswith('.pkl'):
                    with zipf.open(name, 'w') as f:
                        pickle.dump(model, f)
                elif name.endswith('.json'):
                    with zipf.open(name, 'w') as f:
                        f.write(model_to_json(model).encode('utf-8'))
        
        progress_bar.progress(100)
        if all(f"{r.lower()}_neuralprophet_model.pkl" in models for r in REGIONS):
            status_text.success("All models trained successfully!")
            st.balloons()
        else:
            status_text.warning("Models trained with some NeuralProphet failures")
        
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
