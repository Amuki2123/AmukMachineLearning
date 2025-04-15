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
        st.error(f"Error: Data file '{DATA_FILE}' not found.")
        st.info("Please upload the data file first.")
        return None
    
    try:
        df = pd.read_csv(DATA_FILE)
        if df.empty:
            st.error("Error: Loaded data is empty.")
            return None
            
        # Validate required columns
        required_cols = ['Date', 'Region', 'Cases', 'Temperature', 'Rainfall']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None
            
        # Convert and validate dates
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if df['Date'].isnull().any():
            st.error("Invalid date format found in data")
            return None
            
        st.success("Data loaded successfully!")
        return df
        
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return None

def prepare_region_data(df, region):
    """Prepare dataset for a specific region with thorough validation"""
    try:
        # Validate input
        if df is None or region not in REGIONS:
            return None
            
        # Filter and validate region data
        region_df = df[df['Region'] == region][['Date', 'Cases', 'Temperature', 'Rainfall']].copy()
        if region_df.empty:
            st.warning(f"No data found for region: {region}")
            return None
            
        # Create properly named columns
        region_df['ds'] = pd.to_datetime(region_df['Date'])
        region_df['y'] = region_df['Cases']
        
        # Validate we have the required temporal data
        if len(region_df['ds'].dropna()) < 10:  # Minimum 10 days of data
            st.warning(f"Insufficient temporal data for {region}")
            return None
            
        # Set datetime index for interpolation
        region_df = region_df.set_index('ds').sort_index()
        
        # Ensure continuous dates
        full_date_range = pd.date_range(
            start=region_df.index.min(),
            end=region_df.index.max(),
            freq='D'
        )
        region_df = region_df.reindex(full_date_range)
        
        # Handle missing values with validation
        for col in ['y', 'Temperature', 'Rainfall']:
            if region_df[col].isnull().any():
                if region_df[col].isnull().mean() > 0.5:  # More than 50% missing
                    st.warning(f"High percentage of missing values in {col} for {region}")
                    return None
                region_df[col] = region_df[col].interpolate(method='time').ffill().bfill()
        
        return region_df.reset_index()
        
    except Exception as e:
        st.error(f"Error preparing data for {region}: {str(e)}")
        return None

def check_data_quality(df, region):
    """Comprehensive data quality check with visualization"""
    try:
        if df is None:
            return False
            
        st.subheader(f"Data Quality Report for {region}")
        
        # Basic stats
        col1, col2 = st.columns(2)
        with col1:
            st.write("Missing Values:")
            missing = df.isnull().sum()
            st.write(missing)
            if missing.sum() > 0:
                st.warning("Missing values detected")
                
        with col2:
            st.write("Basic Statistics:")
            st.write(df.describe())
        
        # Visualization
        fig, ax = plt.subplots(3, 1, figsize=(12, 8))
        df['y'].plot(ax=ax[0], title='Cases')
        df['Temperature'].plot(ax=ax[1], title='Temperature')
        df['Rainfall'].plot(ax=ax[2], title='Rainfall')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Additional checks
        if len(df) < 30:
            st.warning("Less than 30 days of data - models may not train well")
            
        if df['y'].std() < 1:
            st.warning("Very low variance in cases data")
            
        return True
        
    except Exception as e:
        st.error(f"Data quality check failed: {str(e)}")
        return False

# --- Model Training Functions ---
def train_arima(data):
    """Train ARIMAX model with enhanced validation"""
    try:
        if data is None or 'y' not in data.columns:
            return None
            
        st.write("Training ARIMA model...")
        model = pm.auto_arima(
            data['y'],
            exogenous=data[['Temperature', 'Rainfall']],
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )
        st.success("ARIMA training completed!")
        return model
        
    except Exception as e:
        st.error(f"ARIMA training failed: {str(e)}")
        return None

def train_prophet(data):
    """Train Prophet model with thorough validation"""
    try:
        if data is None or 'ds' not in data.columns or 'y' not in data.columns:
            return None
            
        st.write("Training Prophet model...")
        
        # Prepare data specifically for Prophet
        prophet_df = data.rename(columns={'y': 'Cases'})[['ds', 'Cases', 'Temperature', 'Rainfall']]
        prophet_df = prophet_df.rename(columns={'Cases': 'y'})
        
        model = Prophet()
        model.add_regressor('Temperature')
        model.add_regressor('Rainfall')
        
        with st.spinner("Fitting Prophet model..."):
            model.fit(prophet_df)
            
        st.success("Prophet training completed!")
        return model
        
    except Exception as e:
        st.error(f"Prophet training failed: {str(e)}")
        return None

def train_neuralprophet(data, forecast_horizon=5):
    """Train NeuralProphet with comprehensive error handling"""
    set_log_level("ERROR")
    
    try:
        # Validate input data
        if data is None:
            return None
            
        required_cols = {'ds', 'y'}
        if not required_cols.issubset(data.columns):
            missing = required_cols - set(data.columns)
            st.error(f"Missing required columns: {missing}")
            return None
            
        # Ensure proper datetime format
        data['ds'] = pd.to_datetime(data['ds'])
        if data['ds'].isnull().any():
            st.error("Invalid dates found in data")
            return None
            
        # Final cleaning
        df = data.dropna(subset=['ds', 'y']).copy()
        if len(df) < 10:
            st.error("Insufficient data after cleaning")
            return None
            
        # Model configuration
        model = NeuralProphet(
            n_forecasts=forecast_horizon,
            n_lags=14,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            epochs=50,
            impute_missing=True,
            normalize="soft",
            trainer_config={'progress_bar': False}  # Disable internal progress bars
        )
        
        # Add regressors
        model.add_future_regressor('Temperature')
        model.add_future_regressor('Rainfall')
        
        # Train model
        with st.spinner("Training NeuralProphet (this may take a while)..."):
            metrics = model.fit(df, freq='D')
            
        st.success("NeuralProphet training completed!")
        return model
        
    except Exception as e:
        st.error(f"NeuralProphet training failed: {str(e)}")
        return None

def train_exponential_smoothing(data):
    """Train Exponential Smoothing with robust error handling"""
    try:
        if data is None or 'y' not in data.columns:
            return None
            
        st.write("Training Exponential Smoothing model...")
        
        model = ExponentialSmoothing(
            data['y'],
            trend='add',
            seasonal=None,
        ).fit()
        
        st.success("Exponential Smoothing training completed!")
        return model
        
    except Exception as e:
        st.warning(f"Using simpler Exponential Smoothing: {str(e)}")
        try:
            model = ExponentialSmoothing(
                data['y'],
                trend='add',
            ).fit()
            return model
        except Exception as e:
            st.error(f"Exponential Smoothing failed: {str(e)}")
            return None

# --- Model Training Interface ---
def train_all_models():
    """Complete model training workflow with comprehensive feedback"""
    try:
        # Load data with validation
        df = load_data()
        if df is None:
            st.error("Cannot proceed without valid data")
            return False
            
        models = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        training_errors = []
        trained_regions = set()
        
        # Initialize model tracking
        model_status = {
            region: {
                'ARIMA': False,
                'Prophet': False,
                'NeuralProphet': False,
                'ExponentialSmoothing': False
            } for region in REGIONS
        }
        
        # Train models for each region
        for i, region in enumerate(REGIONS, 1):
            region_status = st.empty()
            region_status.text(f"Processing {region} ({i}/{len(REGIONS)})")
            
            try:
                # Prepare data with validation
                data = prepare_region_data(df, region)
                if data is None:
                    training_errors.append(f"{region}: Data preparation failed")
                    continue
                    
                # Data quality check
                if not check_data_quality(data, region):
                    training_errors.append(f"{region}: Data quality check failed")
                    continue
                    
                trained_regions.add(region)
                
                # --- ARIMA Training ---
                try:
                    region_status.text(f"Training ARIMA for {region}...")
                    arima_model = train_arima(data)
                    if arima_model:
                        models[f"{region.lower()}_arima_model.pkl"] = arima_model
                        model_status[region]['ARIMA'] = True
                        st.success(f"ARIMA trained for {region}")
                    else:
                        training_errors.append(f"{region}: ARIMA training failed")
                except Exception as e:
                    training_errors.append(f"{region}: ARIMA error - {str(e)}")
                
                # --- Prophet Training ---
                try:
                    region_status.text(f"Training Prophet for {region}...")
                    prophet_model = train_prophet(data)
                    if prophet_model:
                        models[f"{region.lower()}_prophet_model.json"] = prophet_model
                        model_status[region]['Prophet'] = True
                        st.success(f"Prophet trained for {region}")
                    else:
                        training_errors.append(f"{region}: Prophet training failed")
                except Exception as e:
                    training_errors.append(f"{region}: Prophet error - {str(e)}")
                
                # --- NeuralProphet Training ---
                try:
                    region_status.text(f"Training NeuralProphet for {region}...")
                    neural_model = train_neuralprophet(data)
                    if neural_model:
                        models[f"{region.lower()}_neuralprophet_model.pkl"] = neural_model
                        model_status[region]['NeuralProphet'] = True
                        st.success(f"NeuralProphet trained for {region}")
                    else:
                        training_errors.append(f"{region}: NeuralProphet training failed")
                except Exception as e:
                    training_errors.append(f"{region}: NeuralProphet error - {str(e)}")
                
                # --- Exponential Smoothing Training ---
                try:
                    region_status.text(f"Training Exponential Smoothing for {region}...")
                    exp_model = train_exponential_smoothing(data)
                    if exp_model:
                        models[f"{region.lower()}_expsmooth_model.pkl"] = exp_model
                        model_status[region]['ExponentialSmoothing'] = True
                        st.success(f"Exponential Smoothing trained for {region}")
                    else:
                        training_errors.append(f"{region}: Exponential Smoothing training failed")
                except Exception as e:
                    training_errors.append(f"{region}: Exponential Smoothing error - {str(e)}")
                
                progress_bar.progress(int(i/len(REGIONS)*100))  # Fixed syntax error here
                
            except Exception as e:
                training_errors.append(f"{region}: Processing failed - {str(e)}")
                continue
        
        # Save models if any were trained successfully
        if models:
            try:
                with zipfile.ZipFile(MODEL_ZIP, 'w') as zipf:
                    for name, model in models.items():
                        try:
                            if name.endswith('.pkl'):
                                with zipf.open(name, 'w') as f:
                                    pickle.dump(model, f)
                            elif name.endswith('.json'):
                                with zipf.open(name, 'w') as f:
                                    f.write(model_to_json(model).encode('utf-8'))
                            st.success(f"Saved {name}")
                        except Exception as e:
                            training_errors.append(f"Failed to save {name}: {str(e)}")
                            continue
                
                # Training summary
                status_text.success(f"Training completed for {len(trained_regions)} regions!")
                
                # Show detailed model status
                with st.expander("Model Training Summary"):
                    st.subheader("Model Training Status by Region")
                    status_df = pd.DataFrame(model_status).T
                    st.dataframe(status_df.style.applymap(
                        lambda x: 'background-color: green' if x else 'background-color: red'
                    ))
                    
                    if training_errors:
                        st.subheader("Errors Encountered")
                        for error in training_errors:
                            st.error(error)
                
                return True
                
            except Exception as e:
                status_text.error(f"Failed to save models: {str(e)}")
                if os.path.exists(MODEL_ZIP):
                    os.remove(MODEL_ZIP)
                return False
        else:
            status_text.error("No models were trained successfully")
            if training_errors:
                st.error("Training errors encountered:")
                for error in training_errors:
                    st.write(f"- {error}")
            return False
            
    except Exception as e:
        status_text.error(f"Unexpected error during training: {str(e)}")
        return False
    finally:
        progress_bar.empty()

# --- Forecasting Functions ---
def forecast_arima(model, days, temp, rain):
    """Generate ARIMAX forecast with validation"""
    try:
        if model is None:
            st.warning("No ARIMA model available - returning zero forecast")
            return pd.date_range(datetime.today(), periods=days), np.zeros(days)
            
        future_exog = pd.DataFrame({
            'Temperature': [temp] * days,
            'Rainfall': [rain] * days
        })
        forecast = model.predict(n_periods=days, exogenous=future_exog)
        return pd.date_range(datetime.today(), periods=days), forecast
        
    except Exception as e:
        st.error(f"ARIMA forecast failed: {str(e)}")
        return pd.date_range(datetime.today(), periods=days), np.zeros(days)

def forecast_prophet(model, days, temp, rain):
    """Generate Prophet forecast with validation"""
    try:
        if model is None:
            st.warning("No Prophet model available - returning zero forecast")
            return pd.date_range(datetime.today(), periods=days), np.zeros(days)
            
        future = model.make_future_dataframe(periods=days)
        future['Temperature'] = temp
        future['Rainfall'] = rain
        forecast = model.predict(future)
        return future['ds'].iloc[-days:], forecast['yhat'].iloc[-days:]
        
    except Exception as e:
        st.error(f"Prophet forecast failed: {str(e)}")
        return pd.date_range(datetime.today(), periods=days), np.zeros(days)

def forecast_neuralprophet(model, days, temp, rain):
    """Generate NeuralProphet forecast with comprehensive error handling"""
    try:
        if model is None:
            st.warning("No NeuralProphet model available - returning zero forecast")
            return pd.date_range(datetime.today(), periods=days).values, np.zeros(days)
            
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
        st.error(f"NeuralProphet forecast failed: {str(e)}")
        return pd.date_range(datetime.today(), periods=days).values, np.zeros(days)

def forecast_expsmooth(model, days, temp, rain):
    """Generate Exponential Smoothing forecast with validation"""
    try:
        if model is None:
            st.warning("No Exponential Smoothing model available - returning zero forecast")
            return pd.date_range(datetime.today(), periods=days), np.zeros(days)
            
        forecast = model.forecast(days)
        return pd.date_range(datetime.today(), periods=days), forecast
        
    except Exception as e:
        st.error(f"Exponential Smoothing forecast failed: {str(e)}")
        return pd.date_range(datetime.today(), periods=days), np.zeros(days)

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
                    st.experimental_rerun()  # Refresh to use new data
                else:
                    st.error(f"Missing required columns: {set(required_cols)-set(df.columns)}")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Model Training Section
    st.header("Model Training")
    if st.button("Train All Models", key="train_button"):
        with st.spinner("Training models (this may take several minutes)..."):
            success = train_all_models()
            if success:
                st.balloons()
            else:
                st.error("Model training encountered errors")
    
    # Forecasting Interface
    st.header("Generate Forecasts")
    col1, col2 = st.columns(2)
    with col1:
        region = st.selectbox("Region", REGIONS, key="region_select")
        model_type = st.selectbox("Model", MODEL_TYPES, key="model_select")
    with col2:
        temp = st.slider("Temperature (°C)", 15.0, 40.0, 25.0, 0.5, key="temp_slider")
        rain = st.slider("Rainfall (mm)", 0.0, 300.0, 50.0, 5.0, key="rain_slider")
        days = st.slider("Forecast Days", 7, 365, 30, 1, key="days_slider")
    
    if st.button("Generate Forecast", type="primary", key="forecast_button"):
        if not os.path.exists(MODEL_ZIP):
            st.error("Please train models first")
            return
            
        try:
            with zipfile.ZipFile(MODEL_ZIP, 'r') as zipf:
                model_file = f"{region.lower()}_{model_type.lower().replace(' ', '_')}_model.{'json' if model_type == 'Prophet' else 'pkl'}"
                
                if model_file not in zipf.namelist():
                    st.error(f"Model not found: {model_file}")
                    st.warning("This model may have failed to train. Check the training logs.")
                    return
                
                with st.spinner(f"Loading {model_type} model..."):
                    with zipf.open(model_file) as f:
                        model = model_from_json(f.read().decode('utf-8')) if model_type == 'Prophet' else pickle.load(f)
                
                forecast_func = {
                    'ARIMA': forecast_arima,
                    'Prophet': forecast_prophet,
                    'NeuralProphet': forecast_neuralprophet,
                    'Exponential Smoothing': forecast_expsmooth
                }.get(model_type)
                
                if forecast_func is None:
                    st.error(f"No forecast function for model type: {model_type}")
                    return
                
                with st.spinner("Generating forecast..."):
                    dates, values = forecast_func(model, days, temp, rain)
                
                forecast_df = pd.DataFrame({
                    'Date': pd.to_datetime(dates),
                    'Cases': np.round(values).astype(int),
                    'Temperature': temp,
                    'Rainfall': rain
                })
                
                st.subheader(f"{region} {model_type} Forecast")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(forecast_df['Date'], forecast_df['Cases'])
                ax.set_xlabel("Date")
                ax.set_ylabel("Cases")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                st.download_button(
                    "📥 Download Forecast",
                    forecast_df.to_csv(index=False),
                    f"{region}_forecast.csv",
                    "text/csv"
                )
                
        except Exception as e:
            st.error(f"Forecasting failed: {str(e)}")

if __name__ == "__main__":
    main()
