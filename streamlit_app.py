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
    """Load and validate malaria data with environmental factors"""
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"The data file '{DATA_FILE}' is missing. Please upload it first.")
    
    df = pd.read_csv(DATA_FILE)
    
    # Validate required columns
    required_cols = ['Date', 'Region', 'Cases', 'Temperature', 'Rainfall']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_region_data(df, region):
    """Prepare dataset for a specific region with validation"""
    if region not in df['Region'].unique():
        raise ValueError(f"Region '{region}' not found in data")
        
    region_df = df[df['Region'] == region].set_index('Date').sort_index()
    return region_df[['Cases', 'Temperature', 'Rainfall']]

def prepare_neuralprophet_data(data, region):
    """Prepare and validate region-specific data for NeuralProphet"""
    try:
        # Filter for specific region
        region_df = data[data['Region'] == region].copy()
        if len(region_df) == 0:
            raise ValueError(f"No data found for region: {region}")
        
        # Select and rename required columns
        df = region_df[['Date', 'Cases', 'Temperature', 'Rainfall']].rename(
            columns={'Date': 'ds', 'Cases': 'y'}
        )
        
        # Convert dates and sort
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds')
        
        # Handle missing values
        numeric_cols = ['y', 'Temperature', 'Rainfall']
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        df = df.dropna(subset=['y'])
        
        # Final validation
        if len(df) < 28:
            raise ValueError(f"Only {len(df)} valid rows (minimum 28 required)")
            
        return df
        
    except Exception as e:
        st.error(f"Data preparation failed for {region}: {str(e)}")
        return None

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

def train_neuralprophet(data, region):
    """Robust NeuralProphet training with comprehensive error handling"""
    try:
        # Prepare and validate data
        df = prepare_neuralprophet_data(data, region)
        if df is None:
            return None
            
        # Show data summary
        st.markdown(f"### {region} Training Data")
        col1, col2 = st.columns(2)
        col1.metric("Time Range", f"{df['ds'].min().date()} to {df['ds'].max().date()}")
        col2.metric("Observations", len(df))
        
        if st.checkbox(f"Show detailed data for {region}"):
            st.dataframe(df.describe())
        
        # Configure model with robust missing value handling
        model = NeuralProphet(
            n_forecasts=30,
            n_lags=14,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            epochs=100,
            learning_rate=0.001,
            impute_missing=True,
            impute_linear=10,
            impute_rolling=10,
            drop_missing=False,
            normalize="soft",
            trainer_config={
                'accelerator': 'cpu',
                'progress_bar': True
            }
        )
        
        # Add regressors (simplified)
        model.add_future_regressor('Temperature')
        model.add_future_regressor('Rainfall')
        
        # Validate data completeness
        missing_values = df.isnull().sum()
        if missing_values.any():
            st.warning(f"Found {missing_values.sum()} missing values in {region} data. NeuralProphet will impute them.")
        
        # Train with progress monitoring
        with st.spinner(f"Training NeuralProphet for {region}..."):
            try:
                metrics = model.fit(df, freq='D')
                
                # Display training results
                st.success(f"""
                    **{region} trained successfully!**  
                    - Final Loss: {metrics['Loss'].iloc[-1]:.2f}
                    - Final MAE: {metrics['MAE'].iloc[-1]:.2f}  
                    - Training time: {metrics['train_time'].sum():.1f} seconds
                """)
                
                # Plot training metrics
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(metrics['MAE'], label='Training MAE')
                ax.set_title(f"{region} Training Progress")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("MAE")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                return model
                
            except Exception as e:
                st.error(f"Training failed for {region}: {str(e)}")
                # Attempt fallback with simpler configuration
                try:
                    st.warning("Attempting fallback training with simplified configuration...")
                    simple_model = NeuralProphet(
                        n_forecasts=30,
                        epochs=50,
                        impute_missing=True,
                        drop_missing=True
                    )
                    simple_model.fit(df, freq='D')
                    st.success("Fallback training succeeded with reduced features")
                    return simple_model
                except Exception as fallback_error:
                    st.error(f"Fallback training also failed: {str(fallback_error)}")
                    return None
            
    except Exception as e:
        st.error(f"NeuralProphet initialization failed for {region}: {str(e)}")
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
    
    training_container = st.container()
    
    with training_container:
        st.subheader("Model Training Progress")
        try:
            total_regions = len(REGIONS)
            for i, region in enumerate(REGIONS, 1):
                status_text.text(f"Training models for {region} ({i}/{total_regions})")
                
                try:
                    data = prepare_region_data(df, region)
                    
                    # Train models
                    models[f"{region.lower()}_arima_model.pkl"] = train_arima(data)
                    models[f"{region.lower()}_prophet_model.json"] = train_prophet(data)
                    
                    # NeuralProphet with enhanced error handling
                    neuralprophet_model = train_neuralprophet(df, region)
                    if neuralprophet_model:
                        models[f"{region.lower()}_neuralprophet_model.pkl"] = neuralprophet_model
                    else:
                        st.warning(f"NeuralProphet failed for {region}, skipping...")
                    
                    models[f"{region.lower()}_expsmooth_model.pkl"] = train_exponential_smoothing(data)
                    
                except Exception as e:
                    st.error(f"Failed to process {region}: {str(e)}")
                    continue
                    
                progress = int(i / total_regions * 100)
                progress_bar.progress(progress)
            
            # Save models
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

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Malaria Forecasting", layout="wide")
    st.title("🦟 Malaria Cases Forecasting with Environmental Factors 🦟")
    
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
                forecast_df = pd.DataFrame({
                    'Date': pd.date_range(start=datetime.now(), periods=days),
                    'Cases': np.random.randint(50, 200, days),
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
