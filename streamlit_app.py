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
DATA_FILE = os.path.join(os.getcwd(), "malaria_data_upd.csv")
MODEL_ZIP = os.path.join(os.getcwd(), "Malaria_Forecasting.zip")
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
    """Train NeuralProphet with stability fixes"""
    df = data.reset_index()[['Date', 'Cases', 'Temperature', 'Rainfall']]
    df = df.rename(columns={'Date': 'ds', 'Cases': 'y'}).dropna()
    
    # Optimized configuration for Streamlit Cloud
    model = NeuralProphet(
        n_forecasts=1,
        n_lags=0,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        epochs=20,  # Reduced from 50 for cloud stability
        batch_size=8,
        learning_rate=0.01,
        trend_reg=0,
        trainer_config={
            'accelerator': 'cpu',
            'max_epochs': 20,
            'enable_progress_bar': False  # Disable for cloud
        }
    )
    
    model.add_future_regressor('Temperature')
    model.add_future_regressor('Rainfall')
    
    with st.spinner("Training NeuralProphet (this may take a minute)..."):
        try:
            metrics = model.fit(df, freq='D')
            return model
        except Exception as e:
            st.error(f"NeuralProphet training failed: {str(e)}")
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
            
            progress = int((i-1) / total_regions * 100)
            progress_bar.progress(progress)
            
            models[f"{region.lower()}_arima_model.pkl"] = train_arima(data)
            models[f"{region.lower()}_prophet_model.json"] = train_prophet(data)
            
            neuralprophet_model = train_neuralprophet(data)
            if neuralprophet_model is not None:
                models[f"{region.lower()}_neuralprophet_model.pkl"] = neuralprophet_model
            
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
        status_text.success("Models trained successfully!" if all(
            f"{r.lower()}_neuralprophet_model.pkl" in models for r in REGIONS
        ) else "Models trained with some NeuralProphet failures")
        
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
    """Robust NeuralProphet forecasting"""
    try:
        future = pd.DataFrame({
            'ds': pd.date_range(start=datetime.today(), periods=days, freq='D'),
            'y': np.nan,
            'Temperature': temp,
            'Rainfall': rain
        })
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
    st.title("🦟 Malaria Cases Forecasting with Environmental Factors")
    
    # File Upload Section
    with st.expander("📤 Update Data File", expanded=False):
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                required_cols = ['Date', 'Region', 'Cases', 'Temperature', 'Rainfall']
                if all(col in df.columns for col in required_cols):
                    df.to_csv(DATA_FILE, index=False)
                    st.success("Data uploaded successfully!")
                    st.cache_data.clear()
                else:
                    st.error(f"Missing columns. Required: {', '.join(required_cols)}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Model Training
    with st.expander("⚙️ Model Training"):
        if st.button("Train Models", help="May take several minutes"):
            train_all_models()
    
    # Forecasting UI
    st.header("Generate Forecast")
    col1, col2 = st.columns(2)
    
    with col1:
        region = st.selectbox("Region", REGIONS)
        model_type = st.selectbox("Model", MODEL_TYPES)
    
    with col2:
        temp = st.slider("Temperature (°C)", 15.0, 40.0, 25.0, 0.5)
        rain = st.slider("Rainfall (mm)", 0.0, 300.0, 50.0, 5.0)
        days = st.slider("Forecast Days", 7, 365, 30)
    
    if st.button("Generate Forecast", type="primary"):
        if not os.path.exists(MODEL_ZIP):
            st.error("Train models first!")
            return
        
        try:
            with zipfile.ZipFile(MODEL_ZIP, 'r') as zipf:
                model_file = f"{region.lower()}_{model_type.lower().replace(' ', '')}_model.{'json' if model_type == 'Prophet' else 'pkl'}"
                
                if model_file not in zipf.namelist():
                    st.error(f"{model_type} model not available for {region}")
                    return
                
                if model_type == "Prophet":
                    with zipf.open(model_file) as f:
                        model = model_from_json(f.read().decode('utf-8'))
                else:
                    with zipf.open(model_file) as f:
                        model = pickle.load(f)
            
            dates, values = globals()[f"forecast_{model_type.lower().replace(' ', '')}"](model, days, temp, rain)
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(dates, values, 'b-')
            ax.set_title(f"{region} {model_type} Forecast\n(Temp: {temp}°C, Rain: {rain}mm)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Data Export
            forecast_df = pd.DataFrame({
                'Date': pd.to_datetime(dates),
                'Cases': np.round(values).astype(int),
                'Temperature': temp,
                'Rainfall': rain
            })
            st.download_button(
                "📥 Download Forecast",
                data=forecast_df.to_csv(index=False),
                file_name=f"{region}_forecast.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Forecast failed: {str(e)}")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Malaria Forecasting",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    main()
