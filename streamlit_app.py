import os
import zipfile
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import os
from typing import Optional, Union 
from datetime import datetime, timedelta
from io import BytesIO
import matplotlib.pyplot as plt
from prophet import Prophet
from neuralprophet import NeuralProphet
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


# --- App Configuration ---
st.set_page_config(
    page_title="Malaria Forecasting System",
    page_icon="🦟",
    layout="wide"
)

# --- Model Loading Function ---
def load_regional_model(
    zip_path: str, 
    region: str, 
    model_type: str
) -> Optional[Union[ARIMA, Prophet, NeuralProphet]]:
    """
    Load a forecasting model from ZIP archive
    Supports ARIMA, Prophet, and NeuralProphet models
    """
    model_files = {
        "arima": f"{region}_arima_model.pkl",
        "prophet": f"{region}_prophet_model.json", 
        "neural": f"{region}_np_model.pkl"
    }
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Find matching file (case-insensitive)
            target_file = model_files[model_type]
            matched = next(
                (f for f in zip_ref.namelist() 
                 if target_file.lower() in f.lower()),
                None
            )
            
            if not matched:
                available = [f for f in zip_ref.namelist() 
                           if region.lower() in f.lower()]
                st.error(f"Model not found. Available models: {available}")
                return None
                
            # Load based on file type
            with zip_ref.open(matched) as f:
                if matched.endswith('.pkl'):
                    model = pickle.load(f)
                    # Validate NeuralProphet models
                    if model_type == "neural" and not hasattr(model, "predict"):
                        raise ValueError("Invalid NeuralProphet model")
                    return model
                elif matched.endswith('.json'):
                    from keras.models import model_from_json
                    return model_from_json(f.read().decode('utf-8'))
    
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None

# --- Forecasting Function ---
def generate_forecast(
    model: Union[ARIMA, Prophet, NeuralProphet],
    days: int,
    temp: float,
    rainfall: float
) -> Optional[pd.DataFrame]:
    """Generate forecasts with environmental factors"""
    try:
        forecast_dates = pd.date_range(datetime.today(), periods=days)
        
        # Handle ARIMA models
        if isinstance(model, ARIMA):
            forecast_values = model.forecast(steps=int(days))
            return pd.DataFrame({
                'date': forecast_dates,
                'cases': forecast_values,
                'temperature': [temp] * days,
                'rainfall': [rainfall] * days
            })
        
        # Handle Prophet models
        elif isinstance(model, Prophet):
            future = model.make_future_dataframe(periods=days)
            future['temp'] = temp
            future['rain'] = rainfall
            forecast = model.predict(future)
            return forecast[['ds', 'yhat']].rename(
                columns={'ds': 'date', 'yhat': 'cases'}
            )
        
        # Handle NeuralProphet models
        elif hasattr(model, "make_future_dataframe"):
            future = model.make_future_dataframe(periods=days)
            future['temp'] = temp
            future['rain'] = rainfall
            forecast = model.predict(future)
            return forecast[['ds', 'yhat']].rename(
                columns={'ds': 'date', 'yhat': 'cases'}
            )
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
    
    except Exception as e:
        st.error(f"""Forecast failed. Possible causes:
        1. Invalid model parameters
        2. Missing required data
        3. Model type mismatch
        Error details: {str(e)}""")
        return None

# --- Streamlit UI ---
st.title("🦟 Regional Malaria Forecasting")

# Sidebar Controls
with st.sidebar:
    st.header("Configuration")
    region = st.selectbox("Region", ["Juba", "Yei", "Wau"])
    model_type = st.selectbox("Model Type", ["arima", "prophet", "neural"])
    
    st.header("Environmental Factors")
    temp = st.slider("Temperature (°C)", 15.0, 40.0, 25.0, 0.5)
    rain = st.slider("Rainfall (mm)", 0.0, 300.0, 50.0, 5.0)
    days = st.slider("Forecast Days", 7, 365, 90, 1)
    
    if st.button("Generate Forecast", type="primary"):
        st.session_state.run_forecast = True

# Main Display Area
if not os.path.exists("Malaria Forecasting.zip"):
    st.error("❌ Missing model file. Please upload 'Malaria_Forecasting.zip'")
    st.stop()

if getattr(st.session_state, 'run_forecast', False):
    with st.spinner(f"Loading {model_type} model..."):
        model = load_regional_model(
            "Malaria Forecasting.zip",
            region.lower(),
            model_type
        )
    
    if model:
        st.success(f"✅ {model_type.upper()} model loaded successfully!")
        
        with st.spinner("Generating forecast..."):
            forecast = generate_forecast(model, days, temp, rain)
        
        # Safe check for valid forecast
        if forecast is not None and not forecast.empty:
            # Visualization
            st.subheader(f"{region} {model_type.upper()} Forecast")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(forecast['date'], forecast['cases'], 'b-', label='Cases')
            ax.set_title(
                f"Predicted Cases | Temp: {temp}°C, Rain: {rain}mm",
                pad=20
            )
            ax.set_xlabel("Date", labelpad=10)
            ax.set_ylabel("Cases", labelpad=10)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Data Export
            csv = forecast.to_csv(index=False)
            st.download_button(
                "📥 Download Forecast",
                data=csv,
                file_name=f"{region}_{model_type}_forecast.csv",
                mime="text/csv"
            )
        elif forecast is None:
            st.error("❌ Forecast generation failed (returned None)")
        else:
            st.error("❌ Forecast generated empty results")

# Debug Section
with st.expander("⚙️ Model Information"):
    if 'model' in locals():
        st.write(f"Loaded Model Type: {type(model).__name__}")
    try:
        with zipfile.ZipFile("Malaria Forecasting.zip") as z:
            st.write("Available Models:", z.namelist())
    except Exception as e:
        st.warning(f"Could not inspect ZIP file: {str(e)}")
