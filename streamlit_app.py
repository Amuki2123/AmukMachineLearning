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
import warnings
warnings.filterwarnings("ignore")
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


# --- Compatibility Fixes ---
try:
    # Fix for NeuralProphet AttributeDict error
    from neuralprophet import utils
    if not hasattr(utils, 'AttributeDict'):
        class AttributeDict(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__
        utils.AttributeDict = AttributeDict
except ImportError:
    pass

# --- Model Loading ---
def load_regional_model(zip_path: str, region: str, model_type: str):
    """Load models with comprehensive error handling"""
    model_files = {
        "arima": f"{region}_arima_model.pkl",
        "prophet": f"{region}_prophet_model.json", 
        "neural": f"{region}_np_model.pkl"
    }
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Case-insensitive file search
            target_file = model_files[model_type]
            matched = next((f for f in zip_ref.namelist() if target_file.lower() in f.lower()), None)
            
            if not matched:
                available = [f for f in zip_ref.namelist() if region.lower() in f.lower()]
                st.error(f"Model not found. Available: {available}")
                return None
                
            with zip_ref.open(matched) as f:
                if matched.endswith('.pkl'):
                    # Special handling for different model types
                    model = pickle.load(f)
                    
                    # ARIMA model check
                    if model_type == "arima" and hasattr(model, 'order'):
                        from statsmodels.tsa.arima.model import ARIMA
                        return ARIMA(model)
                        
                    # NeuralProphet validation
                    elif model_type == "neural":
                        if not hasattr(model, 'predict'):
                            raise ValueError("Invalid NeuralProphet model")
                        return model
                        
                    return model
                    
                elif matched.endswith('.json'):
                    from keras.models import model_from_json
                    return model_from_json(f.read().decode('utf-8'))
    
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None

# --- Forecasting ---
def generate_forecast(model, days: int, temp: float, rain: float) -> Optional[pd.DataFrame]:
    """Universal forecasting function"""
    try:
        dates = pd.date_range(datetime.today(), periods=days)
        
        # ARIMA Models
        if hasattr(model, 'order'):  # ARIMA check
            from statsmodels.tsa.arima.model import ARIMAResults
            if isinstance(model, ARIMAResults):
                forecast = model.forecast(steps=days)
            else:
                forecast = model.fit().forecast(steps=days)
            return pd.DataFrame({
                'date': dates,
                'cases': forecast,
                'temperature': temp,
                'rainfall': rain
            })
        
        # Prophet Models
        elif hasattr(model, 'make_future_dataframe'):
            future = model.make_future_dataframe(periods=days)
            future['temp'] = temp
            future['rain'] = rain
            forecast = model.predict(future)
            return forecast[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'cases'})
        
        # NeuralProphet Models
        elif hasattr(model, 'predict'):
            future = model.make_future_dataframe(periods=days)
            future['temp'] = temp
            future['rain'] = rain
            forecast = model.predict(future)
            return forecast[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'cases'})
        
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
    
    except Exception as e:
        st.error(f"""Forecast failed. Possible causes:
        1. Model not properly initialized
        2. Missing required parameters
        3. Version incompatibility
        Technical details: {str(e)}""")
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="Malaria Forecast", layout="wide")
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

# Main Display
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
        st.success(f"✅ {model_type.upper()} model loaded!")
        
        with st.spinner("Generating forecast..."):
            forecast = generate_forecast(model, days, temp, rain)
        
        if forecast is not None and not forecast.empty:
            # Visualization
            st.subheader(f"{region} Forecast Results")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(forecast['date'], forecast['cases'], 'b-')
            ax.set_title(f"Predicted Cases | Temp: {temp}°C, Rain: {rain}mm")
            ax.set_xlabel("Date")
            ax.set_ylabel("Cases")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Data Export
            csv = forecast.to_csv(index=False)
            st.download_button(
                "📥 Download Forecast",
                data=csv,
                file_name=f"{region}_forecast.csv",
                mime="text/csv"
            )
        elif forecast is None:
            st.error("❌ Forecast generation failed")
        else:
            st.error("❌ Empty forecast results")

# Debug Section
with st.expander("⚙️ Model Information"):
    try:
        with zipfile.ZipFile("Malaria Forecasting.zip") as z:
            st.write("Available models:", z.namelist())
    except:
        st.warning("Could not inspect ZIP file")
