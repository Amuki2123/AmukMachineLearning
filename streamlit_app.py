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
    """Robust data loading with multiple fallbacks"""
    if not os.path.exists(DATA_FILE):
        st.error(f"Error: Data file '{DATA_FILE}' not found.")
        st.info("Please upload the data file first.")
        return None
    
    try:
        # Try multiple possible delimiters
        for delimiter in [',', ';', '\t']:
            try:
                df = pd.read_csv(DATA_FILE, delimiter=delimiter)
                if not df.empty:
                    break
            except:
                continue
                
        if df.empty:
            st.error("Error: Unable to read data file with common delimiters")
            return None
            
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Check required columns
        required_cols = ['Date', 'Region', 'Cases', 'Temperature', 'Rainfall']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None
            
        # Convert dates
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if df['Date'].isnull().any():
            st.warning("Some dates couldn't be parsed - these rows will be dropped")
            df = df.dropna(subset=['Date'])
            
        # Show data summary
        st.write("Data Summary:")
        st.write(f"Total rows: {len(df)}")
        st.write("Missing values per column:")
        st.write(df.isnull().sum())
        
        return df
        
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return None

def prepare_region_data(df, region):
    """Prepare dataset with robust missing value handling"""
    try:
        if df is None or region not in REGIONS:
            return None
            
        # Filter region data
        region_df = df[df['Region'] == region][['Date', 'Cases', 'Temperature', 'Rainfall']].copy()
        if region_df.empty:
            st.warning(f"No data found for region: {region}")
            return None
            
        # Create properly named columns
        region_df['ds'] = pd.to_datetime(region_df['Date'])
        region_df['y'] = region_df['Cases']
        
        # Set datetime index
        region_df = region_df.set_index('ds').sort_index()
        
        # Handle missing values with multiple strategies
        for col in ['y', 'Temperature', 'Rainfall']:
            if region_df[col].isnull().any():
                # First try forward fill
                region_df[col] = region_df[col].ffill()
                
                # Then backward fill
                region_df[col] = region_df[col].bfill()
                
                # Then time-based interpolation
                if region_df[col].isnull().any():
                    region_df[col] = region_df[col].interpolate(method='time')
                
                # Finally, fill with mean if still missing
                if region_df[col].isnull().any():
                    region_df[col] = region_df[col].fillna(region_df[col].mean())
        
        return region_df.reset_index()
        
    except Exception as e:
        st.error(f"Error preparing data for {region}: {str(e)}")
        return None

def check_data_quality(df, region):
    """Flexible data quality checks with warnings"""
    try:
        if df is None:
            return False
            
        st.subheader(f"Data Quality Report for {region}")
        
        # Calculate missing percentage
        missing_pct = df.isnull().mean() * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Missing Values (%):")
            st.write(missing_pct)
            
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
        
        # Warn but don't fail for missing data
        if missing_pct.get('y', 0) > 50:
            st.warning(f"High percentage of missing cases data ({missing_pct.get('y', 0):.1f}%) - models may be less accurate")
        
        if len(df) < 30:
            st.warning("Limited historical data (less than 30 days) - forecasts may be less reliable")
            
        return True  # Always attempt training
        
    except Exception as e:
        st.error(f"Data quality check failed: {str(e)}")
        return False

# --- Model Training Functions --- 
[Previous model training functions remain exactly the same]

# --- Model Training Interface ---
[Previous train_all_models() function remains exactly the same]

# --- Forecasting Functions ---
[Previous forecasting functions remain exactly the same]

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Malaria Forecasting", layout="wide")
    st.title("🦟 Malaria Cases Forecasting with Environmental Factors 🦟")
    
    # File Upload Section
    st.header("Data Management")
    with st.expander("📤 Update Data File", expanded=True):  # Expanded by default for visibility
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                # Try multiple delimiters for upload
                for delimiter in [',', ';', '\t']:
                    try:
                        df = pd.read_csv(uploaded_file, delimiter=delimiter)
                        if not df.empty:
                            break
                    except:
                        continue
                
                if df.empty:
                    st.error("Could not read uploaded file with common delimiters")
                    return
                
                # Validate columns
                required_cols = ['Date', 'Region', 'Cases', 'Temperature', 'Rainfall']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                    return
                
                # Clean and save data
                df.columns = df.columns.str.strip()
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date'])
                df.to_csv(DATA_FILE, index=False)
                
                st.success("File uploaded successfully!")
                st.write("First 5 rows of uploaded data:")
                st.write(df.head())
                st.cache_data.clear()
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Data Diagnostics
    if os.path.exists(DATA_FILE):
        st.header("Data Diagnostics")
        df = load_data()
        if df is not None:
            for region in REGIONS:
                with st.expander(f"Data Summary - {region}"):
                    region_data = df[df['Region'] == region]
                    if not region_data.empty:
                        st.write(f"Date Range: {region_data['Date'].min()} to {region_data['Date'].max()}")
                        st.write("Missing Values:")
                        st.write(region_data[['Cases', 'Temperature', 'Rainfall']].isnull().sum())
                        st.write("Statistics:")
                        st.write(region_data[['Cases', 'Temperature', 'Rainfall']].describe())
                    else:
                        st.warning(f"No data available for {region}")
    
    # Model Training Section
    st.header("Model Training")
    if st.button("Train All Models", key="train_button"):
        with st.spinner("Training models (this may take several minutes)..."):
            success = train_all_models()
            if success:
                st.balloons()
            else:
                st.error("Model training encountered errors - check diagnostics above")
    
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
