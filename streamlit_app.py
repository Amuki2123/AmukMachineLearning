from flask import Flask, render_template, request, jsonify
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
logging.basicConfig(level=logging.INFO)  # Enable logging for debugging

# Load the pre-trained Prophet models for each region
MODELS = {
    'juba': model_from_json(open('juba_prophet_model.json', 'r').read()),
    'yei': model_from_json(open('yei_prophet_model.json', 'r').read()),
    'wau': model_from_json(open('wau_prophet_model.json', 'r').read())
}

def forecast(model, data):
    """
    Perform forecasting using the given Prophet model and input data.
    """
    try:
        # Validate required keys in the input data
        required_keys = ['date', 'Temperature', 'periods']
        if not all(key in data for key in required_keys):
            raise ValueError("Input data must contain 'date', 'Temperature', and 'periods'.")

        # Prepare the DataFrame for Prophet
        df = pd.DataFrame([data]).rename(columns={'date': 'ds', 'Temperature': 'y'})
        future = model.make_future_dataframe(periods=int(data['periods']), freq='M', include_history=False)
        future['y'] = data['Temperature']  # Assuming temperature is applied uniformly

        # Perform forecasting
        forecast_result = model.predict(future)
        result = forecast_result[['ds', 'yhat']].iloc[-1]  # Get the last forecasted value

        return result.to_dict()  # Return as a dictionary for JSON serialization
    except Exception as e:
        logging.error(f"Forecasting error: {e}")
        raise ValueError(f"Forecasting failed: {str(e)}")

@app.route('/')
def index():
    """
    Render the main HTML template for the application.
    """
    return render_template('index.html')

@app.route('/predict/<region>', methods=['POST'])
def predict(region):
    """
    Generalized route for predicting malaria cases in different regions.
    """
    region = region.lower()
    model = MODELS.get(region)
    if not model:
        return jsonify({'status': 'error', 'message': f"Invalid region '{region}' provided."}), 400

    try:
        data = request.get_json()
        result = forecast(model, data)
        return jsonify({'status': 'success', 'region': region.capitalize(), 'result': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
