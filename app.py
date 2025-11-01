from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# --- MODEL CONFIGURATION ---
MODEL_PATH = os.getenv("MODEL_PATH", r"C:\Users\Chandramouli bandaru\OneDrive\Desktop\Weather prediction\WeatherPrediction.keras")
PREDICTION_HORIZON = 5
LOOK_BACK_HOURS = 168
N_FEATURES_MODEL = 4  # Model trained with 4 features

# --- DUMMY SCALER (Replace with real scaler values if available) ---
TARGET_MIN_TEMP = 10.0
TARGET_MAX_TEMP = 45.0

class DummyScaler:
    def __init__(self, data_min, data_max):
        self.min_ = np.array([data_min])
        self.scale_ = np.array([data_max - data_min])

    def inverse_transform(self, X):
        return (X * self.scale_) + self.min_

    def transform(self, X):
        return (X - self.min_) / self.scale_

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    target_scaler = DummyScaler(TARGET_MIN_TEMP, TARGET_MAX_TEMP)
    feature_scaler = DummyScaler(TARGET_MIN_TEMP, TARGET_MAX_TEMP)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None


# --- HELPER FUNCTIONS ---
def fetch_historical_data(lat, lon, hours_needed):
    """Fetch past weather data (168 hours) using Open-Meteo API."""
    end_date = datetime.utcnow().strftime('%Y-%m-%d')
    start_date = (datetime.utcnow() - timedelta(days=7)).strftime('%Y-%m-%d')
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure"
    )
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Open-Meteo API error: {response.status_code}")
    data = response.json()
    df = pd.DataFrame(data["hourly"])
    df["datetime"] = pd.to_datetime(df["time"])
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    return df[["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure"]].tail(hours_needed)


def preprocess_sequence(df_features, feature_scaler, L):
    scaled_data = feature_scaler.transform(df_features.values)
    X_input = scaled_data[-L:].reshape(1, L, scaled_data.shape[1])
    return X_input


# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict_lstm/', methods=['POST'])
def predict_lstm():
    """Fetch live weather data and predict the next 5 hours."""
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 503
    try:
        data = request.get_json(force=True)

        # Get latitude and longitude from request
        lat = float(data.get('lat', 28.6139))  # Default: New Delhi
        lon = float(data.get('lon', 77.2090))

        # 1. Fetch weather data
        raw_data = fetch_historical_data(lat, lon, LOOK_BACK_HOURS)

        # 2. Preprocess
        X_input = preprocess_sequence(raw_data, feature_scaler, LOOK_BACK_HOURS)

        # 3. Predict
        predictions_scaled = model.predict(X_input)

        # 4. Inverse transform to Celsius
        predictions_celsius = [
            float(target_scaler.inverse_transform(np.array([[p]]) )[0][0])
            for p in predictions_scaled[0]
        ]

        # 5. Prepare timestamps for next 5 hours
        forecast_start_time = raw_data.index[-1] + timedelta(hours=1)
        times = [
            (forecast_start_time + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(PREDICTION_HORIZON)
        ]

        return jsonify({
            'predictions': predictions_celsius,
            'times': times,
            'lat': lat,
            'lon': lon,
            'message': '✅ Forecast successful.'
        })

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 Flask server running at http://127.0.0.1:5000")
    app.run(debug=True)
