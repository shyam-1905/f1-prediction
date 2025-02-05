from flask import Flask, request, jsonify
import pandas as pd
import pickle
import logging
from flask_cors import CORS
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load trained model
try:
    with open("tuned_model.pkl", "rb") as file:
        model = pickle.load(file)
    logging.info("✅ Model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading model: {e}")
    model = None

# Health check endpoint
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Race Position Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded!"}), 500

        # Get input data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided!"}), 400

        # Convert input JSON to DataFrame
        input_df = pd.DataFrame([data])

        # Ensure columns match the trained model's input
        expected_columns = ['LapNumber', 'LapTime', 'TyreLife', 'Speedl1', 'Speedl2', 
                            'SpeedFL', 'SpeedST', 'AirTemp', 'TrackTemp', 'Humidity', 
                            'WindSpeed', 'Compound', 'Compound2']
        missing_columns = [col for col in expected_columns if col not in input_df.columns]
        for col in missing_columns:
            input_df[col] = 0  # Fill missing values with 0

        input_df = input_df[expected_columns]

        # Make prediction
        prediction = model.predict(input_df)
        predicted_position = int(round(prediction[0]))

        return jsonify({"predicted_position": predicted_position})

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # For local testing
