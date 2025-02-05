from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained model
with open("tuned_model.pkl", "rb") as file:
    model = pickle.load(file)

# Health check endpoint
@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Race Position Prediction API is running!'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.get_json()

        # Convert input JSON to DataFrame
        input_df = pd.DataFrame([data])

        # Ensure columns match the trained model's input
        expected_columns = ['LapNumber', 'LapTime', 'TyreLife', 'Speedl1', 'Speedl2', 
                            'SpeedFL', 'SpeedST', 'AirTemp', 'TrackTemp', 'Humidity', 
                            'WindSpeed', 'Compound', 'Compound2']
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Fill missing values with 0

        input_df = input_df[expected_columns]

        # Make prediction
        prediction = model.predict(input_df)
        predicted_position = int(round(prediction[0]))

        return jsonify({'predicted_position': predicted_position})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
