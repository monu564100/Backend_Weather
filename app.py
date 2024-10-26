from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import joblib
import numpy as np
from pymongo import MongoClient
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://weather-ml-prediction.netlify.app","https://prediction-model.netlify.app"]}})

# MongoDB setup
mongo_client = MongoClient("mongodb+srv://dreadwing5:UPw4YunHTB6ISOIi@cluster0.5mb2e.mongodb.net/cmrit-mentor")
db = mongo_client["disaster_prediction_db"]
help_requests_collection = db["help_requests"]

# Load the trained models
fire_model = pickle.load(open('model.pkl', 'rb'))  # Fire prediction model
flood_model = joblib.load('flood_model.pkl')       # Flood prediction model

@app.route('/')
def index():
    return "Welcome to the Natural Disaster Prediction API"

# Forest fire prediction route
@app.route('/predict', methods=['POST'])
def predict_fire():
    data = request.get_json()
    try:
        int_features = [float(data['oxygen']), float(data['temperature']), float(data['humidity'])]
        final = [np.array(int_features)]
        prediction = fire_model.predict_proba(final)
        output = float(prediction[0][1])  # Probability of fire occurring

        message = "Your Forest is in Danger." if output > 0.5 else "Your Forest is safe."
        return jsonify({"prediction": message, "probability": output})

    except Exception as e:
        return jsonify({"error": str(e), "message": "Invalid input or model error"}), 400

# Flood prediction route
@app.route('/predict_flood', methods=['POST'])
def predict_flood():
    data = request.get_json()
    try:
        precipitation = float(data['precipitation'])
        temperature = float(data['temperature'])
        X_location = np.array([[precipitation, temperature]])
        prediction = flood_model.predict(X_location)
        result = "Flood-prone" if prediction[0] == 1 else "Not flood-prone"

        return jsonify({"precipitation": precipitation, "temperature": temperature, "prediction": result})

    except Exception as e:
        return jsonify({"error": str(e), "message": "Failed to predict flood risk"}), 400

# Route to submit help request
@app.route('/submit_help', methods=['POST'])
def submit_help():
    data = request.get_json()
    try:
        name = data['name']
        mobile = data['mobile']
        location = data.get('location', {})
        # request_type = data['request_type']  # Specify whether it's "fire" or "flood"

        help_request = {
            "name": name,
            "mobile": mobile,
            # "request_type": request_type,
            "location": {
                "latitude": location.get('latitude'),
                "longitude": location.get('longitude'),
                "city": location.get('city'),
                "district": location.get('district')
            },
            "status": "Pending"
        }

        help_requests_collection.insert_one(help_request)
        return jsonify({"message": "Help request submitted successfully"}), 201

    except Exception as e:
        return jsonify({"error": str(e), "message": "Failed to submit help request"}), 400

# Route to retrieve all help requests
@app.route('/help_requests', methods=['GET'])
def get_help_requests():
    try:
        help_requests = list(help_requests_collection.find({}, {"_id": 0}))  # Exclude MongoDB _id for simplicity
        return jsonify(help_requests), 200

    except Exception as e:
        return jsonify({"error": str(e), "message": "Failed to fetch help requests"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
