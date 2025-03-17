from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Check if model files exist
model_path = "churn_model.pkl"
scaler_path = "scaler.pkl"
geo_encoder_path = "le_geo.pkl"
gender_encoder_path = "le_gender.pkl"

if not all(os.path.exists(path) for path in [model_path, scaler_path, geo_encoder_path, gender_encoder_path]):
    logging.error("One or more model files are missing. Please retrain the model.")
    raise FileNotFoundError("One or more model files are missing. Please retrain the model.")

# Load trained model & scalers
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le_geo = joblib.load(geo_encoder_path)
    le_gender = joblib.load(gender_encoder_path)
except Exception as e:
    logging.error(f"Error loading model files: {e}")
    raise e

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form

        # Extract & Validate Inputs
        try:
            credit_score = int(data.get("CreditScore", 0))
            geography = data.get("Geography", "").strip()
            gender = data.get("Gender", "").strip()
            age = int(data.get("Age", 0))
            tenure = int(data.get("Tenure", 0))
            balance = float(data.get("Balance", 0.0))
            num_of_products = int(data.get("NumOfProducts", 0))
            has_credit_card = int(data.get("HasCrCard", 0))
            is_active_member = int(data.get("IsActiveMember", 0))
            estimated_salary = float(data.get("EstimatedSalary", 0.0))
        except ValueError:
            logging.error("Invalid input data format")
            return jsonify({"error": "Invalid input data format"}), 400

        # Ensure categorical variables exist in the encoder
        if geography not in le_geo.classes_ or gender not in le_gender.classes_:
            logging.error("Invalid Geography or Gender value")
            return jsonify({"error": "Invalid Geography or Gender value"}), 400

        # Encode categorical variables
        geo_encoded = le_geo.transform([geography])[0]
        gender_encoded = le_gender.transform([gender])[0]

        # Create feature array
        input_features = np.array([
            credit_score, geo_encoded, gender_encoded, age, tenure,
            balance, num_of_products, has_credit_card, is_active_member, estimated_salary
        ]).reshape(1, -1)

        # Scale the input
        input_features_scaled = scaler.transform(input_features)

        # Make prediction
        prediction = model.predict(input_features_scaled)[0]
        result = "Churn" if prediction == 1 else "No Churn"

        logging.info(f"Prediction result: {result}")
        return jsonify({"prediction": result})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
