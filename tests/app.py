"""Simple Flask API for Iris flower predictions."""

from math import isfinite

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from train import ARTIFACT_DIR, FEATURES, MODEL_FILE

app = Flask(__name__)

MODEL_PATH = ARTIFACT_DIR / MODEL_FILE


def load_model():
    """Load the previously trained model."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Model not found. Run 'python train.py' first."
        )

    return joblib.load(MODEL_PATH)


# Home route
@app.route("/", methods=["GET"])
def home():
    """Describe the API and its expected input."""
    return jsonify({
        "message": "Iris prediction API is running",
        "endpoint": "POST /predict",
        "required_features": FEATURES,
        "example": {
            "features": {
                "SepalLengthCm": 5.1,
                "SepalWidthCm": 3.5,
                "PetalLengthCm": 1.4,
                "PetalWidthCm": 0.2
            }
        }
    })


# Health check route
@app.route("/health", methods=["GET"])
def health():
    """Check whether the API is running."""
    return jsonify({"status": "healthy"}), 200


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    """Validate flower measurements and return a prediction."""
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "Provide a valid JSON object."}), 400

    features = payload.get("features")

    if not isinstance(features, dict):
        return jsonify({
            "error": "Provide measurements inside a 'features' object."
        }), 400

    missing = [name for name in FEATURES if name not in features]

    if missing:
        return jsonify({
            "error": "Missing measurements: " + ", ".join(missing)
        }), 400

    try:
        values = [float(features[name]) for name in FEATURES]
    except (TypeError, ValueError):
        return jsonify({
            "error": "All measurements must be numbers."
        }), 400

    if not all(isfinite(value) for value in values):
        return jsonify({
            "error": "Measurements must be finite numbers."
        }), 400

    try:
        model = load_model()
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 503

    sample = pd.DataFrame([values], columns=FEATURES)
    species = model.predict(sample)[0]

    return jsonify({"prediction": str(species)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)