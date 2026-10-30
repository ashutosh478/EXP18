
from flask import Flask, jsonify
from pathlib import Path
import joblib

app = Flask(__name__)

MODEL_PATH = Path(__file__).parent / "artifacts" / "iris_model.joblib"
model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return jsonify({
        "message": "Iris ML API is running!",
        "status": "success"
    })

@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

@app.route("/predict", methods=["POST"])
def predict():
    from flask import request

    data = request.get_json(silent=True) or {}
    features = data.get("features")

    if not isinstance(features, list) or len(features) != 4:
        return jsonify({
            "error": "Provide exactly 4 numeric features."
        }), 400

    try:
        features = [float(x) for x in features]
        prediction = model.predict([features])[0]
        return jsonify({"prediction": str(prediction)})
    except (TypeError, ValueError):
        return jsonify({"error": "Features must be numeric."}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
