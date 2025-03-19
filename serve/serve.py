import os
import mlflow
from flask import Flask, request, jsonify

# ✅ Set MLflow tracking URI (this should match the location used during training)
mlflow.set_tracking_uri("http://localhost:5000")  # Use MLflow UI tracking for artifact support

# ✅ Use your actual run ID from the MLflow experiment that logged the model
MODEL_URI = "runs:/62906aa8f8bb43aba63ae583e37fb566/model"
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")  # Optional version display

# ✅ Load model from MLflow
model = mlflow.pyfunc.load_model(MODEL_URI)

# ✅ Initialize Flask app
app = Flask("duration-prediction")


# 🔧 Function to prepare input features
def prepare_features(ride):
    return {
        "PULocationID": str(ride["PULocationID"]),
        "DOLocationID": str(ride["DOLocationID"]),
        "trip_distance": ride["trip_distance"]
    }


# 🚀 Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict_endpoint():
    ride = request.get_json()["ride"]
    features = prepare_features(ride)
    pred = model.predict([features])

    result = {
        "prediction": {
            "duration": float(pred[0])
        },
        "model_version": MODEL_VERSION
    }

    return jsonify(result)


# ▶️ Start the server
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
