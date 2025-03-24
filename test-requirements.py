import mlflow.pyfunc
import os

# Point directly to the saved model location (relative to Jenkins workspace)
MODEL_PATH = "train/mlruns/6021/a3b1ea226ea646aeb0fa88ca97980086/artifacts/model"

# Load model
model = mlflow.pyfunc.load_model(MODEL_PATH)

# Input data
ride = {
    "PULocationID": "10",
    "DOLocationID": "50",
    "trip_distance": 6.2
}

# Predict
features = [ride]
prediction = model.predict(features)

print("✅ Prediction from saved model:", prediction)
