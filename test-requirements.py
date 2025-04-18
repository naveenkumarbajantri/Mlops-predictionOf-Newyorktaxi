import mlflow.pyfunc
import os
MODEL_PATH = "train/mlruns/602170594853194749/a3b1ea226ea646aeb0fa88ca97980086/artifacts/model"
model = mlflow.pyfunc.load_model(MODEL_PATH)

ride = {
    "PULocationID": "43",
    "DOLocationID": "238",
    "trip_distance": 1.16
}

features = [ride]
prediction = model.predict(features)
print("Prediction from saved model:", prediction)
