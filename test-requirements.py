import os
import mlflow
from mlflow.tracking import MlflowClient

# Set the same tracking URI used in training
mlflow.set_tracking_uri("file:./mlruns")

# Match the experiment name used in train.py
experiment_name = "nyc-taxi-experiment"
client = MlflowClient()

# Get experiment details
experiment = client.get_experiment_by_name(experiment_name)
if experiment is None:
    raise Exception(f"❌ Experiment '{experiment_name}' not found.")

# Get latest run ID
runs = client.search_runs(experiment.experiment_id, order_by=["start_time desc"], max_results=1)
if not runs:
    raise Exception("❌ No runs found under the experiment.")

latest_run_id = runs[0].info.run_id

# Load model
model_uri = f"runs:/{latest_run_id}/model"
model = mlflow.pyfunc.load_model(model_uri)

# Sample input for prediction
ride = {
    "PULocationID": "10",
    "DOLocationID": "50",
    "trip_distance": 6.2
}

features = [ride]
prediction = model.predict(features)

print("✅ Prediction from MLflow model:", prediction)
