import os
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# Point to the same tracking URI as train.py
mlflow.set_tracking_uri("file:./mlruns")

# Load the latest model from the experiment
experiment_name = "nyc-taxi-experiment"
client = MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)

if experiment is None:
    raise Exception(f"❌ Experiment '{experiment_name}' not found.")

# Get latest run
runs = client.search_runs(experiment.experiment_id, order_by=["start_time desc"], max_results=1)
latest_run = runs[0]
run_id = latest_run.info.run_id

# Load model from MLflow
model_uri = f"runs:/{run_id}/model"
model = mlflow.pyfunc.load_model(model_uri)

# Prepare input and make prediction
ride = {
    "PULocationID": "10",
    "DOLocationID": "50",
    "trip_distance": 6.2
}
features = [ride]
prediction = model.predict(features)

print("✅ Prediction from MLflow model:", prediction)
