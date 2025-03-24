import os
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# Set same tracking URI as training
mlflow.set_tracking_uri("file:./mlruns")

experiment_name = "nyc-taxi-experiment"
client = MlflowClient()

# Ensure experiment exists
experiment = client.get_experiment_by_name(experiment_name)
if experiment is None:
    raise Exception(f"❌ Experiment '{experiment_name}' not found.")

# Get latest run
runs = client.search_runs(experiment.experiment_id, order_by=["start_time desc"], max_results=1)
run_id = runs[0].info.run_id

# Load model from latest run
model_uri = f"runs:/{run_id}/model"
model = mlflow.pyfunc.load_model(model_uri)

# Prepare input and predict
ride = {
    "PULocationID": "10",
    "DOLocationID": "50",
    "trip_distance": 6.2
}
prediction = model.predict([ride])

print(f"✅ Prediction from MLflow run {run_id}: {prediction}")
