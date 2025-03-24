import os
import mlflow

# Set local MLflow tracking path
mlflow.set_tracking_uri("file:./mlruns")
client = mlflow.MlflowClient()

experiment_name = "nyc-taxi-experiment"
experiment = client.get_experiment_by_name(experiment_name)

if experiment is None:
    raise Exception(f"❌ Experiment '{experiment_name}' not found.")

# Get latest run from the experiment
runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"], max_results=1)
if not runs:
    raise Exception("❌ No runs found in the experiment.")

run_id = runs[0].info.run_id
model_uri = f"mlruns/{experiment.experiment_id}/{run_id}/artifacts/model"
print(f"📦 Loading model from: {model_uri}")

model = mlflow.pyfunc.load_model(model_uri)

ride = {
    "PULocationID": "10",
    "DOLocationID": "50",
    "trip_distance": 6.2
}

prediction = model.predict([ride])
print("✅ Prediction from MLflow model:", prediction)
