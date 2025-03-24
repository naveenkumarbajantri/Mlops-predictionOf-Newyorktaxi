import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# Set your MLflow tracking URI (adjust path if needed)
mlflow.set_tracking_uri("file:./mlruns")

# Set the experiment name used during training
experiment_name = "nyc-taxi-experiment"
client = MlflowClient()

# Get experiment by name
experiment = client.get_experiment_by_name(experiment_name)

# Get the latest run for the experiment
runs = client.search_runs(experiment.experiment_id, order_by=["start_time desc"], max_results=1)
run_id = runs[0].info.run_id

# Load the model from the latest run
model_uri = f"runs:/{run_id}/model"
model = mlflow.pyfunc.load_model(model_uri)

# Sample input
ride = {
    "PULocationID": "10",
    "DOLocationID": "50",
    "trip_distance": 6.2
}

features = [ride]
prediction = model.predict(features)

print(f"✅ Prediction from MLflow run {run_id}: {prediction}")
