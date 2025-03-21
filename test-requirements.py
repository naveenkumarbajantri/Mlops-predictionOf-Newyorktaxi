import os
import mlflow
import json

# 🔁 Set MLflow tracking to your local mlruns directory (adjust path if needed)
mlflow.set_tracking_uri("file://./train/mlruns")

# ✅ Automatically get latest run ID from experiment
client = mlflow.tracking.MlflowClient()
experiment_name = "nyc-taxi-experiment"

# Get experiment
experiment = client.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# Get latest run
runs = client.search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"], max_results=1)
run_id = runs[0].info.run_id

print(f"✅ Latest run ID: {run_id}")

#  Load model from latest run
model_uri = f"runs:/{run_id}/model"
model = mlflow.pyfunc.load_model(model_uri)

# 🧪 Sample input (same structure as training)
ride = {
    "PULocationID": "10",
    "DOLocationID": "50",
    "trip_distance": 6.2
}

# ⏱️ Make prediction
prediction = model.predict([ride])
print("✅ Prediction Result:", json.dumps({"duration": float(prediction[0])}, indent=2))
