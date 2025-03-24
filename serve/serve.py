import os
import mlflow
import mlflow.pyfunc
from flask import Flask, request, jsonify

# Set tracking URI to local mlruns directory
mlflow.set_tracking_uri("file:train/mlruns")

# Get latest run_id from the experiment
EXPERIMENT_NAME = "nyc-taxi-experiment"
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise Exception(f"Experiment '{EXPERIMENT_NAME}' not found.")

# Get the most recent run in this experiment
runs = client.search_runs(experiment.experiment_id, order_by=["start_time desc"], max_results=1)
if not runs:
    raise Exception("No runs found in the experiment.")

run_id = runs[0].info.run_id
print(f"✅ Serving model from run_id: {run_id}")

# Load the model
MODEL_PATH = f"train/mlruns/{experiment.experiment_id}/{run_id}/artifacts/model"
model = mlflow.pyfunc.load_model(MODEL_PATH)

# Flask app
app = Flask("duration-prediction")

@app.route("/predict", methods=["POST"])
def predict():
    ride = request.get_json()
    features = [ride]
    prediction = model.predict(features)
    return jsonify({"prediction": float(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
