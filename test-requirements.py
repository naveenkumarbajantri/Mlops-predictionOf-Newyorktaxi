import requests
import mlflow

# ✅ Use relative path for GitHub runners
mlflow.set_tracking_uri("file:./mlruns")

ride = {
    "ride": {
        "PULocationID": 10,
        "DOLocationID": 50,
        "trip_distance": 6.2
    }
}

try:
    response = requests.post("http://127.0.0.1:9696/predict", json=ride)
    prediction = response.json()
    print("✅ Response from ML Model:")
    print(prediction)
except Exception as e:
    print("❌ Error connecting to the server:", e)
