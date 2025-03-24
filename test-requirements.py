import requests

url = "http://localhost:9696/predict"
ride = {
    "ride": {
        "PULocationID": 10,
        "DOLocationID": 50,
        "trip_distance": 6.2
    }
}

try:
    response = requests.post(url, json=ride)
    print("✅ Prediction from model:", response.json())
except Exception as e:
    print("❌ Failed to connect:", e)
