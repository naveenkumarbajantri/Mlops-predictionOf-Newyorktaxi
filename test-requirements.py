import requests

url = "http://127.0.0.1:9696/predict"

ride = {
    "ride": {
        "PULocationID": 42,
        "DOLocationID": 42,
        "trip_distance": 0.44
    }
}

try:
    response = requests.post(url, json=ride)
    print("✅ Response from ML Model:")
    print(response.json())
except requests.exceptions.RequestException as e:
    print("❌ Error connecting to the server:", e)
