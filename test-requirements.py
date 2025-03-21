import requests

url = "http://127.0.0.1:9696/predict"

ride = {
    "ride": {
        "PULocationID": 166,
        "DOLocationID": 239,
        "trip_distance": 2.53
    }
}

try:
    response = requests.post(url, json=ride)
    print("✅ Response from ML Model:")
    print(response.json())
except requests.exceptions.RequestException as e:
    print("❌ Error connecting to the server:", e)
