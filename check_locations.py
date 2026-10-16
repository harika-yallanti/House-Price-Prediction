import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "house_price_final.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "cleaned_house_data.csv")

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATASET_PATH)

locations = [
    "Whitefield",
    "Sarjapur Road",
    "Electronic City",
    "Indira Nagar",
    "Rajaji Nagar"
]

test_data = pd.DataFrame([
    {
        "location": location,
        "total_sqft": 1200,
        "bath": 2,
        "balcony": 1,
        "bhk": 2
    }
    for location in locations
])

predictions = model.predict(test_data)

for location, price in zip(locations, predictions):
    print(f"{location}: Rs. {price:.2f} lakhs")

