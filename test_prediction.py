import joblib
import pandas as pd

# Load the saved model
model = joblib.load("models/house_price_final.pkl")

# Get a valid location from the dataset
df = pd.read_csv("dataset/cleaned_house_data.csv")

location = (
    df["location"]
    .dropna()
    .astype(str)
    .str.strip()
)

location = location[location != ""]

if location.empty:
    raise ValueError("No valid locations found in the dataset.")

sample_location = location.mode().iloc[0]

# Sample house details
sample_house = pd.DataFrame([{
    "location": sample_location,
    "total_sqft": 1200,
    "bath": 2,
    "balcony": 1,
    "bhk": 2
}])

# Predict the house price
predicted_price = float(model.predict(sample_house)[0])

print("\nHOUSE PRICE PREDICTION")
print("----------------------")
print("Location:", sample_location)
print("Area: 1200 square feet")
print("Bedrooms: 2 BHK")
print("Bathrooms: 2")
print("Balconies: 1")
print(f"Estimated price: ₹{predicted_price:.2f} lakhs")

if predicted_price < 0:
    print("Warning: The model returned a negative price.")
else:
    print("Prediction test completed.")


