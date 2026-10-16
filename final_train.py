import os
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

# Create the models folder if it does not exist
os.makedirs("models", exist_ok=True)

print("Loading cleaned dataset...")

df = pd.read_csv("dataset/cleaned_house_data.csv")

# Apply the same filtering rules used in our experiments
df["sqft_per_bhk"] = df["total_sqft"] / df["bhk"]

df = df[
    (df["total_sqft"] >= 300) &
    (df["total_sqft"] <= 10000) &
    (df["bhk"] <= 10) &
    (df["bath"] <= 10) &
    (df["sqft_per_bhk"] >= 300)
].copy()

df.drop(columns=["sqft_per_bhk"], inplace=True)

# Select the input features and target price
X = df[["location", "total_sqft", "bath", "balcony", "bhk"]]
y = df["price"]

print("Records used for final training:", len(df))

# Convert location names into numerical features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "location",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            ["location"]
        )
    ],
    remainder="passthrough"
)

# Build the selected model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42
        )
    )
])

print("Training the final model...")

# Train using all filtered records
model.fit(X, y)

# Save the complete pipeline
model_path = "models/house_price_final.pkl"
joblib.dump(model, model_path)

print("\nFinal model trained and saved successfully.")
print("Model path:", model_path)
print("Input features:", list(X.columns))
print("Target: price in lakhs")
print("Training records:", len(X))
