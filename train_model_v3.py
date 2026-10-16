
import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the cleaned dataset
df = pd.read_csv("dataset/cleaned_house_data.csv")

df["sqft_per_bhk"] = df["total_sqft"] / df["bhk"]

df = df[
    (df["total_sqft"] >= 300) &
    (df["total_sqft"] <= 10000) &
    (df["bhk"] <= 10) &
    (df["bath"] <= 10) &
    (df["sqft_per_bhk"] >= 300)
].copy()

df.drop(columns=["sqft_per_bhk"], inplace=True)

# Define features
X = df[["location", "total_sqft", "bath", "balcony", "bhk"]]

# Transform the target price using log1p
y = np.log1p(df["price"])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Keep actual prices for evaluation
y_test_actual = np.expm1(y_test)

# Preprocess locations
preprocessor = ColumnTransformer(
    transformers=[
        ("location", OneHotEncoder(handle_unknown="ignore"), ["location"])
    ],
    remainder="passthrough"
)

# Create model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", ExtraTreesRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

# Train on log-transformed prices
print("Training log-transformed Extra Trees model...")
model.fit(X_train, y_train)

# Predict and convert back to lakhs
log_predictions = model.predict(X_test)
predictions = np.maximum(0, np.expm1(log_predictions))

# Evaluate using actual prices
mae = mean_absolute_error(y_test_actual, predictions)
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
r2 = r2_score(y_test_actual, predictions)

print("\nLog-Transformed Model Results")
print("-----------------------------")
print(f"MAE:  {mae:.2f} lakhs")
print(f"RMSE: {rmse:.2f} lakhs")
print(f"R2:   {r2:.4f}")

# Save the model and indicate that its target is logarithmic
os.makedirs("models", exist_ok=True)

with open("models/house_price_model_v3.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved to models/house_price_model_v3.pkl")
print("Important: predictions from this model must be converted with np.expm1().")
