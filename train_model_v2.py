
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

# 1. Load the cleaned dataset
df = pd.read_csv("dataset/cleaned_house_data.csv")

print("Original records:", len(df))

# 2. Remove clearly implausible records for this experiment
df["sqft_per_bhk"] = df["total_sqft"] / df["bhk"]

df = df[
    (df["total_sqft"] >= 300) &
    (df["total_sqft"] <= 10000) &
    (df["bhk"] <= 10) &
    (df["bath"] <= 10) &
    (df["sqft_per_bhk"] >= 300)
].copy()

df.drop(columns=["sqft_per_bhk"], inplace=True)

print("Records after filtering:", len(df))

# 3. Define features and target
X = df[["location", "total_sqft", "bath", "balcony", "bhk"]]
y = df["price"]

# 4. Split the filtered dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 5. Encode locations and build model
preprocessor = ColumnTransformer(
    transformers=[
        ("location", OneHotEncoder(handle_unknown="ignore"), ["location"])
    ],
    remainder="passthrough"
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", ExtraTreesRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

# 6. Train and evaluate
print("\nTraining filtered Extra Trees model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nFiltered Model Results")
print("----------------------")
print(f"MAE:  {mae:.2f} lakhs")
print(f"RMSE: {rmse:.2f} lakhs")
print(f"R2:   {r2:.4f}")

# 7. Save this experiment separately
os.makedirs("models", exist_ok=True)

with open("models/house_price_model_v2.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nFiltered model saved to models/house_price_model_v2.pkl")
