
import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. LOAD THE CLEANED DATASET


df = pd.read_csv("dataset/cleaned_house_data.csv")

print("Dataset shape:", df.shape)

# Features used to predict the house price
X = df[["location", "total_sqft", "bath", "balcony", "bhk"]]

# Target variable: house price in lakhs
y = df["price"]


# 2. SPLIT THE DATA


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# 3. PREPARE THE LOCATION FEATURE


def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "location",
                OneHotEncoder(handle_unknown="ignore"),
                ["location"]
            )
        ],
        remainder="passthrough"
    )


# 4. DEFINE THE MODELS


models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=150,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained_models = {}


# 5. TRAIN AND EVALUATE EACH MODEL


for name, regressor in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("regressor", regressor)
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Predict prices for unseen test records
    predictions = pipeline.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    print(f"MAE:  {mae:.2f} lakhs")
    print(f"RMSE: {rmse:.2f} lakhs")
    print(f"R2:   {r2:.4f}")

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[name] = pipeline


# 6. COMPARE THE RESULTS


results_df = pd.DataFrame(results)

# Rank models by lowest MAE
results_df = results_df.sort_values(
    by="MAE",
    ascending=True
).reset_index(drop=True)

print("\n======================================")
print("MODEL COMPARISON")
print("======================================")
print(results_df.round(4).to_string(index=False))

# Save the comparison report
os.makedirs("models", exist_ok=True)

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)


# 7. SAVE THE BEST MODEL


best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

with open("models/house_price_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("\nBest model based on MAE:", best_model_name)
print("Best MAE:", round(results_df.iloc[0]["MAE"], 2), "lakhs")
print("Comparison saved to models/model_comparison.csv")
print("Best model saved to models/house_price_model.pkl")
