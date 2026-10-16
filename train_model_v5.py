import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Loading dataset...")

df = pd.read_csv("dataset/cleaned_house_data.csv")

# Filter unrealistic property records
df["sqft_per_bhk"] = df["total_sqft"] / df["bhk"]

df = df[
    (df["total_sqft"] >= 300) &
    (df["total_sqft"] <= 10000) &
    (df["bhk"] <= 10) &
    (df["bath"] <= 10) &
    (df["sqft_per_bhk"] >= 300)
].copy()

df.drop(columns=["sqft_per_bhk"], inplace=True)

print("Filtered records:", len(df))

# Input features and target
X = df[["location", "total_sqft", "bath", "balcony", "bhk"]]
y = df["price"]

# One common train-test split for every model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

def create_preprocessor(dense=False):
    return ColumnTransformer(
        transformers=[
            (
                "location",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=not dense
                ),
                ["location"]
            )
        ],
        remainder="passthrough"
    )

# Define candidate models
models = {
    "Extra Trees": {
        "model": ExtraTreesRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "dense": False,
        "log_target": False
    },

    "Random Forest": {
        "model": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "dense": False,
        "log_target": False
    },

    "Histogram Gradient Boosting": {
        "model": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42
        ),
        "dense": True,
        "log_target": False
    },

    "Extra Trees Log Target": {
        "model": ExtraTreesRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "dense": False,
        "log_target": True
    }
}

results = []

for name, settings in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        (
            "preprocessor",
            create_preprocessor(dense=settings["dense"])
        ),
        ("regressor", settings["model"])
    ])

    # Log transformation is used only for this candidate
    if settings["log_target"]:
        target_train = np.log1p(y_train)
    else:
        target_train = y_train

    pipeline.fit(X_train, target_train)

    predictions = pipeline.predict(X_test)

    if settings["log_target"]:
        predictions = np.expm1(predictions)

    # Avoid negative prices in reported predictions
    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
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

# Display results ranked by MAE
results_df = pd.DataFrame(results).sort_values("MAE")

print("\nFAIR MODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "models/model_comparison_v5.csv",
    index=False
)

print("\nResults saved to models/model_comparison_v5.csv")


