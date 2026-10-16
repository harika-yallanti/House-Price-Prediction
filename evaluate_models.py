import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create output folder
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Loading dataset...")

df = pd.read_csv("dataset/cleaned_house_data.csv")

# Apply the same filters used in Experiment 5
df["sqft_per_bhk"] = df["total_sqft"] / df["bhk"]

df = df[
    (df["total_sqft"] >= 300) &
    (df["total_sqft"] <= 10000) &
    (df["bhk"] <= 10) &
    (df["bath"] <= 10) &
    (df["sqft_per_bhk"] >= 300)
].copy()

df.drop(columns=["sqft_per_bhk"], inplace=True)

X = df[["location", "total_sqft", "bath", "balcony", "bhk"]]
y = df["price"]

# Use the same split as Experiment 5
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
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

models = {
    "Extra Trees Log Target": {
        "regressor": ExtraTreesRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "dense": False,
        "log_target": True
    },
    "Histogram Gradient Boosting": {
        "regressor": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42
        ),
        "dense": True,
        "log_target": False
    }
}

predictions_by_model = {}
summary = []

for name, settings in models.items():
    print(f"\nEvaluating {name}...")

    pipeline = Pipeline([
        ("preprocessor", create_preprocessor(settings["dense"])),
        ("regressor", settings["regressor"])
    ])

    target_train = np.log1p(y_train) if settings["log_target"] else y_train

    pipeline.fit(X_train, target_train)

    predictions = pipeline.predict(X_test)

    if settings["log_target"]:
        predictions = np.expm1(predictions)

    predictions = np.maximum(predictions, 0)

    errors = predictions - y_test.to_numpy()

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    predictions_by_model[name] = predictions

    summary.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"MAE: {mae:.2f} lakhs")
    print(f"RMSE: {rmse:.2f} lakhs")
    print(f"R2: {r2:.4f}")
    print(f"Mean signed error: {np.mean(errors):.2f} lakhs")

# Save predictions for further analysis
prediction_results = X_test.reset_index(drop=True).copy()
prediction_results["Actual_Price"] = y_test.reset_index(drop=True)
prediction_results["Extra_Trees_Log_Prediction"] = (
    predictions_by_model["Extra Trees Log Target"]
)
prediction_results["Gradient_Boosting_Prediction"] = (
    predictions_by_model["Histogram Gradient Boosting"]
)

prediction_results["Extra_Trees_Absolute_Error"] = np.abs(
    prediction_results["Actual_Price"] -
    prediction_results["Extra_Trees_Log_Prediction"]
)

prediction_results["Gradient_Boosting_Absolute_Error"] = np.abs(
    prediction_results["Actual_Price"] -
    prediction_results["Gradient_Boosting_Prediction"]
)

prediction_results.to_csv(
    "results/prediction_error_analysis.csv",
    index=False
)

# Save comparison metrics
summary_df = pd.DataFrame(summary)
summary_df.to_csv("results/finalist_model_metrics.csv", index=False)

print("\nFINALIST COMPARISON")
print(summary_df.round(4).to_string(index=False))

# Plot 1: Actual vs predicted prices
plt.figure(figsize=(9, 6))

for name, predictions in predictions_by_model.items():
    plt.scatter(
        y_test,
        predictions,
        alpha=0.35,
        s=18,
        label=name
    )

maximum = max(float(y_test.max()), max(
    float(predictions.max())
    for predictions in predictions_by_model.values()
))

plt.plot([0, maximum], [0, maximum], linestyle="--", label="Perfect prediction")
plt.xlabel("Actual Price (lakhs)")
plt.ylabel("Predicted Price (lakhs)")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.tight_layout()
plt.savefig("results/actual_vs_predicted.png", dpi=150)
plt.show()

# Plot 2: Absolute error distributions
plt.figure(figsize=(9, 6))

plt.hist(
    prediction_results["Extra_Trees_Absolute_Error"],
    bins=50,
    alpha=0.6,
    label="Extra Trees Log Target"
)

plt.hist(
    prediction_results["Gradient_Boosting_Absolute_Error"],
    bins=50,
    alpha=0.6,
    label="Histogram Gradient Boosting"
)

plt.xlabel("Absolute Prediction Error (lakhs)")
plt.ylabel("Number of Properties")
plt.title("Distribution of Absolute Prediction Errors")
plt.legend()
plt.tight_layout()
plt.savefig("results/error_distribution.png", dpi=150)
plt.show()

# Plot 3: Error by actual price range
prediction_results["Price_Range"] = pd.cut(
    prediction_results["Actual_Price"],
    bins=[0, 50, 100, 200, 500, np.inf],
    labels=["0-50", "50-100", "100-200", "200-500", "500+"],
    right=False
)

range_results = prediction_results.groupby(
    "Price_Range", observed=False
).agg(
    Property_Count=("Actual_Price", "size"),
    Extra_Trees_MAE=(
        "Extra_Trees_Absolute_Error", "mean"
    ),
    Gradient_Boosting_MAE=(
        "Gradient_Boosting_Absolute_Error", "mean"
    )
).reset_index()

range_results.to_csv("results/error_by_price_range.csv", index=False)

print("\nERROR BY PRICE RANGE")
print(range_results.round(2).to_string(index=False))

plt.figure(figsize=(9, 6))

x = np.arange(len(range_results))
width = 0.35

plt.bar(
    x - width / 2,
    range_results["Extra_Trees_MAE"],
    width,
    label="Extra Trees Log Target"
)

plt.bar(
    x + width / 2,
    range_results["Gradient_Boosting_MAE"],
    width,
    label="Histogram Gradient Boosting"
)

plt.xticks(x, range_results["Price_Range"].astype(str))
plt.xlabel("Actual Price Range (lakhs)")
plt.ylabel("Mean Absolute Error (lakhs)")
plt.title("Prediction Error by House Price Range")
plt.legend()
plt.tight_layout()
plt.savefig("results/error_by_price_range.png", dpi=150)
plt.show()

print("\nAnalysis completed.")
print("Check the results folder for graphs and CSV reports.")
