
import pandas as pd

df = pd.read_csv("dataset/cleaned_house_data.csv")

print("DATASET SUMMARY")
print(df.describe().round(2))

print("\nPRICE PERCENTILES")
print(df["price"].quantile([0.50, 0.75, 0.90, 0.95, 0.99]))

print("\nHIGHEST HOUSE PRICES")
print(
    df.nlargest(10, "price")[
        ["location", "total_sqft", "bath", "bhk", "price"]
    ].to_string(index=False)
)

print("\nLOWEST HOUSE PRICES")
print(
    df.nsmallest(10, "price")[
        ["location", "total_sqft", "bath", "bhk", "price"]
    ].to_string(index=False)
)

print("\nAREA PER BEDROOM")
df["sqft_per_bhk"] = df["total_sqft"] / df["bhk"]

print(
    df["sqft_per_bhk"].describe().round(2)
)

print("\nVERY SMALL AREA PER BEDROOM")
print(
    df[df["sqft_per_bhk"] < 300][
        ["location", "total_sqft", "bhk", "price"]
    ].head(15).to_string(index=False)
)
