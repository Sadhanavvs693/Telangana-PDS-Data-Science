import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("PDS ML - FEATURE LEAKAGE & RELATIONSHIP CHECK")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load ML dataset
# ---------------------------------------------------------

file_path = Path("output/pds_ml_dataset.csv")

df = pd.read_csv(file_path)

print("\nDataset shape:")
print(df.shape)

target = "high_transaction_flag"

# ---------------------------------------------------------
# 2. Numeric correlation with target
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("NUMERIC FEATURE CORRELATION WITH TARGET")
print("=" * 70)

numeric_df = df.select_dtypes(include="number")

correlation = (
    numeric_df.corr()[target]
    .drop(target)
    .sort_values(key=abs, ascending=False)
)

print("\nCorrelation with high_transaction_flag:")

for feature, value in correlation.items():
    print(f"{feature:25s}: {value:.6f}")

# ---------------------------------------------------------
# 3. Compare feature averages by target
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE MEANS BY TARGET")
print("=" * 70)

important_features = [
    "month",
    "year",
    "rcNfsaAay",
    "unitsNfsaAay",
    "rcNfsaPhh",
    "unitsNfsaPhh",
    "totalRcNfsa",
    "totalUnitsNfsa",
    "rcStateAay",
    "unitsStateAay",
    "rcStatePhh",
    "unitsStatePhh",
    "rcStateAap",
    "unitsStateAap",
    "totalRcState",
    "totalUnitsState",
    "totalRcs",
    "totalUnits",
    "noOfRcs",
    "longitude",
    "latitude",
    "noOfTrans"
]

available_features = [
    col for col in important_features
    if col in df.columns
]

group_means = df.groupby(target)[available_features].mean().T

group_means.columns = [
    "Normal_0",
    "High_1"
]

group_means["difference"] = (
    group_means["High_1"] -
    group_means["Normal_0"]
)

print(group_means.to_string())

# ---------------------------------------------------------
# 4. Check relationship between transaction count
#    and potential predictor variables
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CORRELATION WITH noOfTrans")
print("=" * 70)

transaction_corr = (
    numeric_df.corr()["noOfTrans"]
    .drop("noOfTrans")
    .sort_values(key=abs, ascending=False)
)

for feature, value in transaction_corr.items():
    print(f"{feature:25s}: {value:.6f}")

# ---------------------------------------------------------
# 5. Check target definition
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TARGET DEFINITION CHECK")
print("=" * 70)

threshold = df.loc[
    df["noOfTrans"].notna(),
    "noOfTrans"
].quantile(0.75)

print(f"\n75th percentile threshold: {threshold}")

print("\nMinimum transaction count by target:")

print(
    df.groupby(target)["noOfTrans"]
    .min()
)

print("\nMaximum transaction count by target:")

print(
    df.groupby(target)["noOfTrans"]
    .max()
)

# ---------------------------------------------------------
# 6. Check exact overlap / deterministic relationships
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("POTENTIAL DETERMINISTIC FEATURES")
print("=" * 70)

# Check whether totalRcs and noOfRcs are identical
for col1, col2 in [
    ("totalRcs", "noOfRcs"),
    ("totalUnits", "noOfTrans")
]:

    if col1 in df.columns and col2 in df.columns:

        comparison = (
            df[col1].fillna(-999) ==
            df[col2].fillna(-999)
        )

        print(
            f"\n{col1} == {col2}: "
            f"{comparison.mean() * 100:.4f}%"
        )

# ---------------------------------------------------------
# 7. District target distribution
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TARGET RATE BY DISTRICT")
print("=" * 70)

district_target = (
    df.groupby("district")[target]
    .agg(["count", "mean"])
    .sort_values("mean", ascending=False)
)

district_target["high_percentage"] = (
    district_target["mean"] * 100
)

print(district_target.to_string())

# ---------------------------------------------------------
# 8. Final warning
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("LEAKAGE CHECK COMPLETE")
print("=" * 70)

print("""
IMPORTANT:
A very high model score does NOT automatically mean the model is good.

We must check whether the features contain information
that is too closely related to the target.

Next step will be decided based on these results.
""")