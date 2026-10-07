import pandas as pd
from pathlib import Path


# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"


print("=" * 60)
print("PDS ANOMALY COMPARISON")
print("=" * 60)


# ==================================================
# 1. LOAD ANOMALY FILES
# ==================================================

transaction_file = (
    OUTPUT_DIR / "transaction_outliers.csv"
)

ratio_file = (
    OUTPUT_DIR / "ration_card_ratio_outliers.csv"
)

rice_file = (
    OUTPUT_DIR / "rice_transaction_outliers.csv"
)

isolation_file = (
    OUTPUT_DIR / "isolation_forest_shop_anomalies.csv"
)


transaction_outliers = pd.read_csv(
    transaction_file
)

ratio_outliers = pd.read_csv(
    ratio_file
)

rice_outliers = pd.read_csv(
    rice_file
)

isolation_results = pd.read_csv(
    isolation_file
)


# ==================================================
# 2. CREATE SHOP ID SETS
# ==================================================

transaction_shops = set(
    transaction_outliers["shopNo"]
)

ratio_shops = set(
    ratio_outliers["shopNo"]
)

rice_shops = set(
    rice_outliers["shopNo"]
)

isolation_shops = set(
    isolation_results.loc[
        isolation_results["anomaly_prediction"] == -1,
        "shopNo"
    ]
)


# ==================================================
# 3. BASIC COUNTS
# ==================================================

print("\n" + "=" * 60)
print("ANOMALY COUNTS")
print("=" * 60)

print(
    "\nTransaction outliers:",
    len(transaction_shops)
)

print(
    "Transaction/ration-card outliers:",
    len(ratio_shops)
)

print(
    "Rice/transaction outliers:",
    len(rice_shops)
)

print(
    "Isolation Forest anomalies:",
    len(isolation_shops)
)


# ==================================================
# 4. OVERLAP BETWEEN IQR AND ISOLATION FOREST
# ==================================================

transaction_isolation_overlap = (
    transaction_shops
    & isolation_shops
)

ratio_isolation_overlap = (
    ratio_shops
    & isolation_shops
)

rice_isolation_overlap = (
    rice_shops
    & isolation_shops
)


print("\n" + "=" * 60)
print("IQR / ISOLATION FOREST OVERLAP")
print("=" * 60)

print(
    "\nTransaction + Isolation Forest:",
    len(transaction_isolation_overlap)
)

print(
    "Ratio + Isolation Forest:",
    len(ratio_isolation_overlap)
)

print(
    "Rice + Isolation Forest:",
    len(rice_isolation_overlap)
)


# ==================================================
# 5. ALL STATISTICAL ANOMALIES
# ==================================================

all_iqr_shops = (
    transaction_shops
    | ratio_shops
    | rice_shops
)


# ==================================================
# 6. COMMON ANOMALIES
# ==================================================

common_anomalies = (
    all_iqr_shops
    & isolation_shops
)


print("\n" + "=" * 60)
print("COMMON ANOMALIES")
print("=" * 60)

print(
    "\nShops detected by both statistical and ML methods:",
    len(common_anomalies)
)


# ==================================================
# 7. IQR-ONLY ANOMALIES
# ==================================================

iqr_only = (
    all_iqr_shops
    - isolation_shops
)


print(
    "\nShops detected only by statistical methods:",
    len(iqr_only)
)


# ==================================================
# 8. ISOLATION FOREST ONLY
# ==================================================

isolation_only = (
    isolation_shops
    - all_iqr_shops
)


print(
    "Shops detected only by Isolation Forest:",
    len(isolation_only)
)


# ==================================================
# 9. CREATE COMPARISON DATASET
# ==================================================

all_shops = (
    all_iqr_shops
    | isolation_shops
)


comparison = pd.DataFrame({
    "shopNo": list(all_shops)
})


comparison["iqr_transaction"] = (
    comparison["shopNo"]
    .isin(transaction_shops)
)

comparison["iqr_ratio"] = (
    comparison["shopNo"]
    .isin(ratio_shops)
)

comparison["iqr_rice"] = (
    comparison["shopNo"]
    .isin(rice_shops)
)

comparison["isolation_forest"] = (
    comparison["shopNo"]
    .isin(isolation_shops)
)


# ==================================================
# 10. COUNT NUMBER OF METHODS
# ==================================================

comparison["methods_detected"] = (
    comparison[
        [
            "iqr_transaction",
            "iqr_ratio",
            "iqr_rice",
            "isolation_forest"
        ]
    ]
    .sum(axis=1)
)


# ==================================================
# 11. RISK / PRIORITY CATEGORY
# ==================================================

comparison["priority"] = "Low"

comparison.loc[
    comparison["methods_detected"] == 2,
    "priority"
] = "Medium"

comparison.loc[
    comparison["methods_detected"] >= 3,
    "priority"
] = "High"


# ==================================================
# 12. HIGH PRIORITY SHOPS
# ==================================================

high_priority = (
    comparison[
        comparison["priority"] == "High"
    ]
    .sort_values(
        "methods_detected",
        ascending=False
    )
)


print("\n" + "=" * 60)
print("HIGH PRIORITY ANOMALOUS SHOPS")
print("=" * 60)

print(
    "\nNumber of high-priority shops:",
    len(high_priority)
)


print(
    high_priority.head(20)
    .to_string(index=False)
)


# ==================================================
# 13. METHOD SUMMARY
# ==================================================

method_summary = pd.DataFrame({
    "method": [
        "Transaction IQR",
        "Transaction/Ration Card IQR",
        "Rice/Transaction IQR",
        "Isolation Forest"
    ],

    "anomalous_shops": [
        len(transaction_shops),
        len(ratio_shops),
        len(rice_shops),
        len(isolation_shops)
    ]
})


print("\n" + "=" * 60)
print("METHOD SUMMARY")
print("=" * 60)

print(
    method_summary.to_string(index=False)
)


# ==================================================
# 14. SAVE RESULTS
# ==================================================

comparison_file = (
    OUTPUT_DIR /
    "anomaly_method_comparison.csv"
)

high_priority_file = (
    OUTPUT_DIR /
    "high_priority_anomalous_shops.csv"
)

method_summary_file = (
    OUTPUT_DIR /
    "anomaly_method_summary.csv"
)


comparison.to_csv(
    comparison_file,
    index=False
)

high_priority.to_csv(
    high_priority_file,
    index=False
)

method_summary.to_csv(
    method_summary_file,
    index=False
)


# ==================================================
# 15. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("ANOMALY COMPARISON COMPLETED")
print("=" * 60)

print("\nOutput files:")

print(comparison_file)

print(high_priority_file)

print(method_summary_file)

print("\n" + "=" * 60)