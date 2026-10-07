import pandas as pd
from pathlib import Path


# ============================================================
# PDS MULTI-METHOD ANOMALY INVESTIGATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

print("=" * 65)
print("PDS MULTI-METHOD ANOMALY INVESTIGATION")
print("=" * 65)


# ============================================================
# 1. LOAD FILES
# ============================================================

comparison = pd.read_csv(
    OUTPUT_DIR / "anomaly_method_comparison.csv"
)

isolation = pd.read_csv(
    OUTPUT_DIR / "isolation_forest_shop_anomalies.csv"
)

transaction = pd.read_csv(
    OUTPUT_DIR / "transaction_outliers.csv"
)

ratio = pd.read_csv(
    OUTPUT_DIR / "ration_card_ratio_outliers.csv"
)

rice = pd.read_csv(
    OUTPUT_DIR / "rice_transaction_outliers.csv"
)


# ============================================================
# 2. PREPARE TRANSACTION DATA
# ============================================================

transaction_metrics = transaction[
    [
        "shopNo",
        "district_name",
        "transactions_per_month"
    ]
].copy()

transaction_metrics.rename(
    columns={
        "transactions_per_month":
        "iqr_transactions_per_month"
    },
    inplace=True
)


# ============================================================
# 3. PREPARE RATIO DATA
# ============================================================

ratio_metrics = ratio[
    [
        "shopNo",
        "transactions_per_ration_card"
    ]
].copy()

ratio_metrics.rename(
    columns={
        "transactions_per_ration_card":
        "iqr_transactions_per_ration_card"
    },
    inplace=True
)


# ============================================================
# 4. PREPARE RICE DATA
# ============================================================

rice_metrics = rice[
    [
        "shopNo",
        "rice_per_transaction"
    ]
].copy()

rice_metrics.rename(
    columns={
        "rice_per_transaction":
        "iqr_rice_per_transaction"
    },
    inplace=True
)


# ============================================================
# 5. PREPARE ISOLATION FOREST DATA
# ============================================================

isolation_metrics = isolation[
    [
        "shopNo",
        "district_name",
        "transactions_per_month",
        "transactions_per_ration_card",
        "rice_per_transaction",
        "total_ration_cards",
        "total_transactions",
        "anomaly_score",
        "anomaly_label"
    ]
].copy()

isolation_metrics.rename(
    columns={
        "district_name":
        "ml_district_name",

        "transactions_per_month":
        "ml_transactions_per_month",

        "transactions_per_ration_card":
        "ml_transactions_per_ration_card",

        "rice_per_transaction":
        "ml_rice_per_transaction"
    },
    inplace=True
)


# ============================================================
# 6. MERGE COMPARISON + ISOLATION FOREST
# ============================================================

investigation = comparison.merge(
    isolation_metrics,
    on="shopNo",
    how="left"
)


# ============================================================
# 7. ADD IQR METRICS
# ============================================================

investigation = investigation.merge(
    transaction_metrics[
        [
            "shopNo",
            "district_name",
            "iqr_transactions_per_month"
        ]
    ],
    on="shopNo",
    how="left",
    suffixes=("", "_iqr")
)

investigation = investigation.merge(
    ratio_metrics,
    on="shopNo",
    how="left"
)

investigation = investigation.merge(
    rice_metrics,
    on="shopNo",
    how="left"
)


# ============================================================
# 8. CREATE FINAL DISTRICT COLUMN
# ============================================================

# The transaction IQR dataset contains district information.
# Use it as the main district reference.

if "district_name_iqr" in investigation.columns:

    investigation["district_name"] = (
        investigation["district_name_iqr"]
    )

elif "ml_district_name" in investigation.columns:

    investigation["district_name"] = (
        investigation["ml_district_name"]
    )

else:

    investigation["district_name"] = "Unknown"


investigation["district_name"] = (
    investigation["district_name"]
    .fillna("Unknown")
)


# ============================================================
# 9. HIGH-PRIORITY SHOPS
# ============================================================

high_priority = investigation[
    investigation["methods_detected"] >= 3
].copy()


print("\n" + "=" * 65)
print("HIGH-PRIORITY ANOMALY INVESTIGATION")
print("=" * 65)

print(
    "\nHigh-priority shops:",
    len(high_priority)
)


# ============================================================
# 10. SORT HIGH-PRIORITY SHOPS
# ============================================================

high_priority = high_priority.sort_values(
    by=[
        "methods_detected",
        "anomaly_score"
    ],
    ascending=[
        False,
        True
    ]
)


print("\nTop 20 high-priority shops:")

print(
    high_priority[
        [
            "shopNo",
            "district_name",
            "methods_detected",
            "iqr_transactions_per_month",
            "iqr_transactions_per_ration_card",
            "iqr_rice_per_transaction",
            "anomaly_score"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 11. DISTRICT-LEVEL ANALYSIS
# ============================================================

district_summary = (
    investigation
    .groupby("district_name")
    .agg(
        total_anomalous_shops=(
            "shopNo",
            "nunique"
        ),

        high_priority_shops=(
            "methods_detected",
            lambda x: (x >= 3).sum()
        ),

        very_high_priority_shops=(
            "methods_detected",
            lambda x: (x == 4).sum()
        ),

        average_methods_detected=(
            "methods_detected",
            "mean"
        ),

        average_anomaly_score=(
            "anomaly_score",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# 12. HIGH-PRIORITY PERCENTAGE
# ============================================================

district_summary["high_priority_percentage"] = (
    district_summary["high_priority_shops"]
    / district_summary["total_anomalous_shops"]
    * 100
)


district_summary = district_summary.sort_values(
    "high_priority_shops",
    ascending=False
)


print("\n" + "=" * 65)
print("DISTRICT-LEVEL HIGH-PRIORITY ANOMALIES")
print("=" * 65)

print(
    district_summary
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 13. ALL-4-METHOD ANOMALIES
# ============================================================

four_method = investigation[
    investigation["methods_detected"] == 4
].copy()


four_method = four_method.sort_values(
    "anomaly_score",
    ascending=True
)


print("\n" + "=" * 65)
print("SHOPS FLAGGED BY ALL 4 METHODS")
print("=" * 65)

print(
    "\nTotal shops:",
    len(four_method)
)


print(
    four_method[
        [
            "shopNo",
            "district_name",
            "iqr_transactions_per_month",
            "iqr_transactions_per_ration_card",
            "iqr_rice_per_transaction",
            "anomaly_score"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 14. STRONGEST ISOLATION FOREST ANOMALIES
# ============================================================

extreme_ml = investigation[
    investigation["isolation_forest"] == True
].copy()


extreme_ml = extreme_ml.sort_values(
    "anomaly_score",
    ascending=True
)


print("\n" + "=" * 65)
print("STRONGEST ISOLATION FOREST ANOMALIES")
print("=" * 65)


print(
    extreme_ml[
        [
            "shopNo",
            "district_name",
            "methods_detected",
            "ml_transactions_per_month",
            "ml_transactions_per_ration_card",
            "ml_rice_per_transaction",
            "total_ration_cards",
            "total_transactions",
            "anomaly_score"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 15. DISTRICTS WITH ALL-4-METHOD ANOMALIES
# ============================================================

four_method_districts = (
    four_method
    .groupby("district_name")
    .agg(
        four_method_anomalies=(
            "shopNo",
            "nunique"
        )
    )
    .reset_index()
    .sort_values(
        "four_method_anomalies",
        ascending=False
    )
)


print("\n" + "=" * 65)
print("DISTRICTS WITH ALL-4-METHOD ANOMALIES")
print("=" * 65)


print(
    four_method_districts
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 16. SAVE HIGH-PRIORITY SHOPS
# ============================================================

high_priority_file = (
    OUTPUT_DIR /
    "multi_method_high_priority_shops.csv"
)

high_priority.to_csv(
    high_priority_file,
    index=False
)


# ============================================================
# 17. SAVE DISTRICT SUMMARY
# ============================================================

district_file = (
    OUTPUT_DIR /
    "multi_method_district_summary.csv"
)

district_summary.to_csv(
    district_file,
    index=False
)


# ============================================================
# 18. SAVE FOUR-METHOD SHOPS
# ============================================================

four_method_file = (
    OUTPUT_DIR /
    "four_method_anomalous_shops.csv"
)

four_method.to_csv(
    four_method_file,
    index=False
)


# ============================================================
# 19. SAVE FOUR-METHOD DISTRICT SUMMARY
# ============================================================

four_method_district_file = (
    OUTPUT_DIR /
    "four_method_district_summary.csv"
)

four_method_districts.to_csv(
    four_method_district_file,
    index=False
)


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("MULTI-METHOD ANALYSIS COMPLETED")
print("=" * 65)

print(
    "\nHigh-priority shops:",
    len(high_priority)
)

print(
    "All-4-method anomalous shops:",
    len(four_method)
)

print("\nOutput files:")

print(high_priority_file)
print(district_file)
print(four_method_file)
print(four_method_district_file)

print("\n" + "=" * 65)