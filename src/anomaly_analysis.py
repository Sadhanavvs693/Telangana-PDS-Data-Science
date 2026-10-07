import pandas as pd
import numpy as np
from pathlib import Path

# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data_processed"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

# ==================================================
# 1. LOAD MASTER DATASET
# ==================================================

master = pd.read_csv(
    DATA_DIR / "pds_master_dataset.csv"
)

print("=" * 60)
print("PDS ANOMALY & OUTLIER DETECTION")
print("=" * 60)

print("\nMaster dataset shape:", master.shape)

# ==================================================
# 2. CONSOLIDATE DISTRICT INFORMATION
# ==================================================

master["district_code"] = (
    master["distCode_card"]
    .fillna(master["distCode_trans"])
)

master["district_name"] = (
    master["distName_card"]
    .fillna(master["distName_trans"])
)

# ==================================================
# 3. TRANSACTION AVAILABILITY
# ==================================================

master["transaction_available"] = (
    master["noOfTrans"].notna()
)

# ==================================================
# 4. TOTAL RICE
# ==================================================

master["total_rice"] = (
    master["riceAfsc"].fillna(0)
    + master["riceFsc"].fillna(0)
    + master["riceAap"].fillna(0)
)

# ==================================================
# 5. SHOP-LEVEL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("CREATING SHOP-LEVEL METRICS")
print("=" * 60)

shop_summary = (
    master
    .groupby(
        ["shopNo", "district_code", "district_name"],
        dropna=False
    )
    .agg(
        transaction_months=(
            "transaction_available",
            "sum"
        ),

        total_transactions=(
            "noOfTrans",
            lambda x: x.sum(min_count=1)
        ),

        total_ration_cards=(
            "totalRcs",
            "sum"
        ),

        total_rice=(
            "total_rice",
            "sum"
        ),

        total_wheat=(
            "wheat",
            lambda x: x.sum(min_count=1)
        ),

        total_sugar=(
            "sugar",
            lambda x: x.sum(min_count=1)
        )
    )
    .reset_index()
)

# ==================================================
# 6. SHOP-LEVEL RATIOS
# ==================================================

shop_summary["transactions_per_month"] = (
    shop_summary["total_transactions"]
    / shop_summary["transaction_months"]
)

shop_summary["transactions_per_ration_card"] = (
    shop_summary["total_transactions"]
    / shop_summary["total_ration_cards"]
)

shop_summary["rice_per_transaction"] = (
    shop_summary["total_rice"]
    / shop_summary["total_transactions"]
)

# Replace infinite values caused by division by zero
shop_summary.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

# ==================================================
# 7. REMOVE SHOPS WITHOUT TRANSACTION DATA
# ==================================================

valid_shops = shop_summary[
    shop_summary["transaction_months"] > 0
].copy()

print(
    "\nTotal shops:",
    len(shop_summary)
)

print(
    "Shops with transaction data:",
    len(valid_shops)
)

print(
    "Shops without transaction data:",
    len(shop_summary) - len(valid_shops)
)

# ==================================================
# 8. IQR OUTLIER FUNCTION
# ==================================================

def detect_iqr_outliers(
    df,
    column
):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[
        (df[column] < lower_bound)
        | (df[column] > upper_bound)
    ].copy()

    return (
        outliers,
        q1,
        q3,
        lower_bound,
        upper_bound
    )

# ==================================================
# 9. TRANSACTION OUTLIERS
# ==================================================

print("\n" + "=" * 60)
print("TRANSACTION OUTLIERS")
print("=" * 60)

transaction_outliers, q1, q3, lower, upper = (
    detect_iqr_outliers(
        valid_shops,
        "transactions_per_month"
    )
)

print(f"\nQ1: {q1:.2f}")
print(f"Q3: {q3:.2f}")
print(f"IQR: {(q3 - q1):.2f}")
print(f"Lower bound: {lower:.2f}")
print(f"Upper bound: {upper:.2f}")

print(
    "\nNumber of transaction outlier shops:",
    len(transaction_outliers)
)

# ==================================================
# 10. TOP TRANSACTION OUTLIERS
# ==================================================

print("\n" + "=" * 60)
print("TOP 10 HIGH-TRANSACTION OUTLIER SHOPS")
print("=" * 60)

top_transaction_outliers = (
    transaction_outliers
    .sort_values(
        "transactions_per_month",
        ascending=False
    )
    .head(10)
)

print(
    top_transaction_outliers[
        [
            "shopNo",
            "district_name",
            "transaction_months",
            "total_transactions",
            "transactions_per_month"
        ]
    ].to_string(index=False)
)

# ==================================================
# 11. LOW AND HIGH TRANSACTION OUTLIERS
# ==================================================

print("\n" + "=" * 60)
print("LOW TRANSACTION OUTLIERS")
print("=" * 60)

low_transaction_outliers = valid_shops[
    valid_shops["transactions_per_month"] < lower
].copy()

if len(low_transaction_outliers) > 0:

    low_transaction_outliers = (
        low_transaction_outliers
        .sort_values(
            "transactions_per_month",
            ascending=True
        )
        .head(10)
    )

    print(
        low_transaction_outliers[
            [
                "shopNo",
                "district_name",
                "transaction_months",
                "total_transactions",
                "transactions_per_month"
            ]
        ].to_string(index=False)
    )

else:
    print("\nNo low-transaction outlier shops were detected.")


print("\n" + "=" * 60)
print("HIGH TRANSACTION OUTLIERS")
print("=" * 60)

high_transaction_outliers = valid_shops[
    valid_shops["transactions_per_month"] > upper
].copy()

high_transaction_outliers = (
    high_transaction_outliers
    .sort_values(
        "transactions_per_month",
        ascending=False
    )
    .head(10)
)

print(
    high_transaction_outliers[
        [
            "shopNo",
            "district_name",
            "transaction_months",
            "total_transactions",
            "transactions_per_month"
        ]
    ].to_string(index=False)
)

# ==================================================
# 12. RATION CARD RATIO OUTLIERS
# ==================================================

print("\n" + "=" * 60)
print("TRANSACTIONS / RATION CARD OUTLIERS")
print("=" * 60)

ratio_data = valid_shops[
    valid_shops["total_ration_cards"] > 0
].copy()

ratio_outliers, q1_ratio, q3_ratio, lower_ratio, upper_ratio = (
    detect_iqr_outliers(
        ratio_data,
        "transactions_per_ration_card"
    )
)

print(f"\nQ1: {q1_ratio:.4f}")
print(f"Q3: {q3_ratio:.4f}")
print(f"IQR: {(q3_ratio - q1_ratio):.4f}")
print(f"Lower bound: {lower_ratio:.4f}")
print(f"Upper bound: {upper_ratio:.4f}")

print(
    "\nNumber of ratio outlier shops:",
    len(ratio_outliers)
)

# ==================================================
# 13. TOP RATIO OUTLIERS
# ==================================================

print("\n" + "=" * 60)
print("TOP 10 TRANSACTION/RATION-CARD OUTLIERS")
print("=" * 60)

top_ratio_outliers = (
    ratio_outliers
    .sort_values(
        "transactions_per_ration_card",
        ascending=False
    )
    .head(10)
)

print(
    top_ratio_outliers[
        [
            "shopNo",
            "district_name",
            "transactions_per_ration_card",
            "total_transactions",
            "total_ration_cards"
        ]
    ].to_string(index=False)
)

# ==================================================
# 14. RICE / TRANSACTION OUTLIERS
# ==================================================

print("\n" + "=" * 60)
print("RICE / TRANSACTION OUTLIERS")
print("=" * 60)

rice_data = valid_shops[
    valid_shops["total_transactions"] > 0
].copy()

rice_outliers, q1_rice, q3_rice, lower_rice, upper_rice = (
    detect_iqr_outliers(
        rice_data,
        "rice_per_transaction"
    )
)

print(f"\nQ1: {q1_rice:.2f}")
print(f"Q3: {q3_rice:.2f}")
print(f"IQR: {(q3_rice - q1_rice):.2f}")
print(f"Lower bound: {lower_rice:.2f}")
print(f"Upper bound: {upper_rice:.2f}")

print(
    "\nNumber of rice/transaction outlier shops:",
    len(rice_outliers)
)

# ==================================================
# 15. TOP RICE OUTLIERS
# ==================================================

print("\n" + "=" * 60)
print("TOP 10 RICE/TRANSACTION OUTLIER SHOPS")
print("=" * 60)

top_rice_outliers = (
    rice_outliers
    .sort_values(
        "rice_per_transaction",
        ascending=False
    )
    .head(10)
)

print(
    top_rice_outliers[
        [
            "shopNo",
            "district_name",
            "rice_per_transaction",
            "total_rice",
            "total_transactions"
        ]
    ].to_string(index=False)
)

# ==================================================
# 16. DISTRICT-LEVEL Z-SCORE
# ==================================================

print("\n" + "=" * 60)
print("DISTRICT-LEVEL ANOMALY DETECTION")
print("=" * 60)

district_summary = (
    valid_shops
    .groupby(
        ["district_code", "district_name"],
        dropna=False
    )
    .agg(
        shops=("shopNo", "nunique"),

        avg_transactions_per_month=(
            "transactions_per_month",
            "mean"
        ),

        median_transactions_per_month=(
            "transactions_per_month",
            "median"
        ),

        avg_transactions_per_card=(
            "transactions_per_ration_card",
            "mean"
        ),

        avg_rice_per_transaction=(
            "rice_per_transaction",
            "mean"
        )
    )
    .reset_index()
)

# Calculate z-score for transaction activity
district_mean = (
    district_summary["avg_transactions_per_month"]
    .mean()
)

district_std = (
    district_summary["avg_transactions_per_month"]
    .std()
)

district_summary["transaction_activity_zscore"] = (
    (
        district_summary["avg_transactions_per_month"]
        - district_mean
    )
    / district_std
)

# ==================================================
# 17. HIGH DISTRICT ACTIVITY
# ==================================================

print("\n" + "=" * 60)
print("DISTRICTS WITH HIGH TRANSACTION ACTIVITY")
print("=" * 60)

high_activity = (
    district_summary[
        district_summary["transaction_activity_zscore"] > 2
    ]
    .sort_values(
        "transaction_activity_zscore",
        ascending=False
    )
)

if len(high_activity) > 0:
    print(
        high_activity.to_string(index=False)
    )
else:
    print("\nNo districts exceeded +2 standard deviations.")

# ==================================================
# 18. LOW DISTRICT ACTIVITY
# ==================================================

print("\n" + "=" * 60)
print("DISTRICTS WITH LOW TRANSACTION ACTIVITY")
print("=" * 60)

low_activity = (
    district_summary[
        district_summary["transaction_activity_zscore"] < -2
    ]
    .sort_values(
        "transaction_activity_zscore",
        ascending=True
    )
)

if len(low_activity) > 0:
    print(
        low_activity.to_string(index=False)
    )
else:
    print("\nNo districts were below -2 standard deviations.")

# ==================================================
# 19. SAVE SHOP OUTLIERS
# ==================================================

transaction_outlier_file = (
    OUTPUT_DIR / "transaction_outliers.csv"
)

ratio_outlier_file = (
    OUTPUT_DIR / "ration_card_ratio_outliers.csv"
)

rice_outlier_file = (
    OUTPUT_DIR / "rice_transaction_outliers.csv"
)

district_anomaly_file = (
    OUTPUT_DIR / "district_anomaly_summary.csv"
)

transaction_outliers.to_csv(
    transaction_outlier_file,
    index=False
)

ratio_outliers.to_csv(
    ratio_outlier_file,
    index=False
)

rice_outliers.to_csv(
    rice_outlier_file,
    index=False
)

district_summary.to_csv(
    district_anomaly_file,
    index=False
)

# ==================================================
# 20. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("ANOMALY ANALYSIS COMPLETED")
print("=" * 60)

print("\nOutput files:")
print(transaction_outlier_file)
print(ratio_outlier_file)
print(rice_outlier_file)
print(district_anomaly_file)

print("\n" + "=" * 60)