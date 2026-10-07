import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


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
print("PDS ISOLATION FOREST ANOMALY DETECTION")
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
# 5. CREATE SHOP-LEVEL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("CREATING SHOP-LEVEL FEATURES")
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
        )
    )
    .reset_index()
)


# ==================================================
# 6. CREATE FEATURES
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


# ==================================================
# 7. CLEAN INFINITE VALUES
# ==================================================

shop_summary.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)


# ==================================================
# 8. SELECT VALID SHOPS
# ==================================================

valid_shops = shop_summary[
    shop_summary["transaction_months"] > 0
].copy()


print("\nTotal shops:", len(shop_summary))

print(
    "Shops with transaction data:",
    len(valid_shops)
)

print(
    "Shops without transaction data:",
    len(shop_summary) - len(valid_shops)
)


# ==================================================
# 9. SELECT FEATURES FOR ANOMALY DETECTION
# ==================================================

features = [
    "transactions_per_month",
    "transactions_per_ration_card",
    "rice_per_transaction",
    "total_ration_cards",
    "total_transactions"
]

model_data = valid_shops[
    ["shopNo", "district_code", "district_name"] + features
].copy()


# ==================================================
# 10. REMOVE MISSING VALUES
# ==================================================

model_data = model_data.dropna(
    subset=features
).copy()


print(
    "\nShops used for Isolation Forest:",
    len(model_data)
)


# ==================================================
# 11. ROBUST SCALING
# ==================================================

print("\n" + "=" * 60)
print("FEATURE SCALING")
print("=" * 60)

scaler = RobustScaler()

X_scaled = scaler.fit_transform(
    model_data[features]
)


# ==================================================
# 12. ISOLATION FOREST
# ==================================================

print("\n" + "=" * 60)
print("RUNNING ISOLATION FOREST")
print("=" * 60)

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

model.fit(X_scaled)


# ==================================================
# 13. PREDICT ANOMALIES
# ==================================================

model_data["anomaly_prediction"] = (
    model.predict(X_scaled)
)

model_data["anomaly_score"] = (
    model.decision_function(X_scaled)
)


# -1 = anomaly
#  1 = normal

model_data["anomaly_label"] = np.where(
    model_data["anomaly_prediction"] == -1,
    "Anomaly",
    "Normal"
)


# ==================================================
# 14. ANOMALY SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("ANOMALY SUMMARY")
print("=" * 60)

anomaly_count = (
    model_data["anomaly_prediction"] == -1
).sum()

normal_count = (
    model_data["anomaly_prediction"] == 1
).sum()

print("\nNormal shops:", normal_count)

print("Anomalous shops:", anomaly_count)

print(
    "Anomaly percentage:",
    round(
        anomaly_count / len(model_data) * 100,
        2
    ),
    "%"
)


# ==================================================
# 15. TOP ANOMALOUS SHOPS
# ==================================================

print("\n" + "=" * 60)
print("TOP 20 ANOMALOUS SHOPS")
print("=" * 60)

top_anomalies = (
    model_data[
        model_data["anomaly_prediction"] == -1
    ]
    .sort_values(
        "anomaly_score",
        ascending=True
    )
    .head(20)
)


print(
    top_anomalies[
        [
            "shopNo",
            "district_name",
            "transactions_per_month",
            "transactions_per_ration_card",
            "rice_per_transaction",
            "total_ration_cards",
            "total_transactions",
            "anomaly_score"
        ]
    ].to_string(index=False)
)


# ==================================================
# 16. DISTRICT ANOMALY SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("DISTRICT-LEVEL ISOLATION FOREST SUMMARY")
print("=" * 60)

district_anomaly_summary = (
    model_data
    .groupby("district_name")
    .agg(
        shops=("shopNo", "count"),

        anomalous_shops=(
            "anomaly_prediction",
            lambda x: (x == -1).sum()
        )
    )
    .reset_index()
)

district_anomaly_summary["anomaly_percentage"] = (
    district_anomaly_summary["anomalous_shops"]
    / district_anomaly_summary["shops"]
    * 100
)


district_anomaly_summary = (
    district_anomaly_summary
    .sort_values(
        "anomaly_percentage",
        ascending=False
    )
)


print(
    district_anomaly_summary.head(10)
    .to_string(index=False)
)


# ==================================================
# 17. SAVE RESULTS
# ==================================================

output_file = (
    OUTPUT_DIR /
    "isolation_forest_shop_anomalies.csv"
)

district_output_file = (
    OUTPUT_DIR /
    "isolation_forest_district_summary.csv"
)


model_data.to_csv(
    output_file,
    index=False
)

district_anomaly_summary.to_csv(
    district_output_file,
    index=False
)


# ==================================================
# 18. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("ISOLATION FOREST ANALYSIS COMPLETED")
print("=" * 60)

print("\nOutput files:")

print(output_file)

print(district_output_file)

print("\n" + "=" * 60)