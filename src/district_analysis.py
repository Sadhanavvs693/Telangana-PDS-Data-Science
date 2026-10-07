import pandas as pd
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
print("DISTRICT-WISE PDS ANALYSIS")
print("=" * 60)

print("\nMaster dataset shape:", master.shape)

# ==================================================
# 2. CREATE CONSOLIDATED DISTRICT COLUMNS
# ==================================================

# Card Status contains district information for most records.
# Transaction-only records will use Transaction district information.

master["district_code"] = master["distCode_card"].fillna(
    master["distCode_trans"]
)

master["district_name"] = master["distName_card"].fillna(
    master["distName_trans"]
)

# ==================================================
# 3. CHECK DISTRICT DATA
# ==================================================

print("\nUnique districts:", master["district_name"].nunique())

print("\nDistricts with missing names:")
print(master["district_name"].isna().sum())

# ==================================================
# 4. CREATE DISTRICT SUMMARY
# ==================================================

district_summary = (
    master
    .groupby(
        ["district_code", "district_name"],
        dropna=True
    )
    .agg(
        total_shops=("shopNo", "nunique"),
        total_records=("shopNo", "size"),
        total_transactions=("noOfTrans", "sum"),
        total_ration_cards=("totalRcs", "sum"),
        total_rice_afsc=("riceAfsc", "sum"),
        total_rice_fsc=("riceFsc", "sum"),
        total_rice_aap=("riceAap", "sum"),
        total_wheat=("wheat", "sum"),
        total_sugar=("sugar", "sum"),
        total_transaction_amount=("totalAmount", "sum")
    )
    .reset_index()
)

# ==================================================
# 5. CALCULATE TOTAL RICE
# ==================================================

district_summary["total_rice"] = (
    district_summary["total_rice_afsc"]
    + district_summary["total_rice_fsc"]
    + district_summary["total_rice_aap"]
)

# ==================================================
# 6. CALCULATE AVERAGE TRANSACTIONS PER SHOP
# ==================================================

district_summary["avg_transactions_per_shop"] = (
    district_summary["total_transactions"]
    / district_summary["total_shops"]
)

# ==================================================
# 7. SORT BY TOTAL TRANSACTIONS
# ==================================================

district_summary = district_summary.sort_values(
    "total_transactions",
    ascending=False
)

# ==================================================
# 8. TOP 10 DISTRICTS BY TRANSACTIONS
# ==================================================

print("\n" + "=" * 60)
print("TOP 10 DISTRICTS BY TOTAL TRANSACTIONS")
print("=" * 60)

print(
    district_summary[
        [
            "district_name",
            "total_shops",
            "total_transactions",
            "total_ration_cards",
            "total_transaction_amount",
            "avg_transactions_per_shop"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

# ==================================================
# 9. TOP 10 DISTRICTS BY RICE DISTRIBUTION
# ==================================================

print("\n" + "=" * 60)
print("TOP 10 DISTRICTS BY RICE DISTRIBUTION")
print("=" * 60)

rice_top10 = (
    district_summary
    .sort_values(
        "total_rice",
        ascending=False
    )
    .head(10)
)

print(
    rice_top10[
        [
            "district_name",
            "total_rice",
            "total_wheat",
            "total_sugar"
        ]
    ].to_string(index=False)
)

# ==================================================
# 10. TOP 10 DISTRICTS BY TRANSACTIONS PER SHOP
# ==================================================

print("\n" + "=" * 60)
print("TOP 10 DISTRICTS BY AVERAGE TRANSACTIONS PER SHOP")
print("=" * 60)

print(
    district_summary[
        [
            "district_name",
            "total_shops",
            "total_transactions",
            "avg_transactions_per_shop"
        ]
    ]
    .sort_values(
        "avg_transactions_per_shop",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)

# ==================================================
# 11. SAVE DISTRICT SUMMARY
# ==================================================

output_file = OUTPUT_DIR / "district_summary.csv"

district_summary.to_csv(
    output_file,
    index=False
)

# ==================================================
# 12. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("DISTRICT ANALYSIS COMPLETED")
print("=" * 60)

print("\nDistricts analyzed:", len(district_summary))

print("\nOutput file:")
print(output_file)

print("\nAnalysis columns:")
print(district_summary.columns.tolist())

print("\n" + "=" * 60)