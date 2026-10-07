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
print("SHOP PERFORMANCE ANALYSIS")
print("=" * 60)

print("\nMaster dataset shape:", master.shape)

# ==================================================
# 2. CREATE CONSOLIDATED DISTRICT INFORMATION
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
# 3. IDENTIFY TRANSACTION DATA AVAILABILITY
# ==================================================

master["transaction_available"] = (
    master["noOfTrans"].notna()
)

# ==================================================
# 4. SHOP-LEVEL SUMMARY
# ==================================================

shop_summary = (
    master
    .groupby(
        [
            "shopNo",
            "district_code",
            "district_name"
        ],
        dropna=False
    )
    .agg(
        months_present=(
            "date_card",
            "count"
        ),

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

        total_rice_afsc=(
            "riceAfsc",
            lambda x: x.sum(min_count=1)
        ),

        total_rice_fsc=(
            "riceFsc",
            lambda x: x.sum(min_count=1)
        ),

        total_rice_aap=(
            "riceAap",
            lambda x: x.sum(min_count=1)
        ),

        total_wheat=(
            "wheat",
            lambda x: x.sum(min_count=1)
        ),

        total_sugar=(
            "sugar",
            lambda x: x.sum(min_count=1)
        ),

        total_transaction_amount=(
            "totalAmount",
            lambda x: x.sum(min_count=1)
        ),

        avg_monthly_transactions=(
            "noOfTrans",
            "mean"
        ),

        avg_monthly_ration_cards=(
            "totalRcs",
            "mean"
        ),

        fps_status=(
            "fpsStatus",
            "first"
        ),

        fps_type=(
            "fpsType",
            "first"
        ),

        address=(
            "address",
            "first"
        ),

        latitude=(
            "latitude",
            "first"
        ),

        longitude=(
            "longitude",
            "first"
        )
    )
    .reset_index()
)

# ==================================================
# 5. CALCULATE TOTAL RICE
# ==================================================

shop_summary["total_rice"] = (
    shop_summary["total_rice_afsc"]
    + shop_summary["total_rice_fsc"]
    + shop_summary["total_rice_aap"]
)

# ==================================================
# 6. CALCULATE TRANSACTIONS PER RATION CARD
# ==================================================

shop_summary["transactions_per_ration_card"] = (
    shop_summary["total_transactions"]
    / shop_summary["total_ration_cards"]
)

# ==================================================
# 7. CALCULATE TRANSACTIONS PER AVAILABLE MONTH
# ==================================================

shop_summary["transactions_per_transaction_month"] = (
    shop_summary["total_transactions"]
    / shop_summary["transaction_months"]
)

# ==================================================
# 8. DISPLAY BASIC SHOP SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("SHOP SUMMARY")
print("=" * 60)

print(
    "Unique shops:",
    shop_summary["shopNo"].nunique()
)

print(
    "Districts:",
    shop_summary["district_name"].nunique()
)

print(
    "Average transactions/shop:",
    round(
        shop_summary["total_transactions"].mean(),
        2
    )
)

print(
    "Median transactions/shop:",
    round(
        shop_summary["total_transactions"].median(),
        2
    )
)

# ==================================================
# 9. TOP 10 SHOPS BY TOTAL TRANSACTIONS
# ==================================================

print("\n" + "=" * 60)
print("TOP 10 SHOPS BY TOTAL TRANSACTIONS")
print("=" * 60)

top_shops = (
    shop_summary[
        shop_summary["total_transactions"].notna()
    ]
    .sort_values(
        "total_transactions",
        ascending=False
    )
    .head(10)
)

print(
    top_shops[
        [
            "shopNo",
            "district_name",
            "transaction_months",
            "total_transactions",
            "total_ration_cards",
            "total_rice",
            "avg_monthly_transactions"
        ]
    ].to_string(index=False)
)

# ==================================================
# 10. BOTTOM 10 SHOPS BY TOTAL TRANSACTIONS
# ==================================================

print("\n" + "=" * 60)
print("BOTTOM 10 SHOPS BY TOTAL TRANSACTIONS")
print("=" * 60)

bottom_shops = (
    shop_summary[
        shop_summary["total_transactions"].notna()
    ]
    .sort_values(
        "total_transactions",
        ascending=True
    )
    .head(10)
)

print(
    bottom_shops[
        [
            "shopNo",
            "district_name",
            "transaction_months",
            "total_transactions",
            "total_ration_cards",
            "total_rice",
            "avg_monthly_transactions"
        ]
    ].to_string(index=False)
)

# ==================================================
# 11. TOP 10 SHOPS BY RICE DISTRIBUTION
# ==================================================

print("\n" + "=" * 60)
print("TOP 10 SHOPS BY TOTAL RICE")
print("=" * 60)

top_rice_shops = (
    shop_summary[
        shop_summary["total_rice"].notna()
    ]
    .sort_values(
        "total_rice",
        ascending=False
    )
    .head(10)
)

print(
    top_rice_shops[
        [
            "shopNo",
            "district_name",
            "total_rice",
            "total_transactions",
            "total_ration_cards"
        ]
    ].to_string(index=False)
)

# ==================================================
# 12. TOP 10 SHOPS BY TRANSACTIONS PER MONTH
# ==================================================

print("\n" + "=" * 60)
print("TOP 10 SHOPS BY AVERAGE MONTHLY TRANSACTIONS")
print("=" * 60)

top_avg_shops = (
    shop_summary[
        shop_summary["avg_monthly_transactions"].notna()
    ]
    .sort_values(
        "avg_monthly_transactions",
        ascending=False
    )
    .head(10)
)

print(
    top_avg_shops[
        [
            "shopNo",
            "district_name",
            "transaction_months",
            "total_transactions",
            "avg_monthly_transactions"
        ]
    ].to_string(index=False)
)

# ==================================================
# 13. SHOPS WITH NO TRANSACTION DATA
# ==================================================

print("\n" + "=" * 60)
print("SHOPS WITH NO TRANSACTION DATA")
print("=" * 60)

no_transaction_shops = shop_summary[
    shop_summary["transaction_months"] == 0
]

print(
    "Number of shops with no transaction records:",
    len(no_transaction_shops)
)

if len(no_transaction_shops) > 0:
    print(
        no_transaction_shops[
            [
                "shopNo",
                "district_name",
                "months_present",
                "fps_status"
            ]
        ].head(20).to_string(index=False)
    )

# ==================================================
# 14. FPS STATUS SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("FPS STATUS SUMMARY")
print("=" * 60)

fps_status_summary = (
    shop_summary["fps_status"]
    .fillna("Unknown")
    .value_counts()
)

print(
    fps_status_summary.to_string()
)

# ==================================================
# 15. FPS TYPE SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("FPS TYPE SUMMARY")
print("=" * 60)

fps_type_summary = (
    shop_summary["fps_type"]
    .fillna("Unknown")
    .value_counts()
)

print(
    fps_type_summary.to_string()
)

# ==================================================
# 16. SAVE SHOP SUMMARY
# ==================================================

shop_file = OUTPUT_DIR / "shop_summary.csv"

shop_summary.to_csv(
    shop_file,
    index=False
)

# ==================================================
# 17. SAVE TOP SHOP REPORT
# ==================================================

top_shop_file = OUTPUT_DIR / "top_10_shops.csv"

top_shops.to_csv(
    top_shop_file,
    index=False
)

# ==================================================
# 18. SAVE LOW-PERFORMING SHOP REPORT
# ==================================================

bottom_shop_file = OUTPUT_DIR / "bottom_10_shops.csv"

bottom_shops.to_csv(
    bottom_shop_file,
    index=False
)

# ==================================================
# 19. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("SHOP PERFORMANCE ANALYSIS COMPLETED")
print("=" * 60)

print("\nShop summary:")
print(shop_file)

print("\nTop 10 shops:")
print(top_shop_file)

print("\nBottom 10 shops:")
print(bottom_shop_file)

print("\nTotal shops analyzed:", len(shop_summary))

print("=" * 60)