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
print("COMMODITY DISTRIBUTION ANALYSIS")
print("=" * 60)

print("\nMaster dataset shape:", master.shape)

# ==================================================
# 2. CREATE CONSOLIDATED DISTRICT
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
# 3. CREATE DATE
# ==================================================

master["date"] = pd.to_datetime(
    dict(
        year=master["year"],
        month=master["month"],
        day=1
    ),
    errors="coerce"
)

# ==================================================
# 4. TRANSACTION AVAILABILITY
# ==================================================

master["transaction_available"] = (
    master["noOfTrans"].notna()
)

# ==================================================
# 5. CREATE TOTAL RICE
# ==================================================

master["total_rice"] = (
    master["riceAfsc"].fillna(0)
    + master["riceFsc"].fillna(0)
    + master["riceAap"].fillna(0)
)

# ==================================================
# 6. OVERALL COMMODITY SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("OVERALL COMMODITY DISTRIBUTION")
print("=" * 60)

commodity_summary = pd.DataFrame({
    "commodity": [
        "Rice AFSC",
        "Rice FSC",
        "Rice AAP",
        "Total Rice",
        "Wheat",
        "Sugar",
        "RGDAL",
        "Kerosene",
        "Salt",
        "Other Shop Transactions"
    ],
    "quantity": [
        master["riceAfsc"].sum(min_count=1),
        master["riceFsc"].sum(min_count=1),
        master["riceAap"].sum(min_count=1),
        master["total_rice"].sum(),
        master["wheat"].sum(min_count=1),
        master["sugar"].sum(min_count=1),
        master["rgdal"].sum(min_count=1),
        master["kerosene"].sum(min_count=1),
        master["salt"].sum(min_count=1),
        master["otherShopTransCnt"].sum(min_count=1)
    ]
})

print(
    commodity_summary.to_string(index=False)
)

# ==================================================
# 7. DISTRICT-LEVEL COMMODITY SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("DISTRICT-LEVEL COMMODITY SUMMARY")
print("=" * 60)

district_commodity = (
    master
    .groupby(
        ["district_code", "district_name"],
        dropna=False
    )
    .agg(
        total_rice=(
            "total_rice",
            "sum"
        ),

        rice_afsc=(
            "riceAfsc",
            lambda x: x.sum(min_count=1)
        ),

        rice_fsc=(
            "riceFsc",
            lambda x: x.sum(min_count=1)
        ),

        rice_aap=(
            "riceAap",
            lambda x: x.sum(min_count=1)
        ),

        wheat=(
            "wheat",
            lambda x: x.sum(min_count=1)
        ),

        sugar=(
            "sugar",
            lambda x: x.sum(min_count=1)
        ),

        kerosene=(
            "kerosene",
            lambda x: x.sum(min_count=1)
        ),

        salt=(
            "salt",
            lambda x: x.sum(min_count=1)
        ),

        total_transactions=(
            "noOfTrans",
            lambda x: x.sum(min_count=1)
        ),

        total_ration_cards=(
            "totalRcs",
            "sum"
        )
    )
    .reset_index()
)

print(
    district_commodity
    .sort_values(
        "total_rice",
        ascending=False
    )
    .head(15)
    .to_string(index=False)
)

# ==================================================
# 8. TOP 10 DISTRICTS BY RICE
# ==================================================

print("\n" + "=" * 60)
print("TOP 10 DISTRICTS BY TOTAL RICE")
print("=" * 60)

top_rice_districts = (
    district_commodity
    .sort_values(
        "total_rice",
        ascending=False
    )
    .head(10)
)

print(
    top_rice_districts[
        [
            "district_name",
            "total_rice",
            "total_transactions",
            "total_ration_cards"
        ]
    ].to_string(index=False)
)

# ==================================================
# 9. TOP 10 DISTRICTS BY WHEAT
# ==================================================

print("\n" + "=" * 60)
print("TOP 10 DISTRICTS BY WHEAT")
print("=" * 60)

top_wheat_districts = (
    district_commodity
    .sort_values(
        "wheat",
        ascending=False
    )
    .head(10)
)

print(
    top_wheat_districts[
        [
            "district_name",
            "wheat",
            "total_transactions",
            "total_ration_cards"
        ]
    ].to_string(index=False)
)

# ==================================================
# 10. TOP 10 DISTRICTS BY SUGAR
# ==================================================

print("\n" + "=" * 60)
print("TOP 10 DISTRICTS BY SUGAR")
print("=" * 60)

top_sugar_districts = (
    district_commodity
    .sort_values(
        "sugar",
        ascending=False
    )
    .head(10)
)

print(
    top_sugar_districts[
        [
            "district_name",
            "sugar",
            "total_transactions",
            "total_ration_cards"
        ]
    ].to_string(index=False)
)

# ==================================================
# 11. MONTHLY COMMODITY TREND
# ==================================================

print("\n" + "=" * 60)
print("MONTHLY COMMODITY TREND")
print("=" * 60)

monthly_commodity = (
    master
    .groupby("date")
    .agg(
        total_rice=(
            "total_rice",
            "sum"
        ),

        rice_afsc=(
            "riceAfsc",
            lambda x: x.sum(min_count=1)
        ),

        rice_fsc=(
            "riceFsc",
            lambda x: x.sum(min_count=1)
        ),

        rice_aap=(
            "riceAap",
            lambda x: x.sum(min_count=1)
        ),

        wheat=(
            "wheat",
            lambda x: x.sum(min_count=1)
        ),

        sugar=(
            "sugar",
            lambda x: x.sum(min_count=1)
        ),

        kerosene=(
            "kerosene",
            lambda x: x.sum(min_count=1)
        ),

        salt=(
            "salt",
            lambda x: x.sum(min_count=1)
        ),

        transaction_records=(
            "transaction_available",
            "sum"
        )
    )
    .reset_index()
)

print(
    monthly_commodity.to_string(index=False)
)

# ==================================================
# 12. TOP 5 MONTHS BY RICE
# ==================================================

print("\n" + "=" * 60)
print("TOP 5 MONTHS BY TOTAL RICE")
print("=" * 60)

valid_months = monthly_commodity[
    monthly_commodity["transaction_records"] > 0
]

print(
    valid_months[
        [
            "date",
            "total_rice",
            "wheat",
            "sugar"
        ]
    ]
    .sort_values(
        "total_rice",
        ascending=False
    )
    .head(5)
    .to_string(index=False)
)

# ==================================================
# 13. COMMODITY SHARE
# ==================================================

print("\n" + "=" * 60)
print("RICE COMPONENT SHARE")
print("=" * 60)

rice_afsc_total = master["riceAfsc"].sum()
rice_fsc_total = master["riceFsc"].sum()
rice_aap_total = master["riceAap"].sum()

rice_total = (
    rice_afsc_total
    + rice_fsc_total
    + rice_aap_total
)

rice_share = pd.DataFrame({
    "rice_category": [
        "Rice AFSC",
        "Rice FSC",
        "Rice AAP"
    ],
    "quantity": [
        rice_afsc_total,
        rice_fsc_total,
        rice_aap_total
    ]
})

rice_share["percentage"] = (
    rice_share["quantity"]
    / rice_total
    * 100
)

print(
    rice_share.to_string(index=False)
)

# ==================================================
# 14. RICE PER TRANSACTION
# ==================================================

district_commodity["rice_per_transaction"] = (
    district_commodity["total_rice"]
    / district_commodity["total_transactions"]
)

print("\n" + "=" * 60)
print("TOP 10 DISTRICTS BY RICE PER TRANSACTION")
print("=" * 60)

print(
    district_commodity[
        district_commodity["total_transactions"] > 0
    ][
        [
            "district_name",
            "rice_per_transaction",
            "total_rice",
            "total_transactions"
        ]
    ]
    .sort_values(
        "rice_per_transaction",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)

# ==================================================
# 15. SAVE OVERALL COMMODITY SUMMARY
# ==================================================

commodity_file = (
    OUTPUT_DIR / "commodity_summary.csv"
)

commodity_summary.to_csv(
    commodity_file,
    index=False
)

# ==================================================
# 16. SAVE DISTRICT COMMODITY SUMMARY
# ==================================================

district_file = (
    OUTPUT_DIR / "district_commodity_summary.csv"
)

district_commodity.to_csv(
    district_file,
    index=False
)

# ==================================================
# 17. SAVE MONTHLY COMMODITY SUMMARY
# ==================================================

monthly_file = (
    OUTPUT_DIR / "monthly_commodity_summary.csv"
)

monthly_commodity.to_csv(
    monthly_file,
    index=False
)

# ==================================================
# 18. SAVE RICE SHARE
# ==================================================

rice_share_file = (
    OUTPUT_DIR / "rice_component_share.csv"
)

rice_share.to_csv(
    rice_share_file,
    index=False
)

# ==================================================
# 19. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("COMMODITY ANALYSIS COMPLETED")
print("=" * 60)

print("\nTotal rice:", rice_total)
print("Total wheat:", master["wheat"].sum(min_count=1))
print("Total sugar:", master["sugar"].sum(min_count=1))

print("\nOutput files:")

print(commodity_file)
print(district_file)
print(monthly_file)
print(rice_share_file)

print("\n" + "=" * 60)