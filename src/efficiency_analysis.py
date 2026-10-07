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
print("RATION CARD & TRANSACTION EFFICIENCY ANALYSIS")
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
# 3. CREATE TOTAL RICE
# ==================================================

master["total_rice"] = (
    master["riceAfsc"].fillna(0)
    + master["riceFsc"].fillna(0)
    + master["riceAap"].fillna(0)
)

# ==================================================
# 4. TRANSACTION AVAILABILITY
# ==================================================

master["transaction_available"] = (
    master["noOfTrans"].notna()
)

# ==================================================
# 5. DISTRICT-LEVEL EFFICIENCY
# ==================================================

print("\n" + "=" * 60)
print("DISTRICT-LEVEL EFFICIENCY")
print("=" * 60)

district_efficiency = (
    master
    .groupby(
        ["district_code", "district_name"],
        dropna=False
    )
    .agg(
        shops=(
            "shopNo",
            "nunique"
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
# 6. TRANSACTIONS PER RATION CARD
# ==================================================

district_efficiency["transactions_per_ration_card"] = (
    district_efficiency["total_transactions"]
    / district_efficiency["total_ration_cards"]
)

# ==================================================
# 7. RICE PER TRANSACTION
# ==================================================

district_efficiency["rice_per_transaction"] = (
    district_efficiency["total_rice"]
    / district_efficiency["total_transactions"]
)

# ==================================================
# 8. TRANSACTIONS PER SHOP-MONTH
# ==================================================

district_efficiency["transactions_per_shop_month"] = (
    district_efficiency["total_transactions"]
    / district_efficiency["transaction_months"]
)

# ==================================================
# 9. DISPLAY FULL SUMMARY
# ==================================================

print(
    district_efficiency[
        [
            "district_name",
            "shops",
            "transaction_months",
            "total_transactions",
            "total_ration_cards",
            "total_rice",
            "transactions_per_ration_card",
            "rice_per_transaction",
            "transactions_per_shop_month"
        ]
    ]
    .sort_values(
        "transactions_per_ration_card",
        ascending=False
    )
    .to_string(index=False)
)

# ==================================================
# 10. TOP DISTRICTS BY TRANSACTIONS PER RATION CARD
# ==================================================

print("\n" + "=" * 60)
print("TOP 10 DISTRICTS BY TRANSACTIONS PER RATION CARD")
print("=" * 60)

top_transaction_card = (
    district_efficiency[
        district_efficiency["total_ration_cards"] > 0
    ]
    .sort_values(
        "transactions_per_ration_card",
        ascending=False
    )
    .head(10)
)

print(
    top_transaction_card[
        [
            "district_name",
            "transactions_per_ration_card",
            "total_transactions",
            "total_ration_cards"
        ]
    ].to_string(index=False)
)

# ==================================================
# 11. LOWEST DISTRICTS BY TRANSACTIONS PER RATION CARD
# ==================================================

print("\n" + "=" * 60)
print("BOTTOM 10 DISTRICTS BY TRANSACTIONS PER RATION CARD")
print("=" * 60)

bottom_transaction_card = (
    district_efficiency[
        district_efficiency["total_ration_cards"] > 0
    ]
    .sort_values(
        "transactions_per_ration_card",
        ascending=True
    )
    .head(10)
)

print(
    bottom_transaction_card[
        [
            "district_name",
            "transactions_per_ration_card",
            "total_transactions",
            "total_ration_cards"
        ]
    ].to_string(index=False)
)

# ==================================================
# 12. TOP DISTRICTS BY TRANSACTIONS PER SHOP-MONTH
# ==================================================

print("\n" + "=" * 60)
print("TOP 10 DISTRICTS BY TRANSACTIONS PER SHOP-MONTH")
print("=" * 60)

top_shop_month = (
    district_efficiency[
        district_efficiency["transaction_months"] > 0
    ]
    .sort_values(
        "transactions_per_shop_month",
        ascending=False
    )
    .head(10)
)

print(
    top_shop_month[
        [
            "district_name",
            "transactions_per_shop_month",
            "shops",
            "transaction_months",
            "total_transactions"
        ]
    ].to_string(index=False)
)

# ==================================================
# 13. LOWEST DISTRICTS BY TRANSACTIONS PER SHOP-MONTH
# ==================================================

print("\n" + "=" * 60)
print("BOTTOM 10 DISTRICTS BY TRANSACTIONS PER SHOP-MONTH")
print("=" * 60)

bottom_shop_month = (
    district_efficiency[
        district_efficiency["transaction_months"] > 0
    ]
    .sort_values(
        "transactions_per_shop_month",
        ascending=True
    )
    .head(10)
)

print(
    bottom_shop_month[
        [
            "district_name",
            "transactions_per_shop_month",
            "shops",
            "transaction_months",
            "total_transactions"
        ]
    ].to_string(index=False)
)

# ==================================================
# 14. TOP DISTRICTS BY RICE PER TRANSACTION
# ==================================================

print("\n" + "=" * 60)
print("TOP 10 DISTRICTS BY RICE PER TRANSACTION")
print("=" * 60)

top_rice_efficiency = (
    district_efficiency[
        district_efficiency["total_transactions"] > 0
    ]
    .sort_values(
        "rice_per_transaction",
        ascending=False
    )
    .head(10)
)

print(
    top_rice_efficiency[
        [
            "district_name",
            "rice_per_transaction",
            "total_rice",
            "total_transactions"
        ]
    ].to_string(index=False)
)

# ==================================================
# 15. OVERALL METRICS
# ==================================================

total_transactions = master["noOfTrans"].sum(
    min_count=1
)

total_ration_cards = master["totalRcs"].sum()

total_rice = master["total_rice"].sum()

transaction_card_ratio = (
    total_transactions
    / total_ration_cards
)

rice_transaction_ratio = (
    total_rice
    / total_transactions
)

print("\n" + "=" * 60)
print("OVERALL EFFICIENCY METRICS")
print("=" * 60)

print(
    f"\nTotal transactions: {total_transactions:,.0f}"
)

print(
    f"Total ration cards recorded: {total_ration_cards:,.0f}"
)

print(
    f"Transactions per ration card: "
    f"{transaction_card_ratio:.4f}"
)

print(
    f"Total recorded rice: {total_rice:,.0f}"
)

print(
    f"Rice per transaction: "
    f"{rice_transaction_ratio:.4f}"
)

# ==================================================
# 16. IDENTIFY POTENTIAL HIGH-RATIO DISTRICTS
# ==================================================

print("\n" + "=" * 60)
print("POTENTIAL HIGH TRANSACTION/CARD RATIO DISTRICTS")
print("=" * 60)

ratio_mean = (
    district_efficiency[
        district_efficiency["total_ration_cards"] > 0
    ]["transactions_per_ration_card"]
    .mean()
)

ratio_std = (
    district_efficiency[
        district_efficiency["total_ration_cards"] > 0
    ]["transactions_per_ration_card"]
    .std()
)

high_ratio_threshold = ratio_mean + (2 * ratio_std)

high_ratio_districts = (
    district_efficiency[
        district_efficiency["transactions_per_ration_card"]
        > high_ratio_threshold
    ]
    .sort_values(
        "transactions_per_ration_card",
        ascending=False
    )
)

print(
    f"\nAverage ratio: {ratio_mean:.4f}"
)

print(
    f"Standard deviation: {ratio_std:.4f}"
)

print(
    f"High-ratio threshold: {high_ratio_threshold:.4f}"
)

if len(high_ratio_districts) > 0:
    print(
        high_ratio_districts[
            [
                "district_name",
                "transactions_per_ration_card",
                "total_transactions",
                "total_ration_cards"
            ]
        ].to_string(index=False)
    )
else:
    print("\nNo districts exceeded the high-ratio threshold.")

# ==================================================
# 17. SAVE DISTRICT EFFICIENCY
# ==================================================

district_output = (
    OUTPUT_DIR / "district_efficiency_summary.csv"
)

district_efficiency.to_csv(
    district_output,
    index=False
)

# ==================================================
# 18. SAVE TOP/BOTTOM RATIO DISTRICTS
# ==================================================

top_output = (
    OUTPUT_DIR / "top_transaction_card_districts.csv"
)

bottom_output = (
    OUTPUT_DIR / "bottom_transaction_card_districts.csv"
)

top_transaction_card.to_csv(
    top_output,
    index=False
)

bottom_transaction_card.to_csv(
    bottom_output,
    index=False
)

# ==================================================
# 19. FINAL MESSAGE
# ==================================================

print("\n" + "=" * 60)
print("EFFICIENCY ANALYSIS COMPLETED")
print("=" * 60)

print("\nOutput files:")
print(district_output)
print(top_output)
print(bottom_output)

print("\n" + "=" * 60)