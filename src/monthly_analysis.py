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
print("MONTHLY PDS ANALYSIS")
print("=" * 60)

print("\nMaster dataset shape:", master.shape)

# ==================================================
# 2. CREATE DATE COLUMN
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
# 3. MONTHLY SUMMARY
# ==================================================

monthly_summary = (
    master
    .groupby("date")
    .agg(
        total_shops=("shopNo", "nunique"),
        total_records=("shopNo", "size"),
        total_transactions=("noOfTrans", "sum"),
        total_ration_cards=("totalRcs", "sum"),
        rice_afsc=("riceAfsc", "sum"),
        rice_fsc=("riceFsc", "sum"),
        rice_aap=("riceAap", "sum"),
        wheat=("wheat", "sum"),
        sugar=("sugar", "sum"),
        total_transaction_amount=("totalAmount", "sum")
    )
    .reset_index()
)

# ==================================================
# 4. TOTAL RICE
# ==================================================

monthly_summary["total_rice"] = (
    monthly_summary["rice_afsc"]
    + monthly_summary["rice_fsc"]
    + monthly_summary["rice_aap"]
)

# ==================================================
# 5. AVERAGE TRANSACTIONS PER SHOP
# ==================================================

monthly_summary["avg_transactions_per_shop"] = (
    monthly_summary["total_transactions"]
    / monthly_summary["total_shops"]
)

# ==================================================
# 6. ADD YEAR AND MONTH
# ==================================================

monthly_summary["year"] = (
    monthly_summary["date"].dt.year
)

monthly_summary["month"] = (
    monthly_summary["date"].dt.month
)

monthly_summary["month_name"] = (
    monthly_summary["date"].dt.strftime("%B")
)

# ==================================================
# 7. DISPLAY MONTHLY SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("MONTHLY TRANSACTION TREND")
print("=" * 60)

print(
    monthly_summary[
        [
            "date",
            "total_shops",
            "total_transactions",
            "total_ration_cards",
            "total_rice",
            "total_transaction_amount"
        ]
    ].to_string(index=False)
)

# ==================================================
# 8. HIGHEST TRANSACTION MONTHS
# ==================================================

print("\n" + "=" * 60)
print("TOP 5 MONTHS BY TRANSACTIONS")
print("=" * 60)

print(
    monthly_summary[
        [
            "date",
            "total_transactions",
            "total_rice",
            "total_transaction_amount"
        ]
    ]
    .sort_values(
        "total_transactions",
        ascending=False
    )
    .head(5)
    .to_string(index=False)
)

# ==================================================
# 9. LOWEST TRANSACTION MONTHS
# ==================================================

print("\n" + "=" * 60)
print("BOTTOM 5 MONTHS BY TRANSACTIONS")
print("=" * 60)

print(
    monthly_summary[
        [
            "date",
            "total_transactions",
            "total_rice",
            "total_transaction_amount"
        ]
    ]
    .sort_values(
        "total_transactions",
        ascending=True
    )
    .head(5)
    .to_string(index=False)
)

# ==================================================
# 10. YEARLY SUMMARY
# ==================================================

yearly_summary = (
    monthly_summary
    .groupby("year")
    .agg(
        total_transactions=("total_transactions", "sum"),
        total_rice=("total_rice", "sum"),
        total_ration_cards=("total_ration_cards", "sum"),
        total_transaction_amount=("total_transaction_amount", "sum")
    )
    .reset_index()
)

print("\n" + "=" * 60)
print("YEARLY SUMMARY")
print("=" * 60)

print(
    yearly_summary.to_string(index=False)
)

# ==================================================
# 11. SAVE MONTHLY SUMMARY
# ==================================================

monthly_file = OUTPUT_DIR / "monthly_summary.csv"

monthly_summary.to_csv(
    monthly_file,
    index=False
)

# ==================================================
# 12. SAVE YEARLY SUMMARY
# ==================================================

yearly_file = OUTPUT_DIR / "yearly_summary.csv"

yearly_summary.to_csv(
    yearly_file,
    index=False
)

# ==================================================
# 13. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("MONTHLY ANALYSIS COMPLETED")
print("=" * 60)

print("\nMonths analyzed:", len(monthly_summary))

print("\nMonthly output:")
print(monthly_file)

print("\nYearly output:")
print(yearly_file)

print("\n" + "=" * 60)