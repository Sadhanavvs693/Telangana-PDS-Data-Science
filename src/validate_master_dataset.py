import pandas as pd
from pathlib import Path

# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data_processed"

# ==================================================
# 1. LOAD MASTER DATASET
# ==================================================

master = pd.read_csv(
    DATA_DIR / "pds_master_dataset.csv"
)

print("=" * 60)
print("MASTER DATASET VALIDATION")
print("=" * 60)

print("\nShape:")
print(master.shape)

# ==================================================
# 2. CHECK MISSING VALUES
# ==================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = master.isna().sum()

missing = missing[missing > 0].sort_values(
    ascending=False
)

print(missing)

# ==================================================
# 3. CHECK DATE RANGE
# ==================================================

master["date_card"] = pd.to_datetime(
    master["date_card"],
    errors="coerce"
)

master["date_trans"] = pd.to_datetime(
    master["date_trans"],
    errors="coerce"
)

print("\n" + "=" * 60)
print("DATE RANGE")
print("=" * 60)

print(
    "Card Status:",
    master["date_card"].min(),
    "to",
    master["date_card"].max()
)

print(
    "Transactions:",
    master["date_trans"].min(),
    "to",
    master["date_trans"].max()
)

# ==================================================
# 4. UNIQUE DISTRICTS
# ==================================================

print("\n" + "=" * 60)
print("DISTRICTS")
print("=" * 60)

print(
    "Unique districts:",
    master["distCode_card"].nunique()
)

print(
    "District names:",
    master["distName_card"].nunique()
)

# ==================================================
# 5. UNIQUE SHOPS
# ==================================================

print("\n" + "=" * 60)
print("SHOPS")
print("=" * 60)

print(
    "Unique shops:",
    master["shopNo"].nunique()
)

# ==================================================
# 6. FPS STATUS
# ==================================================

print("\n" + "=" * 60)
print("FPS STATUS")
print("=" * 60)

print(
    master["fpsStatus"].value_counts(
        dropna=False
    )
)

# ==================================================
# 7. TRANSACTION SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("TRANSACTION SUMMARY")
print("=" * 60)

transaction_columns = [
    "noOfRcs",
    "noOfTrans",
    "riceAfsc",
    "riceFsc",
    "riceAap",
    "wheat",
    "sugar",
    "rgdal",
    "kerosene",
    "totalAmount",
    "salt",
    "otherShopTransCnt"
]

print(
    master[transaction_columns].sum(
        numeric_only=True
    )
)

# ==================================================
# 8. CARD STATUS SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("CARD STATUS SUMMARY")
print("=" * 60)

card_columns = [
    "totalRcs",
    "totalUnits",
    "totalRcNfsa",
    "totalUnitsNfsa",
    "totalRcState",
    "totalUnitsState"
]

print(
    master[card_columns].sum(
        numeric_only=True
    )
)

# ==================================================
# 9. DUPLICATE CHECK
# ==================================================

print("\n" + "=" * 60)
print("DUPLICATE CHECK")
print("=" * 60)

duplicates = master.duplicated(
    subset=["shopNo", "month", "year"]
).sum()

print(
    "Duplicate shop-month-year records:",
    duplicates
)

# ==================================================
# 10. GPS COVERAGE
# ==================================================

print("\n" + "=" * 60)
print("GPS COVERAGE")
print("=" * 60)

print(
    "Missing latitude:",
    master["latitude"].isna().sum()
)

print(
    "Missing longitude:",
    master["longitude"].isna().sum()
)

# ==================================================
# 11. FINAL MESSAGE
# ==================================================

print("\n" + "=" * 60)
print("VALIDATION COMPLETED")
print("=" * 60)