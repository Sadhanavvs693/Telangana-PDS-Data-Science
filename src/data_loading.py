import pandas as pd
from pathlib import Path


# ==================================================
# PROJECT PATH
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ==================================================
# 1. LOAD CARD STATUS DATA
# ==================================================

card_files = list((BASE_DIR / "card_status data").rglob("*.csv"))

card_dfs = []

for file in card_files:
    df = pd.read_csv(file)
    card_dfs.append(df)

card_status = pd.concat(card_dfs, ignore_index=True)


# ==================================================
# 2. LOAD TRANSACTION DATA
# ==================================================

transaction_files = list((BASE_DIR / "Transaction data").rglob("*.csv"))

transaction_dfs = []

for file in transaction_files:
    df = pd.read_csv(file)
    transaction_dfs.append(df)

transactions = pd.concat(transaction_dfs, ignore_index=True)


# ==================================================
# 3. LOAD FPS LOCATION DATA
# ==================================================

fps_files = list((BASE_DIR / "FPS location data").rglob("*.csv"))

fps_dfs = []

for file in fps_files:
    df = pd.read_csv(file)
    fps_dfs.append(df)

fps_location = pd.concat(fps_dfs, ignore_index=True)


# ==================================================
# 4. BASIC DATASET INFORMATION
# ==================================================

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print("\nCARD STATUS")
print("Files:", len(card_files))
print("Shape:", card_status.shape)

print("\nTRANSACTIONS")
print("Files:", len(transaction_files))
print("Shape:", transactions.shape)

print("\nFPS LOCATION")
print("Files:", len(fps_files))
print("Shape:", fps_location.shape)


# ==================================================
# 5. MISSING VALUES
# ==================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print("\nCARD STATUS")
print(card_status.isnull().sum().sort_values(ascending=False))

print("\nTRANSACTIONS")
print(transactions.isnull().sum().sort_values(ascending=False))

print("\nFPS LOCATION")
print(fps_location.isnull().sum().sort_values(ascending=False))


# ==================================================
# 6. DUPLICATE ROWS
# ==================================================

print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)

print("Card Status duplicates:", card_status.duplicated().sum())
print("Transaction duplicates:", transactions.duplicated().sum())
print("FPS Location duplicates:", fps_location.duplicated().sum())


# ==================================================
# 7. UNIQUE DISTRICTS AND SHOPS
# ==================================================

print("\n" + "=" * 60)
print("UNIQUE VALUES")
print("=" * 60)

print("Card Status - Districts:", card_status["distName"].nunique())
print("Card Status - Shops:", card_status["shopNo"].nunique())

print("Transactions - Districts:", transactions["distName"].nunique())
print("Transactions - Shops:", transactions["shopNo"].nunique())

print("FPS Location - Districts:", fps_location["distName"].nunique())
print("FPS Location - Shops:", fps_location["shopNo"].nunique())


# ==================================================
# 8. TIME COVERAGE
# ==================================================

print("\n" + "=" * 60)
print("TIME COVERAGE")
print("=" * 60)

print("\nCARD STATUS")
print(card_status.groupby("year")["month"].agg(["min", "max"]))

print("\nTRANSACTIONS")
print(transactions.groupby("year")["month"].agg(["min", "max"]))


# ==================================================
# 9. DATA TYPES
# ==================================================

print("\n" + "=" * 60)
print("CARD STATUS DATA TYPES")
print("=" * 60)

print(card_status.dtypes)

print("\n" + "=" * 60)
print("TRANSACTION DATA TYPES")
print("=" * 60)

print(transactions.dtypes)

print("\n" + "=" * 60)
print("FPS LOCATION DATA TYPES")
print("=" * 60)

print(fps_location.dtypes)
# ==================================================
# 10. CHECK DATA GRAIN
# ==================================================

print("\n" + "=" * 60)
print("DATA GRAIN CHECK")
print("=" * 60)

# Card Status
card_key_duplicates = card_status.duplicated(
    subset=["shopNo", "month", "year"]
).sum()

print("\nCard Status duplicate Shop-Month-Year combinations:")
print(card_key_duplicates)


# Transactions
transaction_key_duplicates = transactions.duplicated(
    subset=["shopNo", "month", "year"]
).sum()

print("\nTransaction duplicate Shop-Month-Year combinations:")
print(transaction_key_duplicates)


# FPS Location
fps_duplicate_shops = fps_location["shopNo"].duplicated().sum()

print("\nFPS Location duplicate Shop numbers:")
print(fps_duplicate_shops)


# Number of unique Shop-Month-Year combinations

print("\nCard Status unique Shop-Month-Year:")
print(
    card_status[["shopNo", "month", "year"]]
    .drop_duplicates()
    .shape[0]
)

print("\nTransaction unique Shop-Month-Year:")
print(
    transactions[["shopNo", "month", "year"]]
    .drop_duplicates()
    .shape[0]
)

print("\nFPS Location unique shops:")
print(
    fps_location["shopNo"].nunique()
)

print("\nCARD STATUS FILES")
for file in sorted(card_files):
    print(file)

print("\nTRANSACTION FILES")
for file in sorted(transaction_files):
    print(file)