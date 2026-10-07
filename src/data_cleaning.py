import pandas as pd
from pathlib import Path


# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "data_processed"
OUTPUT_DIR.mkdir(exist_ok=True)


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
# 4. CREATE DATE COLUMN
# ==================================================

card_status["date"] = pd.to_datetime(
    dict(
        year=card_status["year"],
        month=card_status["month"],
        day=1
    )
)

transactions["date"] = pd.to_datetime(
    dict(
        year=transactions["year"],
        month=transactions["month"],
        day=1
    )
)


# ==================================================
# 5. REMOVE COMPLETELY EMPTY CARD STATUS COLUMNS
# ==================================================

empty_card_columns = [
    column
    for column in card_status.columns
    if card_status[column].isna().all()
]

print("\nCompletely empty Card Status columns:")
print(empty_card_columns)

card_status = card_status.drop(columns=empty_card_columns)


# ==================================================
# 6. CONVERT FPS DATETIME
# ==================================================

fps_location["dateTime"] = pd.to_datetime(
    fps_location["dateTime"],
    errors="coerce"
)


# ==================================================
# 7. STANDARDIZE TEXT COLUMNS
# ==================================================

for df in [card_status, transactions, fps_location]:

    text_columns = df.select_dtypes(include=["object", "string"]).columns

    for column in text_columns:
        df[column] = df[column].str.strip()


# ==================================================
# 8. SAVE CLEANED DATA
# ==================================================

card_status.to_csv(
    OUTPUT_DIR / "card_status_cleaned.csv",
    index=False
)

transactions.to_csv(
    OUTPUT_DIR / "transactions_cleaned.csv",
    index=False
)

fps_location.to_csv(
    OUTPUT_DIR / "fps_location_cleaned.csv",
    index=False
)


# ==================================================
# 9. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("CLEANING COMPLETED")
print("=" * 60)

print("\nCard Status:", card_status.shape)

print("Transactions:", transactions.shape)

print("FPS Location:", fps_location.shape)

print("\nSaved files:")

print(OUTPUT_DIR / "card_status_cleaned.csv")
print(OUTPUT_DIR / "transactions_cleaned.csv")
print(OUTPUT_DIR / "fps_location_cleaned.csv")