import pandas as pd
from pathlib import Path

# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data_processed"

# ==================================================
# 1. LOAD CLEANED DATA
# ==================================================

card_status = pd.read_csv(DATA_DIR / "card_status_cleaned.csv")
transactions = pd.read_csv(DATA_DIR / "transactions_cleaned.csv")
fps_location = pd.read_csv(DATA_DIR / "fps_location_cleaned.csv")

print("=" * 60)
print("LOADED CLEANED DATA")
print("=" * 60)

print("Card Status:", card_status.shape)
print("Transactions:", transactions.shape)
print("FPS Location:", fps_location.shape)

# ==================================================
# 2. CHECK DUPLICATES BEFORE MERGING
# ==================================================

merge_keys = ["shopNo", "month", "year"]

card_duplicates = card_status.duplicated(
    subset=merge_keys
).sum()

transaction_duplicates = transactions.duplicated(
    subset=merge_keys
).sum()

print("\nDuplicate Card Status records:", card_duplicates)
print("Duplicate Transaction records:", transaction_duplicates)

# ==================================================
# 3. CHECK CARD STATUS ↔ TRANSACTION MATCHING
# ==================================================

card_keys = card_status[merge_keys].drop_duplicates()

transaction_keys = transactions[merge_keys].drop_duplicates()

matching_keys = pd.merge(
    card_keys,
    transaction_keys,
    on=merge_keys,
    how="inner"
)

print("\nMonthly shop combinations:")
print("Card Status:", len(card_keys))
print("Transactions:", len(transaction_keys))
print("Matching:", len(matching_keys))

# ==================================================
# 4. MERGE CARD STATUS + TRANSACTIONS
# ==================================================

master = pd.merge(
    card_status,
    transactions,
    on=merge_keys,
    how="outer",
    suffixes=("_card", "_trans"),
    indicator=True
)

print("\n" + "=" * 60)
print("CARD STATUS + TRANSACTIONS MERGE")
print("=" * 60)

print("Master shape:", master.shape)

print("\nMerge status:")
print(master["_merge"].value_counts())

# ==================================================
# 5. CHECK FPS LOCATION MATCHING
# ==================================================

master_shops = master["shopNo"].drop_duplicates()

fps_shops = fps_location["shopNo"].drop_duplicates()

matching_fps_shops = master_shops[
    master_shops.isin(fps_shops)
]

unmatched_fps_shops = master_shops[
    ~master_shops.isin(fps_shops)
]

print("\n" + "=" * 60)
print("FPS LOCATION MATCHING")
print("=" * 60)

print("Unique shops in Master:", len(master_shops))
print("Unique shops in FPS Location:", len(fps_shops))
print("Matching shops:", len(matching_fps_shops))
print("Unmatched shops:", len(unmatched_fps_shops))

# ==================================================
# 6. CHECK FPS DUPLICATES
# ==================================================

fps_duplicates = fps_location.duplicated(
    subset=["shopNo"]
).sum()

print("\nDuplicate FPS shop records:", fps_duplicates)

# ==================================================
# 7. MERGE FPS LOCATION
# ==================================================

master = pd.merge(
    master,
    fps_location,
    on="shopNo",
    how="left",
    suffixes=("", "_fps"),
    validate="many_to_one"
)

# ==================================================
# 8. DROP MERGE INDICATOR
# ==================================================

master = master.drop(columns=["_merge"])

# ==================================================
# 9. SAVE MASTER DATASET
# ==================================================

output_file = DATA_DIR / "pds_master_dataset.csv"

master.to_csv(
    output_file,
    index=False
)

# ==================================================
# 10. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("MASTER DATASET CREATED")
print("=" * 60)

print("Final shape:", master.shape)

print("\nFinal columns:")
print(master.columns.tolist())

print("\nSaved to:")
print(output_file)

print("\n" + "=" * 60)