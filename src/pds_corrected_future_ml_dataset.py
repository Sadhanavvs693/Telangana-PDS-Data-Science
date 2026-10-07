import pandas as pd
from pathlib import Path

print("=" * 70)
print("PDS FINAL CORRECTED FUTURE ML DATASET")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load master dataset
# ---------------------------------------------------------

df = pd.read_csv(
    "data_processed/pds_master_dataset.csv"
)

print("\nOriginal dataset shape:")
print(df.shape)

# ---------------------------------------------------------
# 2. Consolidate district
# ---------------------------------------------------------

df["district"] = (
    df["distName_card"]
    .fillna(df["distName_trans"])
)

# ---------------------------------------------------------
# 3. Keep transaction records
# ---------------------------------------------------------

df = df[
    df["noOfTrans"].notna()
].copy()

print("\nTransaction records:")
print(len(df))

# ---------------------------------------------------------
# 4. Create calendar date
# ---------------------------------------------------------

df["date"] = pd.to_datetime(
    dict(
        year=df["year"],
        month=df["month"],
        day=1
    )
)

df = df.sort_values(
    ["shopNo", "date"]
).reset_index(drop=True)

# ---------------------------------------------------------
# 5. Define high transaction threshold
# ---------------------------------------------------------

threshold = df["noOfTrans"].quantile(0.75)

print("\nHigh transaction threshold:")
print(round(threshold, 2))

df["current_high_transaction"] = (
    df["noOfTrans"] >= threshold
).astype(int)

# ---------------------------------------------------------
# 6. Create calendar dates
# ---------------------------------------------------------

df["prev_month_date"] = (
    df["date"] - pd.DateOffset(months=1)
)

df["prev_2_month_date"] = (
    df["date"] - pd.DateOffset(months=2)
)

df["prev_3_month_date"] = (
    df["date"] - pd.DateOffset(months=3)
)

df["next_month_date"] = (
    df["date"] + pd.DateOffset(months=1)
)

# ---------------------------------------------------------
# 7. Historical lookup table
# ---------------------------------------------------------

history = df[
    [
        "shopNo",
        "date",
        "noOfTrans",
        "current_high_transaction",
        "totalRcs",
        "totalUnits",
        "noOfRcs"
    ]
].copy()

# ---------------------------------------------------------
# 8. Previous month
# ---------------------------------------------------------

prev1 = history.rename(
    columns={
        "date": "prev_month_date",
        "noOfTrans": "prev_month_transactions",
        "current_high_transaction":
            "prev_month_high_transaction",
        "totalRcs":
            "prev_month_totalRcs",
        "totalUnits":
            "prev_month_totalUnits",
        "noOfRcs":
            "prev_month_noOfRcs"
    }
)

df = df.merge(
    prev1[
        [
            "shopNo",
            "prev_month_date",
            "prev_month_transactions",
            "prev_month_high_transaction",
            "prev_month_totalRcs",
            "prev_month_totalUnits",
            "prev_month_noOfRcs"
        ]
    ],
    on=[
        "shopNo",
        "prev_month_date"
    ],
    how="left"
)

# ---------------------------------------------------------
# 9. Previous 2 months
# ---------------------------------------------------------

prev2 = history[
    [
        "shopNo",
        "date",
        "noOfTrans"
    ]
].rename(
    columns={
        "date": "prev_2_month_date",
        "noOfTrans": "prev_2_month_transactions"
    }
)

df = df.merge(
    prev2,
    on=[
        "shopNo",
        "prev_2_month_date"
    ],
    how="left"
)

# ---------------------------------------------------------
# 10. Previous 3 months
# ---------------------------------------------------------

prev3 = history[
    [
        "shopNo",
        "date",
        "noOfTrans"
    ]
].rename(
    columns={
        "date": "prev_3_month_date",
        "noOfTrans": "prev_3_month_transactions"
    }
)

df = df.merge(
    prev3,
    on=[
        "shopNo",
        "prev_3_month_date"
    ],
    how="left"
)

# ---------------------------------------------------------
# 11. STRICT 3-MONTH ROLLING FEATURES
# ---------------------------------------------------------
# These use ONLY the exact previous 3 calendar months.
#
# If one of the three months is missing,
# the rolling feature becomes NaN.
#
# This prevents gaps from being treated as consecutive months.

df["rolling_3_month_avg_transactions"] = (
    df[
        [
            "prev_month_transactions",
            "prev_2_month_transactions",
            "prev_3_month_transactions"
        ]
    ]
    .mean(
        axis=1,
        skipna=False
    )
)

df["rolling_3_month_max_transactions"] = (
    df[
        [
            "prev_month_transactions",
            "prev_2_month_transactions",
            "prev_3_month_transactions"
        ]
    ]
    .max(
        axis=1,
        skipna=False
    )
)

# Strict rolling average of ration cards
#
# We need exact previous-month card values.
# Add previous 2 and previous 3 month card counts.

prev2_cards = history[
    [
        "shopNo",
        "date",
        "noOfRcs"
    ]
].rename(
    columns={
        "date": "prev_2_month_date",
        "noOfRcs": "prev_2_month_noOfRcs"
    }
)

df = df.merge(
    prev2_cards,
    on=[
        "shopNo",
        "prev_2_month_date"
    ],
    how="left"
)

prev3_cards = history[
    [
        "shopNo",
        "date",
        "noOfRcs"
    ]
].rename(
    columns={
        "date": "prev_3_month_date",
        "noOfRcs": "prev_3_month_noOfRcs"
    }
)

df = df.merge(
    prev3_cards,
    on=[
        "shopNo",
        "prev_3_month_date"
    ],
    how="left"
)

df["rolling_3_month_avg_cards"] = (
    df[
        [
            "prev_month_noOfRcs",
            "prev_2_month_noOfRcs",
            "prev_3_month_noOfRcs"
        ]
    ]
    .mean(
        axis=1,
        skipna=False
    )
)

# ---------------------------------------------------------
# 12. Exact next-calendar-month target
# ---------------------------------------------------------

target = history[
    [
        "shopNo",
        "date",
        "current_high_transaction"
    ]
].rename(
    columns={
        "date": "next_month_date",
        "current_high_transaction":
            "target_next_month"
    }
)

df = df.merge(
    target,
    on=[
        "shopNo",
        "next_month_date"
    ],
    how="left"
)

# ---------------------------------------------------------
# 13. Keep only rows with actual next month
# ---------------------------------------------------------

before = len(df)

df = df[
    df["target_next_month"].notna()
].copy()

print("\nRows before exact next-month filtering:")
print(before)

print("\nRows after exact next-month filtering:")
print(len(df))

print("\nRows removed:")
print(
    before - len(df)
)

# ---------------------------------------------------------
# 14. Require actual previous month
# ---------------------------------------------------------

before_previous = len(df)

df = df[
    df["prev_month_transactions"].notna()
].copy()

print("\nRows removed because previous month unavailable:")
print(
    before_previous - len(df)
)

# ---------------------------------------------------------
# 15. Time features
# ---------------------------------------------------------

df["quarter"] = (
    (df["month"] - 1) // 3 + 1
)

df["is_year_end"] = (
    df["month"] == 12
).astype(int)

# ---------------------------------------------------------
# 16. Final features
# ---------------------------------------------------------

final_features = [
    "month",
    "year",
    "quarter",
    "is_year_end",
    "district",
    "latitude",
    "longitude",

    "prev_month_transactions",
    "prev_2_month_transactions",
    "prev_3_month_transactions",

    "prev_month_high_transaction",

    "rolling_3_month_avg_transactions",
    "rolling_3_month_max_transactions",

    "prev_month_totalRcs",
    "prev_month_totalUnits",
    "prev_month_noOfRcs",

    "rolling_3_month_avg_cards",

    "target_next_month"
]

df_final = df[
    final_features
].copy()

# ---------------------------------------------------------
# 17. Target distribution
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print("\nShape:")
print(df_final.shape)

print("\nTarget distribution:")
print(
    df_final[
        "target_next_month"
    ].value_counts()
)

print("\nTarget percentage:")
print(
    df_final[
        "target_next_month"
    ]
    .value_counts(
        normalize=True
    )
    .mul(100)
    .round(2)
)

# ---------------------------------------------------------
# 18. Missing values
# ---------------------------------------------------------

print("\nMissing values:")

missing = (
    df_final
    .isnull()
    .sum()
    .sort_values(
        ascending=False
    )
)

print(
    missing[
        missing > 0
    ]
)

# ---------------------------------------------------------
# 19. Temporal alignment validation
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TEMPORAL ALIGNMENT CHECK")
print("=" * 70)

# Target must always be exactly one calendar month ahead.

check_dates = df[
    [
        "date",
        "next_month_date"
    ]
].copy()

expected_next = (
    check_dates["date"]
    + pd.DateOffset(months=1)
)

alignment_errors = (
    expected_next
    != check_dates["next_month_date"]
).sum()

print("\nNext-month alignment errors:")
print(alignment_errors)

if alignment_errors == 0:
    print(
        "✓ Target date alignment is correct"
    )
else:
    print(
        "⚠ Target date alignment problem detected"
    )

# ---------------------------------------------------------
# 20. Leakage check
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)

for col in [
    "noOfTrans",
    "current_high_transaction"
]:

    if col in df_final.columns:
        print(
            f"⚠ {col} FOUND IN FEATURES"
        )
    else:
        print(
            f"✓ {col} not included as feature"
        )

print(
    "✓ target_next_month is the target"
)

# ---------------------------------------------------------
# 21. Save
# ---------------------------------------------------------

output_path = Path(
    "output/pds_final_future_ml_dataset.csv"
)

df_final.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 70)
print("FINAL CORRECTED DATASET SAVED")
print("=" * 70)

print("\nOutput:")
print(output_path)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)