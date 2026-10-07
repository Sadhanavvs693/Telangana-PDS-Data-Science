import pandas as pd
from pathlib import Path

print("=" * 70)
print("PDS FINAL FUTURE ML PREPROCESSING")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load final corrected dataset
# ---------------------------------------------------------

input_path = Path(
    "output/pds_final_future_ml_dataset.csv"
)

df = pd.read_csv(input_path)

print("\nDataset shape:")
print(df.shape)

# ---------------------------------------------------------
# 2. Target
# ---------------------------------------------------------

target = "target_next_month"

# ---------------------------------------------------------
# 3. Chronological split
# ---------------------------------------------------------
# 2023-2024 -> Training
# 2025       -> Testing
#
# This prevents future information from entering training.

train_df = df[
    df["year"] < 2025
].copy()

test_df = df[
    df["year"] == 2025
].copy()

# ---------------------------------------------------------
# 4. Separate X and y
# ---------------------------------------------------------

X_train = train_df.drop(
    columns=[target]
)

y_train = train_df[target]

X_test = test_df.drop(
    columns=[target]
)

y_test = test_df[target]

# ---------------------------------------------------------
# 5. Feature types
# ---------------------------------------------------------

numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

print("\n" + "=" * 70)
print("FEATURE INFORMATION")
print("=" * 70)

print("\nNumeric features:")
for feature in numeric_features:
    print(" -", feature)

print("\nCategorical features:")
for feature in categorical_features:
    print(" -", feature)

# ---------------------------------------------------------
# 6. Dataset sizes
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print("\nTraining shape:")
print(train_df.shape)

print("\nTesting shape:")
print(test_df.shape)

# ---------------------------------------------------------
# 7. Target distribution
# ---------------------------------------------------------

print("\nTraining target distribution:")
print(
    y_train.value_counts()
)

print("\nTraining target percentage:")
print(
    y_train
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nTesting target distribution:")
print(
    y_test.value_counts()
)

print("\nTesting target percentage:")
print(
    y_test
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# ---------------------------------------------------------
# 8. Missing values
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print("\nTraining missing values:")

train_missing = (
    X_train
    .isnull()
    .sum()
    .sort_values(
        ascending=False
    )
)

print(
    train_missing[
        train_missing > 0
    ]
)

print("\nTesting missing values:")

test_missing = (
    X_test
    .isnull()
    .sum()
    .sort_values(
        ascending=False
    )
)

print(
    test_missing[
        test_missing > 0
    ]
)

# ---------------------------------------------------------
# 9. Duplicate check
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

print(
    "\nTraining duplicate rows:",
    X_train.duplicated().sum()
)

print(
    "Testing duplicate rows:",
    X_test.duplicated().sum()
)

# ---------------------------------------------------------
# 10. Leakage check
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)

leakage_columns = [
    "noOfTrans",
    "current_high_transaction"
]

for column in leakage_columns:

    if column in X_train.columns:

        print(
            f"⚠ WARNING: {column} FOUND"
        )

    else:

        print(
            f"✓ {column} not present"
        )

if target not in X_train.columns:
    print(
        "✓ target_next_month not present in features"
    )

# ---------------------------------------------------------
# 11. Verify chronological split
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CHRONOLOGICAL VALIDATION")
print("=" * 70)

print(
    "\nTraining years:",
    sorted(train_df["year"].unique())
)

print(
    "Testing years:",
    sorted(test_df["year"].unique())
)

if train_df["year"].max() < test_df["year"].min():

    print(
        "✓ Training data occurs before testing data"
    )

else:

    print(
        "⚠ Temporal overlap detected"
    )

# ---------------------------------------------------------
# 12. Save datasets
# ---------------------------------------------------------

train_output = Path(
    "output/pds_final_future_ml_train.csv"
)

test_output = Path(
    "output/pds_final_future_ml_test.csv"
)

train_df.to_csv(
    train_output,
    index=False
)

test_df.to_csv(
    test_output,
    index=False
)

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print("\nTraining:")
print(train_output)

print("\nTesting:")
print(test_output)

print("\n" + "=" * 70)
print("FINAL PREPROCESSING COMPLETE")
print("=" * 70)