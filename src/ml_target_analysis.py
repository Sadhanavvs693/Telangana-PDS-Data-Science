import pandas as pd
from pathlib import Path


# ============================================================
# PDS ML TARGET ANALYSIS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR /
    "data_processed" /
    "pds_master_dataset.csv"
)


print("=" * 65)
print("PDS MACHINE LEARNING TARGET ANALYSIS")
print("=" * 65)


# ============================================================
# 1. LOAD MASTER DATASET
# ============================================================

df = pd.read_csv(DATA_FILE)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# 2. DISPLAY ALL COLUMNS
# ============================================================

print("\n" + "=" * 65)
print("ALL DATASET COLUMNS")
print("=" * 65)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:02d}. {column}")


# ============================================================
# 3. DATA TYPES
# ============================================================

print("\n" + "=" * 65)
print("DATA TYPES")
print("=" * 65)

print(
    df.dtypes.to_string()
)


# ============================================================
# 4. MISSING VALUES
# ============================================================

print("\n" + "=" * 65)
print("MISSING VALUES")
print("=" * 65)

missing = (
    df.isnull()
    .sum()
    .sort_values(
        ascending=False
    )
)

missing_percentage = (
    missing
    / len(df)
    * 100
)

missing_summary = pd.DataFrame({
    "missing_count": missing,
    "missing_percentage": missing_percentage
})

print(
    missing_summary[
        missing_summary["missing_count"] > 0
    ]
    .to_string()
)


# ============================================================
# 5. UNIQUE VALUES
# ============================================================

print("\n" + "=" * 65)
print("UNIQUE VALUE COUNTS")
print("=" * 65)

unique_counts = (
    df.nunique(dropna=True)
    .sort_values()
)

print(
    unique_counts.to_string()
)


# ============================================================
# 6. NUMERIC COLUMNS
# ============================================================

numeric_columns = (
    df.select_dtypes(
        include=["number"]
    )
    .columns
    .tolist()
)

print("\n" + "=" * 65)
print("NUMERIC COLUMNS")
print("=" * 65)

for column in numeric_columns:
    print(column)


# ============================================================
# 7. CATEGORICAL COLUMNS
# ============================================================

categorical_columns = (
    df.select_dtypes(
        include=["object", "category"]
    )
    .columns
    .tolist()
)

print("\n" + "=" * 65)
print("CATEGORICAL COLUMNS")
print("=" * 65)

for column in categorical_columns:
    print(column)


# ============================================================
# 8. POTENTIAL TARGET COLUMNS
# ============================================================

print("\n" + "=" * 65)
print("POTENTIAL TARGET COLUMNS")
print("=" * 65)


# Display columns with relatively small numbers
# of unique values — useful for classification targets.

potential_targets = []

for column in df.columns:

    unique_count = df[column].nunique(
        dropna=True
    )

    if unique_count <= 20:

        potential_targets.append({
            "column": column,
            "unique_values": unique_count,
            "data_type": str(
                df[column].dtype
            )
        })


potential_targets_df = pd.DataFrame(
    potential_targets
)

print(
    potential_targets_df
    .to_string(index=False)
)


# ============================================================
# 9. NUMERIC SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("NUMERIC SUMMARY")
print("=" * 65)

print(
    df[numeric_columns]
    .describe()
    .T
    .to_string()
)


# ============================================================
# 10. SAVE TARGET ANALYSIS
# ============================================================

OUTPUT_DIR = BASE_DIR / "output"

target_file = (
    OUTPUT_DIR /
    "ml_target_analysis.csv"
)

potential_targets_df.to_csv(
    target_file,
    index=False
)


print("\n" + "=" * 65)
print("TARGET ANALYSIS COMPLETED")
print("=" * 65)

print(
    "\nSaved:",
    target_file
)