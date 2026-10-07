# ============================================================
# FINAL FUTURE ML - XGBOOST
# Telangana PDS Analytics Project
# ============================================================

import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)

TRAIN_FILE = os.path.join(
    OUTPUT_DIR,
    "pds_final_future_ml_train.csv"
)

TEST_FILE = os.path.join(
    OUTPUT_DIR,
    "pds_final_future_ml_test.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("FINAL FUTURE ML - XGBOOST")
print("=" * 70)

print("\nLoading training data...")
train_df = pd.read_csv(TRAIN_FILE)

print("Loading testing data...")
test_df = pd.read_csv(TEST_FILE)

print("\nTrain shape:", train_df.shape)
print("Test shape :", test_df.shape)


# ============================================================
# 3. TARGET
# ============================================================

TARGET = "target_next_month"

if TARGET not in train_df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found in training data."
    )

if TARGET not in test_df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found in testing data."
    )


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X_train = train_df.drop(
    columns=[TARGET]
)

y_train = train_df[TARGET]

X_test = test_df.drop(
    columns=[TARGET]
)

y_test = test_df[TARGET]


print("\nTraining target distribution:")
print(
    y_train.value_counts()
    .sort_index()
)

print("\nTesting target distribution:")
print(
    y_test.value_counts()
    .sort_index()
)


# ============================================================
# 5. LEAKAGE CHECK
# ============================================================

leakage_columns = [
    "noOfTrans",
    "current_high_transaction"
]

found_leakage = [
    col
    for col in leakage_columns
    if col in X_train.columns
]

if found_leakage:

    print("\nWARNING: Potential leakage columns found:")
    print(found_leakage)

    X_train = X_train.drop(
        columns=found_leakage
    )

    X_test = X_test.drop(
        columns=found_leakage
    )

    print("Leakage columns removed.")

else:

    print("\nLeakage check passed.")
    print(
        "No current-month transaction target/proxy columns found."
    )


# ============================================================
# 6. IDENTIFY FEATURES
# ============================================================

numeric_features = X_train.select_dtypes(
    include=[
        "int64",
        "int32",
        "float64",
        "float32"
    ]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=[
        "object",
        "category",
        "bool"
    ]
).columns.tolist()


print("\nNumber of numeric features:",
      len(numeric_features))

print("Number of categorical features:",
      len(categorical_features))

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 7. NUMERIC PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


# ============================================================
# 8. CATEGORICAL PREPROCESSING
# ============================================================

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ============================================================
# 9. COLUMN TRANSFORMER
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ],
    remainder="drop"
)


# ============================================================
# 10. XGBOOST MODEL
# ============================================================

xgb_model = XGBClassifier(

    n_estimators=300,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=42,

    n_jobs=-1
)


# ============================================================
# 11. COMPLETE PIPELINE
# ============================================================

model_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            xgb_model
        )
    ]
)


# ============================================================
# 12. TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING XGBOOST")
print("=" * 70)

print("\nTraining model...")

model_pipeline.fit(
    X_train,
    y_train
)

print("Training completed successfully.")


# ============================================================
# 13. PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = model_pipeline.predict(
    X_test
)

y_prob = model_pipeline.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 14. METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# 15. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST RESULTS")
print("=" * 70)

print(
    f"\nAccuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)

print("\nConfusion Matrix:")

print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 16. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({

    "model": [
        "XGBoost"
    ],

    "accuracy": [
        accuracy
    ],

    "precision": [
        precision
    ],

    "recall": [
        recall
    ],

    "f1_score": [
        f1
    ],

    "roc_auc": [
        roc_auc
    ]
})


metrics_file = os.path.join(
    OUTPUT_DIR,
    "final_future_xgboost_metrics.csv"
)


metrics_df.to_csv(
    metrics_file,
    index=False
)


print("\nMetrics saved to:")

print(metrics_file)


# ============================================================
# 17. SAVE PREDICTIONS
# ============================================================

predictions_df = test_df.copy()

predictions_df[
    "predicted_next_month"
] = y_pred

predictions_df[
    "prediction_probability"
] = y_prob


predictions_file = os.path.join(
    OUTPUT_DIR,
    "final_future_xgboost_predictions.csv"
)


predictions_df.to_csv(
    predictions_file,
    index=False
)


print("\nPredictions saved to:")

print(predictions_file)


# ============================================================
# 18. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)


preprocessor_fitted = (
    model_pipeline
    .named_steps["preprocessor"]
)


feature_names = (
    preprocessor_fitted
    .get_feature_names_out()
)


importances = (
    model_pipeline
    .named_steps["model"]
    .feature_importances_
)


feature_importance_df = pd.DataFrame({

    "feature":
        feature_names,

    "importance":
        importances
})


feature_importance_df = (
    feature_importance_df
    .sort_values(
        by="importance",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# 19. SAVE FEATURE IMPORTANCE
# ============================================================

importance_file = os.path.join(
    OUTPUT_DIR,
    "final_future_xgboost_feature_importance.csv"
)


feature_importance_df.to_csv(
    importance_file,
    index=False
)


print("\nFeature importance saved to:")

print(importance_file)


# ============================================================
# 20. DISPLAY TOP 15 FEATURES
# ============================================================

print("\nTop 15 Features:")

print(
    feature_importance_df
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST TRAINING COMPLETED")
print("=" * 70)

print("\nFinal Metrics:")

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1 Score : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC  : {roc_auc * 100:.2f}%"
)


print("\nOutput files created:")

print(
    "1.",
    metrics_file
)

print(
    "2.",
    predictions_file
)

print(
    "3.",
    importance_file
)

print(
    "\nXGBoost pipeline completed successfully."
)