# Telangana PDS Data Analytics & Anomaly Detection

## 📌 Project Overview

This project analyzes Telangana Public Distribution System (PDS) data to understand ration distribution patterns, identify statistically unusual Fair Price Shops (FPS), and predict future high transaction activity using Machine Learning.

The project follows an end-to-end Data Science workflow:

**Data Cleaning → Data Integration → EDA → Anomaly Detection → Leakage Investigation → Feature Engineering → Machine Learning → Model Comparison → Streamlit Dashboard**

> **Important:** Anomalies identified in this project are statistical signals for further investigation and should not be interpreted as confirmed fraud.

---

## 🎯 Business Problem

The Telangana PDS system contains large volumes of data related to ration cards, Fair Price Shops, transactions, commodities, and shop locations.

The objectives of this project are to:

* Analyze transaction activity across districts and Fair Price Shops.
* Identify statistically unusual transaction behavior.
* Detect shops repeatedly flagged by multiple anomaly detection methods.
* Understand historical patterns affecting future transaction activity.
* Predict whether a shop is likely to have high transaction activity in the following month.
* Provide an interactive dashboard for data exploration and monitoring.

---

## 📊 Dataset

The final integrated master dataset contains:

* **517,666 records**
* **53 columns**
* **17,576 unique shops**
* **33 districts**

The project integrates three major data sources:

1. Card Status Data
2. Transaction Data
3. FPS Location Data

### Major Data Categories

| Category         | Important Columns                                              |
| ---------------- | -------------------------------------------------------------- |
| Shop Information | `shopNo`, `fpsStatus`, `fpsType`                               |
| District         | `distName`, `distCode`                                         |
| Time             | `year`, `month`, `dateTime`                                    |
| Ration Cards     | `totalRcs`, `noOfRcs`                                          |
| Transactions     | `noOfTrans`, `otherShopTransCnt`                               |
| Commodities      | `riceAfsc`, `riceFsc`, `riceAap`, `wheat`, `sugar`, `kerosene` |
| Location         | `latitude`, `longitude`, `address`                             |
| Financial        | `totalAmount`                                                  |

The large source datasets are not included in the GitHub repository.

---

# 🔄 Project Workflow

```text
Raw PDS Data
      ↓
Data Cleaning & Integration
      ↓
Master Dataset
      ↓
Exploratory Data Analysis
      ↓
District & Transaction Analysis
      ↓
Statistical Anomaly Detection
      ↓
Isolation Forest
      ↓
Multi-Method Anomaly Comparison
      ↓
ML Target & Leakage Investigation
      ↓
Temporal Feature Engineering
      ↓
Future-Month Prediction Dataset
      ↓
Logistic Regression
      ↓
Random Forest
      ↓
XGBoost
      ↓
Model Comparison
      ↓
Streamlit Dashboard
```

---

# 1. Data Preparation

The project combines card, transaction, and FPS location datasets into a master analytical dataset.

The final master dataset contains:

**517,666 rows × 53 columns**

Data preparation included:

* Data type validation
* Missing-value analysis
* Dataset merging
* Duplicate checks
* District and shop validation
* Date and time processing
* Feature creation
* Transaction-level aggregation

There were **17,576 unique shops**, of which **17,491 had transaction data**.

---

# 2. Exploratory Data Analysis

The analysis focuses on:

* District-level transaction activity
* Shop-level transaction activity
* Monthly transaction trends
* Yearly transaction trends
* Ration-card distribution
* Commodity distribution
* Transaction-to-ration-card relationships
* Geographical shop information

### Key observations

Hyderabad showed the highest average transaction activity among districts.

High-activity districts included:

* Hyderabad
* Medchal
* Ranga Reddy

These districts were also important during anomaly analysis.

---

# 3. Statistical Anomaly Detection

Three statistical approaches were used to identify unusual shop behavior.

### 3.1 Transaction Outliers

The Interquartile Range (IQR) method was applied to shop-level transaction activity.

Results:

* Q1 = **276.77**
* Q3 = **536.54**
* IQR = **259.77**
* Upper outlier threshold = **926.19**
* Transaction outliers = **910**

---

### 3.2 Transaction-to-Ration-Card Ratio

The transaction-to-ration-card ratio was analyzed to identify shops with unusual transaction behavior relative to their registered ration-card base.

Results:

* Ratio outliers = **958**

---

### 3.3 Rice per Transaction

Rice distribution relative to transaction activity was also analyzed.

Results:

* Rice/transaction outliers = **267**

---

# 4. Isolation Forest

Isolation Forest was used as an unsupervised Machine Learning method for anomaly detection.

### Features Used

* Transactions per month
* Transactions per ration card
* Rice per transaction
* Total ration cards
* Total transactions

### Configuration

```python
IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)
```

### Results

| Category        |  Count |
| --------------- | -----: |
| Shops analyzed  | 17,491 |
| Normal shops    | 16,616 |
| Anomalous shops |    875 |
| Anomaly rate    |     5% |

---

# 5. Multi-Method Anomaly Detection

The project compared four anomaly detection methods:

1. Transaction IQR outliers
2. Transaction/Ration-card ratio outliers
3. Rice/transaction outliers
4. Isolation Forest anomalies

### Results

| Measure                                |   Count |
| -------------------------------------- | ------: |
| Transaction outliers                   |     910 |
| Ratio outliers                         |     958 |
| Rice outliers                          |     267 |
| Isolation Forest anomalies             |     875 |
| Transaction + Isolation Forest overlap |     627 |
| Ratio + Isolation Forest overlap       |     383 |
| Rice + Isolation Forest overlap        |      74 |
| Common IQR + Isolation Forest          |     781 |
| High-priority shops                    | **296** |
| Shops flagged by all 4 methods         |   **7** |

The **296 high-priority shops** were identified based on agreement across multiple anomaly detection methods.

Seven shops were flagged by all four methods, and these were located in Hyderabad.

### Important Interpretation

These results identify **shops requiring further investigation**.

They do **not** prove fraud or misconduct.

---

# 6. Initial Machine Learning Approach

An initial classification model was created to predict high transaction activity.

The first Logistic Regression model achieved:

* Accuracy: **99.73%**
* ROC-AUC: **99.99%**

Although these metrics appeared excellent, the performance was considered suspiciously high.

This led to a detailed leakage investigation.

---

# 7. Data Leakage Investigation

The analysis found extremely strong relationships between current-month variables.

For example:

```text
Correlation(noOfRcs, noOfTrans) = 0.999804
```

Other highly correlated variables included:

* `totalRcs`
* `totalUnits`
* `totalUnitsNfsa`
* `unitsNfsaPhh`

The initial target was also directly derived from the current month's transaction count.

Therefore, the initial ML setup was not considered a realistic predictive problem.

### Corrective Action

The ML problem was redesigned as a **future-month prediction task**.

Instead of predicting:

> Whether a shop has high transaction activity this month

the final model predicts:

> Whether a shop will have high transaction activity next month using historical information.

This significantly reduced the risk of temporal and target leakage.

---

# 8. Final Temporal Machine Learning Dataset

The final ML dataset uses exact calendar-month alignment.

### Dataset

* **412,665 rows**
* Target: `target_next_month`
* Normal: **308,714 (74.81%)**
* High transaction: **103,951 (25.19%)**

### Historical Features

The model uses historical information such as:

* Previous-month transactions
* Previous-month ration cards
* Previous-month high-transaction status
* Previous two-month transactions
* Previous three-month transactions
* Three-month average transactions
* Three-month maximum transactions
* Previous-month total ration cards
* Previous-month total units
* District
* Month
* Year
* Latitude
* Longitude

The final temporal alignment validation produced:

**0 calendar-alignment errors**

---

# 9. Train/Test Strategy

A chronological train/test split was used to simulate real-world prediction.

### Training Data

**2023–2024**

* 343,725 rows
* Normal: 257,933
* High transaction: 85,792

### Testing Data

**2025**

* 68,940 rows
* Normal: 50,781
* High transaction: 18,159

Using a time-based split prevents future observations
