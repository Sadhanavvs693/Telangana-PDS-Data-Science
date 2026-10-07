# Telangana PDS Data Analytics, Anomaly Detection & Machine Learning

## 📌 Project Overview

This project analyzes **Telangana Public Distribution System (PDS)** data to understand ration distribution patterns, identify unusual Fair Price Shop (FPS) activity, and build a machine learning model to predict high transaction activity for the following month.

The project combines:

* Data cleaning and integration
* Exploratory data analysis
* District and shop-level analysis
* Statistical anomaly detection
* Isolation Forest
* Multi-method anomaly investigation
* Leakage investigation
* Temporal feature engineering
* Machine learning classification
* Streamlit dashboard development

---

## 🎯 Business Problem

The Telangana PDS system contains transaction, ration-card, commodity, and Fair Price Shop information across multiple districts and months.

The key business questions addressed are:

1. Which districts and shops have the highest transaction activity?
2. Which shops show unusual transaction patterns?
3. Can multiple anomaly-detection methods identify high-priority shops for investigation?
4. Can historical transaction behavior predict high transaction activity in the following month?
5. Which factors are most useful for predicting future transaction activity?
6. How can the results be presented through an interactive dashboard?

> **Important:** Anomalies identified in this project are investigation signals and should not automatically be interpreted as fraud.

---

# 📊 Dataset

The project integrates three major sources of PDS information:

### 1. Card Status Data

Contains information related to ration cards, units, card categories, districts, shops, and monthly records.

### 2. Transaction Data

Contains monthly transaction and commodity distribution information for Fair Price Shops.

### 3. FPS Location Data

Contains Fair Price Shop information including:

* District
* Shop number
* Address
* Latitude
* Longitude
* FPS status
* FPS type

These datasets were cleaned and integrated into a master dataset.

### Master Dataset

**Shape:** `517,666
