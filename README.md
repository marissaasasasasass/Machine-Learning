# ML: HR Promotion Prediction & Airbnb Price Estimation

This repository contains a Machine Learning project (Python) that tackles two prediction problems:

1) **HR Analytics (Classification):** Predict whether an employee is likely to be promoted using a logistic regression approach, supported by data cleaning, feature engineering, and evaluation.  
2) **Airbnb (Regression):** Predict a listing’s rental price using a linear regression approach, with strong emphasis on handling overfitting through preprocessing techniques and feature selection.

The work emphasizes learning outcomes around underfitting/overfitting, transformations, encoding, scaling, regularization, and performance analysis.  

---

## Project Overview

### HR Analytics (Classification)
**Problem:** HR teams face challenges in manually processing employee data, which can be time-consuming and prone to missing patterns.  
**Goal:** Build a predictive model to identify employees most likely to be promoted.  

Key ideas used:
- Dropped high-cardinality identifiers (e.g., employee_id) to reduce encoding/overfitting issues.
- Imputed missing values using median/mode for relevant fields.
- Addressed class imbalance using **undersampling** to avoid bias towards the majority class.
- Applied **ordinal encoding** to keep category identity and reduce high-dimensional expansion.
- Used **standardization** to make feature scales comparable.
- Feature engineering using interaction features such as:
  - `dept_gender_interaction` (department × gender)
  - `perf_train` (KPIs_met>80% × avg_training_score)
- Feature selection decisions supported by statistical significance (p-values), e.g. dropping recruitment_channel when weakly significant.

---

### Airbnb (Regression)
**Problem:** Listing prices are affected by many factors (location, reviews, host activity), making accurate prediction difficult.  
**Goal:** Predict rental price based on listing and location features.

Key ideas used:
- Converted `last_review` to datetime, then extracted useful fields such as:
  - number of days since last review
  - year of review
- Dropped high-cardinality identifiers (e.g., id, host_id).
- Outlier handling with **winsorisation** to reduce extreme-value impact while retaining dataset size.
- Tested transformations for numeric columns (selected based on distribution/QQ-plots).
- Tackled overfitting with:
  - **Equal-width discretization (binning)** to simplify continuous variables
  - **Regularization** (ridge, lasso, elastic net; selected elastic net in the write-up)
  - **Standardization**
  - **PCA** for dimensionality reduction
- Feature engineering examples:
  - `latitude_longitude` (lat × long interaction)
  - `host_active` (calculated_host_listings_count × number_of_reviews interaction)

**Note on results:** The linear regression model struggled to explain variance (very low R²), indicating weak predictive signal and/or missing structure in features.

---

## Methods Used

- **Data Exploration**
  - rare category checks, distributions, cardinality checks
- **Data Cleaning**
  - missing value imputation (median/mode/mean depending on feature)
  - dropping high-cardinality identifiers
- **Preprocessing**
  - ordinal encoding
  - standardization/scaling
  - outlier handling (winsorisation)
- **Feature Engineering**
  - interaction features for HR + Airbnb tasks
  - datetime feature extraction for Airbnb
- **Modeling**
  - logistic regression (HR)
  - linear regression (Airbnb)
- **Evaluation**
  - train/test split comparisons
  - confusion matrix + classification metrics (HR)
  - MSE and R² (Airbnb)
  - k-fold cross-validation (where applicable)

---
