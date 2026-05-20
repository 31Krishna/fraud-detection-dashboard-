# AI-Powered Fraud Detection and Risk Analysis Dashboard

## 📌 Project Overview

This project is an end-to-end Machine Learning based Fraud Detection System developed using the IEEE-CIS Fraud Detection dataset. The system detects suspicious financial transactions, analyzes fraud patterns, and provides an interactive dashboard for fraud monitoring and risk analysis.

The project combines:
- Machine Learning
- Explainable AI (SHAP)
- Fraud Risk Segmentation
- Interactive Streamlit Dashboard
- Business Insights & Visualization

---

# 🎯 Objectives

- Detect fraudulent transactions using Machine Learning
- Handle severe class imbalance using SMOTE
- Explain fraud predictions using SHAP
- Segment transactions into risk categories
- Build a live fraud monitoring dashboard

---

# 🛠️ Technologies Used

| Tool / Library | Purpose |
|---|---|
| Python | Main programming language |
| Pandas / NumPy | Data processing |
| Scikit-learn | ML preprocessing & evaluation |
| LightGBM | Primary fraud detection model |
| XGBoost | Model comparison |
| Isolation Forest | Anomaly detection |
| SHAP | Explainable AI |
| Streamlit | Dashboard development |
| Plotly | Interactive visualizations |
| Seaborn / Matplotlib | Charts & analytics |
| imbalanced-learn | SMOTE balancing |

---

# 📂 Dataset

Dataset Used:
IEEE-CIS Fraud Detection Dataset

Files:
- train_transaction.csv
- train_identity.csv

Dataset Link:
https://www.kaggle.com/c/ieee-fraud-detection/data

---

# ⚙️ Project Workflow

## TASK 1 — Exploratory Data Analysis
- Merged transaction and identity datasets
- Analysed fraud imbalance
- Investigated missing values
- Created correlation heatmaps
- Visualized transaction amount distribution

## TASK 2 — Preprocessing & Feature Engineering
- Dropped columns with >50% missing values
- Median/Mode imputation
- Label encoding
- RobustScaler normalization
- SMOTE balancing
- Engineered fraud features

## TASK 3 — Model Training & Evaluation
Models Used:
- LightGBM
- XGBoost
- Isolation Forest

Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC

Additional:
- Threshold Optimization
- ROC Curves
- Precision-Recall Curves

## TASK 4 — Explainable AI (SHAP)
- SHAP Summary Plot
- SHAP Waterfall Plot
- SHAP Dependence Plot
- Fraud explanation analysis

## TASK 5 — Risk Segmentation
Transactions segmented into:
- 🔴 Critical Risk
- 🟡 Suspicious
- 🟢 Clear

Analysis Performed:
- Fraud trend analysis
- Hour-of-day patterns
- Transaction behavior analysis
- Risk tier comparisons

## TASK 6 — Streamlit Dashboard
Dashboard Features:
- KPI overview
- Risk tier visualization
- Fraud trend chart
- Transaction explorer
- Fraud risk scoring

---

# 📊 Dashboard Features

## Overview Page
- Total transactions
- Fraud transaction count
- Detection rate
- Average fraud amount
- Risk tier donut chart
- Fraud trend analysis

## Transaction Explorer
- Search by TransactionID
- Fraud risk score
- Transaction details table

---

# 📈 Key Insights

- Fraudulent transactions are highly imbalanced (~3.5%)
- PR-AUC is more meaningful than accuracy in fraud detection
- Certain transaction hours show elevated fraud probability
- SHAP revealed Transaction Amount and behavioral features as strong fraud indicators
- Risk segmentation improves fraud monitoring efficiency
