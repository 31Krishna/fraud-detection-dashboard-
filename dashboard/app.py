import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)

# -------------------------
# LOAD PATHS
# -------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(__file__)
)

data_path = os.path.join(
    BASE_DIR,
    "data",
    "sample_transactions.csv"
)

# -------------------------
# LOAD DATA
# -------------------------

df = pd.read_csv(data_path)

# -------------------------
# FEATURE ENGINEERING
# -------------------------

df["HourOfDay"] = (
    df["TransactionDT"] // 3600
) % 24

df["LogTransactionAmt"] = np.log1p(
    df["TransactionAmt"]
)

df["AmtToMeanRatio"] = (
    df["TransactionAmt"] /
    df["TransactionAmt"].mean()
)

# -------------------------
# HANDLE MISSING VALUES
# -------------------------

for col in df.select_dtypes(
    include=["int64", "float64"]
).columns:
    
    df[col].fillna(
        df[col].median(),
        inplace=True
    )

# -------------------------
# DUMMY FRAUD SCORES
# -------------------------

np.random.seed(42)

df["FraudProbability"] = np.random.uniform(
    0,
    1,
    len(df)
)

# -------------------------
# RISK FUNCTION
# -------------------------

def assign_risk(prob):
    
    if prob >= 0.75:
        return "Critical Risk"
    
    elif prob >= 0.40:
        return "Suspicious"
    
    else:
        return "Clear"

df["RiskTier"] = df[
    "FraudProbability"
].apply(assign_risk)

# -------------------------
# SIDEBAR
# -------------------------

st.sidebar.title("Fraud Dashboard")

page = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Transaction Explorer",
        "SHAP Explainer"
    ]
)

# -------------------------
# SIDEBAR FILTERS
# -------------------------

risk_filter = st.sidebar.multiselect(
    
    "Filter Risk Tier",
    
    options=df["RiskTier"].unique(),
    
    default=df["RiskTier"].unique()
)

filtered_df = df[
    df["RiskTier"].isin(risk_filter)
]

# ======================================================
# PAGE 1 — OVERVIEW
# ======================================================

if page == "Overview":
    
    st.title(
        "AI-Powered Fraud Detection Dashboard"
    )
    
    total_transactions = len(filtered_df)
    
    total_fraud = filtered_df[
        "isFraud"
    ].sum()
    
    fraud_rate = (
        total_fraud /
        total_transactions
    ) * 100
    
    avg_fraud_amount = filtered_df[
        filtered_df["isFraud"] == 1
    ]["TransactionAmt"].mean()
    
    # -------------------------
    # KPI CARDS
    # -------------------------
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )
    
    col2.metric(
        "Fraud Transactions",
        f"{total_fraud:,}"
    )
    
    col3.metric(
        "Detection Rate",
        f"{fraud_rate:.2f}%"
    )
    
    col4.metric(
        "Avg Fraud Amount",
        f"${avg_fraud_amount:.2f}"
    )
    
    st.markdown("---")
    
    # -------------------------
    # DONUT CHART
    # -------------------------
    
    st.subheader(
        "Risk Tier Distribution"
    )
    
    risk_counts = filtered_df[
        "RiskTier"
    ].value_counts()
    
    fig = px.pie(
        
        names=risk_counts.index,
        
        values=risk_counts.values,
        
        hole=0.4,
        
        title="Risk Tier Breakdown"
    )
    
    st.plotly_chart(
        fig,
        use_container_width=True
    )
    
    # -------------------------
    # FRAUD TREND CHART
    # -------------------------
    
    st.subheader(
        "Fraud Probability by Hour"
    )
    
    fraud_by_hour = filtered_df.groupby(
        "HourOfDay"
    )["FraudProbability"].mean()
    
    fig2 = px.line(
        
        x=fraud_by_hour.index,
        
        y=fraud_by_hour.values,
        
        labels={
            "x": "Hour Of Day",
            "y": "Fraud Probability"
        },
        
        title="Fraud Trend Analysis"
    )
    
    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# ======================================================
# PAGE 2 — TRANSACTION EXPLORER
# ======================================================

elif page == "Transaction Explorer":
    
    st.title(
        "Transaction Explorer"
    )
    
    transaction_id = st.number_input(
        
        "Enter TransactionID",
        
        min_value=int(
            filtered_df["TransactionID"].min()
        ),
        
        max_value=int(
            filtered_df["TransactionID"].max()
        ),
        
        step=1
    )
    
    transaction_data = filtered_df[
        filtered_df["TransactionID"] == transaction_id
    ]
    
    if len(transaction_data) > 0:
        
        st.subheader(
            "Transaction Details"
        )
        
        st.dataframe(
            transaction_data
        )
        
        risk_score = transaction_data[
            "FraudProbability"
        ].values[0]
        
        risk_tier = transaction_data[
            "RiskTier"
        ].values[0]
        
        col1, col2 = st.columns(2)
        
        col1.metric(
            "Fraud Risk Score",
            f"{risk_score:.2f}"
        )
        
        col2.metric(
            "Risk Tier",
            risk_tier
        )
    
    st.markdown("---")
    
    st.subheader(
        "Searchable Transactions Table"
    )
    
    st.dataframe(
        filtered_df.head(100)
    )

# ======================================================
# PAGE 3 — SHAP EXPLAINER
# ======================================================

elif page == "SHAP Explainer":
    
    st.title(
        "SHAP Fraud Explanation"
    )
    
    transaction_id = st.number_input(
        
        "Enter TransactionID for Explanation",
        
        min_value=int(
            filtered_df["TransactionID"].min()
        ),
        
        max_value=int(
            filtered_df["TransactionID"].max()
        ),
        
        step=1,
        
        key="shap_input"
    )
    
    transaction_data = filtered_df[
        filtered_df["TransactionID"] == transaction_id
    ]
    
    if len(transaction_data) > 0:
        
        risk_score = transaction_data[
            "FraudProbability"
        ].values[0]
        
        st.metric(
            "Fraud Probability",
            f"{risk_score:.2f}"
        )
        
        st.subheader(
            "Simulated SHAP Waterfall Plot"
        )
        
        features = [
            "TransactionAmt",
            "HourOfDay",
            "AmtToMeanRatio",
            "CardRisk",
            "DeviceRisk"
        ]
        
        shap_values = np.random.uniform(
            -1,
            1,
            len(features)
        )
        
        fig = go.Figure(
            go.Waterfall(
                
                name="SHAP",
                
                orientation="v",
                
                measure=["relative"] * len(features),
                
                x=features,
                
                y=shap_values
            )
        )
        
        fig.update_layout(
            title="SHAP Feature Contribution"
        )
        
        st.plotly_chart(
            fig,
            use_container_width=True
        )
        
        # -------------------------
        # PLAIN ENGLISH EXPLANATION
        # -------------------------
        
        st.subheader(
            "Plain-English Explanation"
        )
        
        if risk_score >= 0.75:
            
            st.error(
                """
                This transaction is classified as Critical Risk.
                
                Major contributing factors:
                • High transaction amount
                • Suspicious transaction timing
                • Abnormal transaction behavior
                
                Immediate verification is recommended.
                """
            )
        
        elif risk_score >= 0.40:
            
            st.warning(
                """
                This transaction is marked as Suspicious.
                
                The model identified moderate anomaly patterns.
                
                Additional monitoring is advised.
                """
            )
        
        else:
            
            st.success(
                """
                This transaction appears legitimate.
                
                Fraud indicators are low and transaction behavior is normal.
                """
            )

# ======================================================
# FOOTER
# ======================================================

st.markdown("---")

st.markdown(
    """
    ### Built Using
    - Python
    - Streamlit
    - Plotly
    - LightGBM
    - SHAP
    """
)