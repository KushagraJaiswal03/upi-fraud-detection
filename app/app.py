"""
UPI Fraud Detection — Streamlit app.

Two pages:
  1. Analytics Dashboard — EDA insights from the training dataset.
  2. Fraud Checker — enter a transaction's key details, get a live risk score.
"""

import sys
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(__file__))
from features import build_features, get_feature_lists

st.set_page_config(page_title="UPI Fraud Detection", page_icon="🔒", layout="wide")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fraud_pipeline.joblib")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "upi_transactions.csv")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


bundle = load_model()
pipeline = bundle["pipeline"]
model_name = bundle["model_name"]
defaults = bundle["defaults"]
feature_columns = bundle["feature_columns"]

st.title("🔒 UPI Fraud Detection")
st.caption(f"Deployed model: **{model_name}** · Trained on {load_data().shape[0]:,} transactions "
           f"({load_data()['is_fraud'].mean():.1%} fraud rate)")

page = st.sidebar.radio("Navigate", ["📊 Analytics Dashboard", "🔍 Fraud Checker"])

# =======================================================================
# PAGE 1 — ANALYTICS DASHBOARD
# =======================================================================
if page == "📊 Analytics Dashboard":
    df = load_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total transactions", f"{len(df):,}")
    col2.metric("Fraud cases", f"{df['is_fraud'].sum():,}")
    col3.metric("Fraud rate", f"{df['is_fraud'].mean():.2%}")
    col4.metric("Avg. fraud amount", f"₹{df.loc[df.is_fraud==1, 'amount'].mean():,.0f}")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Class balance")
        fig, ax = plt.subplots(figsize=(5, 4))
        df["is_fraud"].value_counts().rename({0: "Legit", 1: "Fraud"}).plot(
            kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with c2:
        st.subheader("Amount distribution by label")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.kdeplot(data=df, x="amount", hue="is_fraud", common_norm=False, fill=True, ax=ax)
        st.pyplot(fig)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Fraud rate by hour of day")
        fig, ax = plt.subplots(figsize=(5, 4))
        df.groupby("transaction_time_of_day")["is_fraud"].mean().plot(kind="bar", ax=ax, color="#C44E52")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Fraud rate")
        st.pyplot(fig)

    with c4:
        st.subheader("Fraud rate by transaction type")
        fig, ax = plt.subplots(figsize=(5, 4))
        df.groupby("transaction_type")["is_fraud"].mean().plot(kind="bar", ax=ax, color="#8172B2")
        ax.set_ylabel("Fraud rate")
        st.pyplot(fig)

    st.divider()
    st.subheader("What drives the model's fraud predictions")
    fitted_model = pipeline.named_steps["model"]
    if hasattr(fitted_model, "feature_importances_"):
        preprocessor = pipeline.named_steps["preprocess"]
        numeric_cols = bundle["numeric_cols"]
        categorical_cols = bundle["categorical_cols"]
        ohe = preprocessor.named_transformers_["cat"]
        cat_names = list(ohe.get_feature_names_out(categorical_cols))
        all_names = numeric_cols + cat_names
        importances = pd.Series(fitted_model.feature_importances_, index=all_names)
        top15 = importances.sort_values(ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(8, 6))
        top15.sort_values().plot(kind="barh", ax=ax, color="#55A868")
        ax.set_xlabel("Feature importance")
        st.pyplot(fig)

    st.info(
        "Note: this dataset is synthetic and was generated with strongly "
        "separable fraud patterns, so the model reaches very high accuracy "
        "here. On real, noisier data, expect lower but still strong "
        "precision/recall — the pipeline and feature set are what transfer, "
        "not this exact accuracy number.",
        icon="ℹ️",
    )

# =======================================================================
# PAGE 2 — FRAUD CHECKER
# =======================================================================
else:
    st.subheader("Check a transaction")
    st.caption("Fill in what you know — anything left out uses a typical/default value.")

    with st.form("txn_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            amount = st.number_input("Amount (₹)", min_value=0.0, value=3000.0, step=100.0)
            transaction_type = st.selectbox("Transaction type", ["payment", "collection_request"])
            session_source = st.selectbox("Session source", ["app", "link"])
            authorization_method = st.selectbox("Authorization method", ["pin", "otp"])
            pin_entry_method = st.selectbox("PIN entry method", ["manual", "pasted"])

        with c2:
            receiver_account_age = st.number_input("Receiver account age (days)", min_value=0, value=180)
            receiver_transaction_history = st.number_input("Receiver's past transaction count", min_value=0, value=40)
            handle_verification_status = st.selectbox("UPI handle verification", ["verified", "unverified"])
            handle_typo_analysis = st.selectbox("Handle typo check", ["none", "typo_squatting"])
            handle_registration_pattern = st.selectbox("Handle registration pattern", ["none", "recent"])

        with c3:
            merchant_category_code = st.selectbox(
                "Merchant category",
                ["food", "services", "utilities", "retail", "entertainment", "unknown"])
            unusual_device_flag = st.checkbox("Unusual device detected")
            unusual_ip_flag = st.checkbox("Unusual IP detected")
            unusual_location_flag = st.checkbox("Unusual location detected")
            unusual_transaction_amount_flag = st.checkbox("Amount unusual for this sender")
            time_pressure_indicators = st.slider("Time-pressure indicators (0=none, 5=high)", 0, 5, 0)

        submitted = st.form_submit_button("Check transaction", type="primary", use_container_width=True)

    if submitted:
        row = dict(defaults)  # start from typical values for every feature
        row.update({
            "amount": amount,
            "transaction_type": transaction_type,
            "session_source": session_source,
            "authorization_method": authorization_method,
            "pin_entry_method": pin_entry_method,
            "receiver_account_age": receiver_account_age,
            "receiver_transaction_history": receiver_transaction_history,
            "handle_verification_status": handle_verification_status,
            "handle_typo_analysis": handle_typo_analysis,
            "handle_registration_pattern": handle_registration_pattern,
            "merchant_category_code": merchant_category_code,
            "unusual_device_flag": int(unusual_device_flag),
            "unusual_ip_flag": int(unusual_ip_flag),
            "unusual_location_flag": int(unusual_location_flag),
            "unusual_transaction_amount_flag": int(unusual_transaction_amount_flag),
            "time_pressure_indicators": time_pressure_indicators,
        })

        X_input = pd.DataFrame([row])[feature_columns]
        proba = pipeline.predict_proba(X_input)[0, 1]
        pred = int(proba >= 0.5)

        st.divider()
        if pred == 1:
            st.error(f"⚠️ **High fraud risk** — estimated probability: {proba:.1%}")
        elif proba >= 0.2:
            st.warning(f"🟡 **Elevated risk** — estimated probability: {proba:.1%}")
        else:
            st.success(f"✅ **Low fraud risk** — estimated probability: {proba:.1%}")

        st.progress(min(proba, 1.0))
        st.caption(
            "This score reflects patterns learned from historical UPI transaction data. "
            "Use it as one input alongside your own judgment, not as a sole decision-maker."
        )
