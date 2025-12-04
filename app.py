import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Load Artifacts
# ----------------------------
model = joblib.load("model/rf_model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")
feature_names = joblib.load("model/feature_names.pkl")

st.set_page_config(page_title="Customer Churn Prediction App")

st.title("📊 Customer Churn Prediction App")
st.write("Provide customer details below to predict the likelihood of churn.")

# ----------------------------
# Streamlit Form Inputs
# ----------------------------
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", min_value=18, max_value=100, value=35)
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Tenure = st.number_input("Tenure (months)", min_value=0, value=12)
        Usage = st.number_input("Usage Frequency", min_value=0, value=10)
        SupportCalls = st.number_input("Support Calls", min_value=0, value=2)

    with col2:
        PaymentDelay = st.number_input("Payment Delay (days)", min_value=0, value=1)
        TotalSpend = st.number_input("Total Spend", min_value=0, value=500)
        LastInteraction = st.number_input("Last Interaction (days)", min_value=0, value=30)

    SubscriptionType = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    ContractLength = st.selectbox("Contract Length", ["Annual", "Monthly", "Quarterly"])

    submitted = st.form_submit_button("Predict Churn")

# ----------------------------
# Prediction Logic
# ----------------------------
def preprocess_input():
    # Convert gender
    gender_val = 1 if Gender == "Female" else 0

    # One-hot subscription
    sub_basic = 1 if SubscriptionType == "Basic" else 0
    sub_standard = 1 if SubscriptionType == "Standard" else 0
    sub_premium = 1 if SubscriptionType == "Premium" else 0

    # One-hot contract
    con_annual = 1 if ContractLength == "Annual" else 0
    con_monthly = 1 if ContractLength == "Monthly" else 0
    con_quarterly = 1 if ContractLength == "Quarterly" else 0

    # Engineered features
    Avg_Monthly_Spend = TotalSpend / max(Tenure, 1)
    Support_Intensity = SupportCalls / max(Usage, 1)
    Recency_Tenure_Ratio = LastInteraction / max(Tenure, 1)

    # Create initial feature dict
    input_dict = {
        "Age": Age,
        "Gender": gender_val,
        "Tenure": Tenure,
        "Usage Frequency": Usage,
        "Support Calls": SupportCalls,
        "Payment Delay": PaymentDelay,
        "Total Spend": TotalSpend,
        "Last Interaction": LastInteraction,
        "Subscription Type_Basic": sub_basic,
        "Subscription Type_Standard": sub_standard,
        "Subscription Type_Premium": sub_premium,
        "Contract Length_Annual": con_annual,
        "Contract Length_Monthly": con_monthly,
        "Contract Length_Quarterly": con_quarterly,
        "Avg_Monthly_Spend": Avg_Monthly_Spend,
        "Support_Intensity": Support_Intensity,
        "Recency_Tenure_Ratio": Recency_Tenure_Ratio,
    }

    # Convert to DataFrame
    df = pd.DataFrame([input_dict])

    # Apply label encoders (if any)
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))

    # Add missing columns
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # Reorder
    df = df[feature_names]

    # Scale
    df_scaled = scaler.transform(df)

    return df_scaled


# ----------------------------
# Run Prediction
# ----------------------------
if submitted:
    processed = preprocess_input()
    proba = model.predict_proba(processed)[0][1]
    pred = int(proba > 0.5)

    st.subheader("🔍 Prediction Result")
    st.write(f"**Churn Probability:** `{proba:.2f}`")

    if pred == 1:
        st.error("⚠️ The customer is **likely to churn**.")
    else:
        st.success("✅ The customer is **not likely to churn**.")
