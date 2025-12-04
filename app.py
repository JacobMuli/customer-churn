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

# Streamlit UI
st.set_page_config(page_title="Customer Churn Prediction", page_icon="🔮")

st.title("🔮 Customer Churn Prediction App")
st.write("Fill the form below to predict whether a customer is likely to churn.")

# ----------------------------
# Streamlit Form Inputs
# ----------------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        tenure = st.number_input("Tenure (Months)", min_value=0, value=12)
        usage = st.number_input("Usage Frequency", min_value=0, value=10)
        support_calls = st.number_input("Support Calls", min_value=0, value=2)

    with col2:
        payment_delay = st.number_input("Payment Delay (Days)", min_value=0, value=1)
        total_spend = st.number_input("Total Spend", min_value=0, value=500)
        last_interaction = st.number_input("Last Interaction (Days)", min_value=0, value=30)

    subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    contract = st.selectbox("Contract Length", ["Annual", "Monthly", "Quarterly"])

    submitted = st.form_submit_button("Predict Churn")

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess_input():
    gender_val = 1 if gender == "Female" else 0

    sub_basic = 1 if subscription == "Basic" else 0
    sub_standard = 1 if subscription == "Standard" else 0
    sub_premium = 1 if subscription == "Premium" else 0

    con_annual = 1 if contract == "Annual" else 0
    con_monthly = 1 if contract == "Monthly" else 0
    con_quarterly = 1 if contract == "Quarterly" else 0

    avg_monthly_spend = total_spend / max(tenure, 1)
    support_intensity = support_calls / max(usage, 1)
    recency_tenure_ratio = last_interaction / max(tenure, 1)

    # Construct feature dictionary
    features = {
        "Age": age,
        "Gender": gender_val,
        "Tenure": tenure,
        "Usage Frequency": usage,
        "Support Calls": support_calls,
        "Payment Delay": payment_delay,
        "Total Spend": total_spend,
        "Last Interaction": last_interaction,
        "Subscription Type_Basic": sub_basic,
        "Subscription Type_Standard": sub_standard,
        "Subscription Type_Premium": sub_premium,
        "Contract Length_Annual": con_annual,
        "Contract Length_Monthly": con_monthly,
        "Contract Length_Quarterly": con_quarterly,
        "Avg_Monthly_Spend": avg_monthly_spend,
        "Support_Intensity": support_intensity,
        "Recency_Tenure_Ratio": recency_tenure_ratio,
    }

    df = pd.DataFrame([features])

    # Apply label encoders
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))

    # Add any missing training columns
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # Reorder to match training
    df = df[feature_names]

    # Scale for model
    scaled = scaler.transform(df)
    return scaled

# ----------------------------
# Prediction Execution
# ----------------------------
if submitted:
    X_processed = preprocess_input()

    proba = model.predict_proba(X_processed)[0][1]
    pred = int(proba > 0.5)

    st.subheader("🔍 Prediction Result")
    st.write(f"**Churn Probability:** `{proba:.2f}`")

    if pred == 1:
        st.error("⚠️ The customer is **likely to churn**.")
    else:
        st.success("✅ The customer is **not likely to churn**.")
