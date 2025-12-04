import streamlit as st

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------------------
# HOMEPAGE TITLE
# -------------------------------------------------
st.title("📊 Customer Churn Analytics Dashboard")
st.write("""
Welcome to the end-to-end churn analytics and prediction platform.
Use the navigation tiles below to explore different analytical modules.
""")

st.markdown("---")

# -------------------------------------------------
# DASHBOARD TILE ROW 1
# -------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔮 Predict Customer Churn")
    st.write("Make real-time predictions using the trained ML model.")
    if st.button("Go to Prediction"):
        st.switch_page("pages/0_Prediction_Page.py")

with col2:
    st.markdown("### 📊 EDA Overview")
    st.write("Explore dataset summary, distributions, and churn patterns.")
    if st.button("Open EDA Overview"):
        st.switch_page("pages/1_EDA_Dashboard.py")

with col3:
    st.markdown("### 📈 Numeric Feature Analysis")
    st.write("Analyze numerical features and their impact on churn.")
    if st.button("Explore Numeric Analysis"):
        st.switch_page("pages/2_Numeric_Feature_Analysis.py")

st.markdown("---")

# -------------------------------------------------
# DASHBOARD TILE ROW 2
# -------------------------------------------------
col4, col5 = st.columns(2)

with col4:
    st.markdown("### 📉 Categorical Feature Analysis")
    st.write("Visualize and compare categorical attributes across churn classes.")
    if st.button("Explore Categorical Analysis"):
        st.switch_page("pages/3_Categorical_Feature_Analysis.py")

with col5:
    st.markdown("### 📑 Statistical Insights")
    st.write("Review t-tests and statistical significance of features.")
    if st.button("Open Statistical Insights"):
        st.switch_page("pages/4_Statistical_Insights.py")

st.markdown("---")

st.info("💡 Tip: You can also use the left sidebar to navigate between pages.")
