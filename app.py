import streamlit as st

# -------------------------------------------------
# PAGE CONFIG (must be first Streamlit command)
# -------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------------------
# HOMEPAGE UI
# -------------------------------------------------
st.title("📊 Customer Churn Analytics Dashboard")
st.write("""
Welcome to the end-to-end churn analytics and prediction platform.
Use the options below to navigate through different analytical sections.
""")

st.markdown("---")

# -------------------------------------------------
# DASHBOARD TILES
# -------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔮 Predict Customer Churn")
    st.write("Make real-time churn predictions using the ML model.")
    if st.button("Go to Prediction"):
        st.switch_page("pages/0_🔮_Prediction_Page.py")

with col2:
    st.markdown("### 📊 EDA Overview")
    st.write("Explore dataset summary, distribution, and churn balance.")
    if st.button("Open EDA Overview"):
        st.switch_page("pages/1_📊_EDA_Overview.py")

with col3:
    st.markdown("### 📈 Numeric Feature Analysis")
    st.write("Deep-dive into numeric variables and their churn impact.")
    if st.button("Explore Numeric Analysis"):
        st.switch_page("pages/2_📈_Numeric_Feature_Analysis.py")

st.markdown("---")

col4, col5 = st.columns(2)

with col4:
    st.markdown("### 📉 Categorical Feature Analysis")
    st.write("Understand how categorical attributes affect churn.")
    if st.button("Explore Categorical Analysis"):
        st.switch_page("pages/3_📉_Categorical_Feature_Analysis.py")

with col5:
    st.markdown("### 📑 Statistical Insights")
    st.write("View significance tests, feature relevance, and churn factors.")
    if st.button("Open Statistical Insights"):
        st.switch_page("pages/4_📑_Statistical_Insights.py")

st.markdown("---")

st.info("Use the navigation buttons above or the sidebar to explore the churn analytics environment.")
