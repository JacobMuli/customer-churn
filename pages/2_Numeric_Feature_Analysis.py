import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📈 Numeric Feature Analysis")

df = pd.read_csv("data/test_processed.csv")

numeric_cols = [
    'Age','Tenure','Usage Frequency','Support Calls',
    'Payment Delay','Total Spend','Last Interaction',
    'Avg_Monthly_Spend','Support_Intensity','Recency_Tenure_Ratio'
]

st.write("""
Below are distribution and boxplot analyses for numeric features to understand churn patterns.
""")

for col in numeric_cols:
    st.subheader(f"📌 {col}")

    # Histogram
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)

    # Boxplot vs Churn
    fig, ax = plt.subplots()
    sns.boxplot(x=df["Churn"], y=df[col], ax=ax)
    st.pyplot(fig)
