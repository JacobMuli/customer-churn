import streamlit as st
import pandas as pd
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📑 Statistical Insights")

df = pd.read_csv("data/test_processed.csv")

numeric_cols = [
    'Age','Tenure','Usage Frequency','Support Calls',
    'Payment Delay','Total Spend','Last Interaction',
    'Avg_Monthly_Spend','Support_Intensity','Recency_Tenure_Ratio'
]

st.write("""
This page shows t-test statistical significance to determine whether numerical features
differ significantly between churners and non-churners.
""")

results = []

for col in numeric_cols:
    churn0 = df[df["Churn"] == 0][col]
    churn1 = df[df["Churn"] == 1][col]
    stat, p = ttest_ind(churn0, churn1, equal_var=False)
    results.append({"Feature": col, "p-value": p})

results_df = pd.DataFrame(results)
st.dataframe(results_df)

st.subheader("Interpretation")
st.write("""
- **p < 0.05** indicates statistically significant difference between churn and non-churn groups.
- Usually, features like `Support Calls`, `Tenure`, and `Payment Delay` show strong significance.
""")
