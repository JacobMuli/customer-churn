import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 Exploratory Data Analysis (EDA) Dashboard")

st.write("This page shows the key EDA insights performed during the analysis phase.")

# Load your processed data
df = pd.read_csv("data/test_processed.csv")

st.subheader("Dataset Overview")
st.write(df.head())

# Distribution of Churn
st.subheader("Churn Distribution")
fig, ax = plt.subplots()
sns.countplot(data=df, x="Churn", ax=ax)
st.pyplot(fig)

# Numeric distributions
numeric_cols = ['Age','Tenure','Usage Frequency','Support Calls','Payment Delay','Total Spend','Last Interaction']

st.subheader("Numeric Feature Distributions")
for col in numeric_cols:
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(df[numeric_cols + ["Churn"]].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)
