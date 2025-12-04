import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📉 Categorical Feature Analysis")

df = pd.read_csv("data/test_processed.csv")

categorical_cols = [
    'Gender',
    'Subscription Type_Basic','Subscription Type_Standard','Subscription Type_Premium',
    'Contract Length_Annual','Contract Length_Monthly','Contract Length_Quarterly'
]

st.write("Visualizing relationships between categorical features and churn.")

for col in categorical_cols:
    st.subheader(f"📌 {col}")

    fig, ax = plt.subplots()
    sns.countplot(data=df, x=col, hue="Churn", ax=ax)
    st.pyplot(fig)
