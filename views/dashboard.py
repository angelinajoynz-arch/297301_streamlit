import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pickle
import shap

st.set_page_config(layout="wide")
# Import Data
churn_df = pd.read_csv('churn_analysis_results.csv')

col1, col2, col3 = st.columns(3)
with col1:
    # Bar chart showing high, med and low risk customers
    risk_counts = churn_df['Churn_Risk'].value_counts()
    plt.bar(risk_counts.index, risk_counts.values)
    plt.xlabel('Churn Risk Level')
    plt.ylabel('Number of Customers')
    plt.title('Distribution of Churn Risk Levels')
    st.pyplot(plt)

with col2:
    st.subheader("Bar Chart")

with col3:
    st.subheader("Area Chart")