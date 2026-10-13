import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pickle
import shap

st.set_page_config(layout="wide")
# Import Data
churn_df = pd.read_csv('churn_analysis_results.csv').copy()
churned_customers = churn_df['Churned'].value_counts()
not_churned_customers = churn_df[churn_df['Churned'] == 0].shape[0]
churn_rate = churned_customers[1] / (churned_customers[1] + not_churned_customers) * 100

cx_high = churn_df[churn_df['Churn_Risk'] == 'High'].shape[0]
cx_medium = churn_df[churn_df['Churn_Risk'] == 'Medium'].shape[0]
cx_low = churn_df[churn_df['Churn_Risk'] == 'Low'].shape[0]

cx_strings = []
cx_string = ""
if cx_high > 0 and cx_medium > 0 and cx_low > 0:
    cx_strings.append(f"There are {cx_high} customers at high churn risk, ")
    cx_strings.append(f"{cx_medium} at medium risk, and ")
    cx_strings.append(f"{cx_low} at low risk of churn.")
    cx_string = "".join(cx_strings)
elif cx_high > 0:
    cx_string = f"There are {cx_high} customers at high churn risk."
elif cx_medium > 0:
    cx_string = f"There are {cx_medium} customers at medium churn risk."
elif cx_low > 0:
    cx_string = f"There are {cx_low} customers at low churn risk."

high_value_customers = churn_df[churn_df['Lifetime_Value'] >= 2000].shape[0]
high_value_churned = churn_df[(churn_df['Lifetime_Value'] >= 2000) & (churn_df['Churned'] == 1)].shape[0]
high_value_churned_percent = (high_value_churned / high_value_customers) * 100 if high_value_customers > 0 else 0
hv_customers = churn_df[churn_df['Lifetime_Value'] >= 2000].copy()

hvc_high = hv_customers[hv_customers['Churn_Risk'] == 'High'].shape[0]
hvc_med = hv_customers[hv_customers['Churn_Risk'] == 'Medium'].shape[0]
hvc_low = hv_customers[hv_customers['Churn_Risk'] == 'Low'].shape[0]

hv_strings = []
hv_string = ""
if hvc_high > 0 and hvc_med > 0 and hvc_low > 0:
    hv_strings.append(f"There are {hvc_high} high value customers at high churn risk, ")
    hv_strings.append(f"{hvc_med} at medium risk, and ")
    hv_strings.append(f"{hvc_low} at low risk of churn.")
    hv_string = "".join(hv_strings)
elif hvc_high > 0 and hvc_med > 0:
    hv_strings.append(f"There are {hvc_high} high value customers at high churn risk, ")
    hv_strings.append(f"and {hvc_med} at medium risk of churn.")
    hv_string = "".join(hv_strings)
elif hvc_med > 0 and hvc_low > 0:
    hv_strings.append(f"There are {hvc_med} high value customers at medium churn risk, ")
    hv_strings.append(f"and {hvc_low} at low risk of churn.")
    hv_string = "".join(hv_strings)
elif hvc_high > 0:
    hv_strings.append(f"There are {hvc_high} high value customers at high churn risk of churn.")
    hv_string = "".join(hv_strings)
elif hvc_med > 0:
    hv_strings.append(f"There are {hvc_med} high value customers at medium churn risk of churn.")
    hv_string = "".join(hv_strings)
elif hvc_low > 0:
    hv_strings.append(f"There are {hvc_low} high value customers at low churn risk of churn.")
    hv_string = "".join(hv_strings)

bar_colours = ['#FEC3E1','#B9D9D5',  '#9BABBE']
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.markdown("<span style='font-weight: bold;'>Current Churn</span>:", unsafe_allow_html=True)
        st.write(f"Currently, {churn_rate:.2f}% of customers are suspected to have churned. This represents {churned_customers[1]} out of {churned_customers[1] + not_churned_customers} total customers.")

        # Bar chart
        risk_counts = churn_df['Churn_Risk'].value_counts()
        ax1.bar(risk_counts.index, risk_counts.values, color=bar_colours)
        ax1.set_xlabel('Churn Risk Level')
        ax1.set_ylabel('Number of Customers')
        ax1.set_title('Distribution of Churn Risk Levels')
        st.pyplot(fig1)

        st.write(cx_string)

with col2:
    with st.container(border=True):
        st.markdown("<span style='font-weight: bold;'>High Value Churn</span>:", unsafe_allow_html=True)
        st.write(f"{high_value_churned_percent:.2f}% of high value customers are suspected to have churned. This represents {high_value_churned} out of {high_value_customers} total high value customers.")
        # Bar chart
        risk_counts_hv = hv_customers['Churn_Risk'].value_counts()
        ax2.bar(risk_counts_hv.index, risk_counts_hv.values, color=bar_colours)
        ax2.set_xlabel('Churn Risk Level')
        ax2.set_ylabel('Number of Customers')
        ax2.set_title('Distribution of High Value Churn Risk Levels')
        st.pyplot(fig2)

        st.write(hv_string)