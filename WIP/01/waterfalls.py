import pandas as pd
import numpy as np
import streamlit as st
import pickle
import shap
import matplotlib.pyplot as plt

# Import data
churn_dataset = pd.read_csv('datasets/customer_churn.csv')
churn_df = pd.read_csv('churn_analysis_results.csv')

# Import shap values
@st.cache_data  # Cache this so it doesn't reload on every user click
def load_shap_data():
  with open("Model/rf_shap.pkl", "rb") as f:
    return pickle.load(f)

loaded_shap = load_shap_data()

shap_explains = shap.Explanation(
    values=loaded_shap["values"],
    base_values=loaded_shap["base_values"],
    data=loaded_shap["data"],
    feature_names=loaded_shap["feature_names"],
)

st.title("Customer Churn Analysis")

filtered_index = None
customer_input = st.text_input("Enter Customer ID (ie. CX20201) to check risk:")

if customer_input:
  filtered_df = churn_dataset[churn_dataset["Customer_ID"] == customer_input]
  filtered_index = filtered_df.index

  st.write("Matching Results:")
  st.dataframe(filtered_df[['Customer_ID','Age','Gender']])

  if not filtered_df.empty:
    found_id = filtered_df["Customer_ID"].values[0]
    st.success(f"Found Customer ID: {found_id}")

    class_explanation = shap_explains[:, :, 1][filtered_index]
    exp = shap_explains[filtered_index, :, 1]
    f_x = exp.values.sum() + exp.base_values
    fx_percent = f_x * 100

    # Show plot
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_explains[:, :, 1][filtered_index][0], show=False)
    plt.title('SHAP Waterfall Plot for Customer ' + customer_input + f', Churn probability: {fx_percent[0]:.2f}%')
    st.pyplot(fig)
    plt.close(fig)

    # Customer-specific feature values
    feature_vals = shap_explains[filtered_index, :, 1].data.tolist()
    flat_features = feature_vals[0]

    session_dur_avg = flat_features[0]
    pagesper_session = flat_features[1]
    cart_abndnmnt_rate = flat_features[2]
    ttl_purchases = flat_features[3]
    email_opens = flat_features[4]
    ltv = flat_features[5]
    credit = flat_features[6]

  else:
    st.warning("No matching customer found.")
