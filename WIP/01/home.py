import pandas as pd
import numpy as np
import streamlit as st
import time
import pickle
import shap

### Import data and model
churn_dataset = pd.read_csv('../datasets/ecommerce_customer_churn_dataset.csv')
with open("../Model/final_model.pkl", "rb") as file:
    model = pickle.load(file)

# Apply customer IDs to rows
start_id = 10001
customer_ids = [f"CX{i}" for i in range(start_id, start_id + len(churn_dataset))]
churn_dataset['Customer_ID'] = np.random.choice(customer_ids, size=len(churn_dataset), replace=False)

# Select specific features
feature_list2 = ['Session_Duration_Avg', 'Pages_Per_Session', 'Cart_Abandonment_Rate', 'Total_Purchases', 'Email_Open_Rate', 'Lifetime_Value',
       'Credit_Balance', 'Churned']
churn_df = churn_dataset[feature_list2].copy()

# Split into data and target features
y = churn_df['Churned']
X = churn_df.drop(columns=['Churned']).copy()

y_proba = model.predict(X)

# --- STREAMLIT
st.title("Customer Churn Analysis")
st.write('Enter customer details to predict churn risk:')

customer_input = st.text_input("Enter Customer ID:")

if customer_input:
  filtered_df = churn_dataset[churn_dataset["Customer_ID"] == customer_input]
  filtered_index = filtered_df.index

  st.write("Matching Results:")
  st.dataframe(filtered_df[['Customer_ID','Age','Gender']])

  if not filtered_df.empty:
    found_id = filtered_df["Customer_ID"].values[0]
    st.success(f"Found Customer ID: {found_id}")
  else:
    st.warning("No matching customer found.")