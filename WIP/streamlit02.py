import pandas as pd
import numpy as np
import streamlit as st
import pickle
import shap
import matplotlib.pyplot as plt

### Import data, model and SHAP values
churn_dataset = pd.read_csv('../datasets/ecommerce_customer_churn_dataset.csv')
with open("../Model/final_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("../Model/rf_shap.pkl", "rb") as f:
  loaded = pickle.load(f)

shap_explains = shap.Explanation(
    values=loaded["values"],
    base_values=loaded["base_values"],
    data=loaded["data"],
    feature_names=loaded["feature_names"],
)

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

# Apply model to data
y_preds = model.predict(X)
y_probas = model.predict_proba(X)

# --- STREAMLIT
"""st.title("Customer Churn Analysis")
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
    st.warning("No matching customer found.")"""

class_index = 1
class_explanation = shap_explains[:, :, class_index]

shap.plots.beeswarm(class_explanation)

# Individual prediction plot (e.g., for the very first sample row)
shap.plots.waterfall(class_explanation[0])

import pickle
import matplotlib.pyplot as plt
import shap
import streamlit as st

st.title("🌲 Random Forest SHAP Explainer")

# 1. Load your raw saved data bundle
@st.cache_data  # Cache this so it doesn't reload on every user click
def load_shap_data():
  with open("../Model/rf_shap.pkl", "rb") as f:
    return pickle.load(f)

loaded_data = load_shap_data()

# 2. Reconstruct the 3D SHAP Explanation object
full_explanation = shap.Explanation(
    values=loaded_data["values"],
    base_values=loaded_data["base_values"],
    data=loaded_data["data"],
    feature_names=loaded_data["feature_names"],
)

# 3. Streamlit Sidebar controls for interactivity
st.sidebar.header("Plot Configurations")

# Let the user pick which class to visualize
# Change [0, 1] to match your classes if you have multi-class data
class_index = st.sidebar.selectbox("Select Target Class Index", options=[0, 1])
class_explanation = full_explanation[:, :, class_index]

# --- RENDER THE PLOTS ---

st.subheader("📊 Global Feature Importance (Beeswarm Plot)")

# Step A: Create a clean matplotlib figure context
fig1, ax1 = plt.subplots(figsize=(8, 5))

# Step B: Pass show=False so SHAP draws on our active figure instead of opening a window
shap.plots.beeswarm(class_explanation, show=False)

# Step C: Use Streamlit to display the figure
st.pyplot(fig1)
plt.close(fig1)  # Clear memory

st.divider()

st.subheader("🔍 Individual Prediction Breakdown (Waterfall Plot)")

# Let the user choose a specific row/sample to explain dynamically
sample_index = st.number_input(
    "Select Sample Row Index to Explain",
    min_value=0,
    max_value=len(class_explanation) - 1,
    value=0,
)

fig2, ax2 = plt.subplots(figsize=(8, 5))
# Slice the single row selected by the user
shap.plots.waterfall(class_explanation[sample_index], show=False)
st.pyplot(fig2)
plt.close(fig2)
