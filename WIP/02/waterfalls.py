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

# -- Customer blurb function 
def customer_blurb():
    churn_label = str(customer_churn_label)

    if churn_label == 'Medium':
        st.warning("This customer is at medium risk of churning.")
    elif churn_label == 'High':
        st.error("This customer is at high risk of churning.")
    else:
        st.write("This customer is at low risk of churning.")

    if (ltv >= 1950) and (churn_label == 'High'):
        issues.append("High lifetime value with high churn risk, indicating potential loss of valuable customer.")
    elif ltv >= 1950:
        issues.append("High lifetime value, which can indicate a valuable customer at risk.")
    else:
        nonissues.append("lifetime value")

    if int(cart_abndnmnt_rate) >= 70:
        issues.append("High cart abandonment rate, which may indicate dissatisfaction with the purchasing process.")
    else:
        nonissues.append("cart abandonment rate")

    if int(ttl_purchases) <= 6:
        issues.append("Low total purchases made, suggesting limited engagement.")
    else:
        nonissues.append("total purchases")

    if email_opens < 10:
        issues.append("Low email open rate, which may indicate disengagement.")
    else:
        nonissues.append("email open rate")

    if session_dur_avg < 20:
        issues.append("Low session duration, which may indicate a poor browsing experience.")
    else:
        nonissues.append("session duration")

    if pagesper_session < 7:
        issues.append("Low pages per session, which may indicate a lack of interest.")
    else:
        nonissues.append("pages per session")

    if len(issues) > 0:
        st.write("This customer has the following characteristics:")
        for i in issues:
            st.write(f"- {i}")

    if len(issues) > 0 and len(nonissues) > 0:
        if len(nonissues) == 1:
            st.write("They are otherwise within acceptable ranges for " + nonissues[0] + ".")
        else:
            last_phrase = len(nonissues) - 1
            string_nonissues = ", ".join(nonissues[:last_phrase]) + f" and {nonissues[last_phrase]}"
            st.write("They are otherwise within acceptable ranges for  " + string_nonissues + ".")
    elif len(nonissues) > 0:
        if len(nonissues) == 1:
            st.write("There is no immediate concern for this customer as they are within acceptable ranges for " + nonissues[0] + ".")
        else:
            last_phrase = len(nonissues) - 1
            string_nonissues = ", ".join(nonissues[:last_phrase]) + f" and {nonissues[last_phrase]}"
            st.write("There is no immediate concern for this customer as they are within acceptable ranges for " + string_nonissues + ".")
# -- end customer_blurb() function

st.title("Customer Churn Analysis")

filtered_index = None
feature_vals = None
flat_features = None
customer_churn_label = None
session_dur_avg = None
pagesper_session = None
cart_abndnmnt_rate = None
ttl_purchases = None
email_opens = None
ltv = 0
credit = None
issues = []
nonissues = []

customer_input = st.text_input("Enter Customer ID (ie. CX20201) to check risk:")

if customer_input:
  filtered_df = churn_dataset[churn_dataset["Customer_ID"] == customer_input]
  filtered_index = filtered_df.index

  st.write("Matching Results:")
  st.dataframe(filtered_df[['Customer_ID', 'First Name', 'Age', 'Gender']])

  if not filtered_df.empty:
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
    customer_churn_label = churn_df.iloc[filtered_index]['Churn_Risk'].iloc[0]

    session_dur_avg = flat_features[0]
    pagesper_session = flat_features[1]
    cart_abndnmnt_rate = flat_features[2]
    ttl_purchases = flat_features[3]
    email_opens = flat_features[4]
    ltv = flat_features[5]
    credit = flat_features[6]

### Blurb about this customer.
    customer_blurb()

  else:
    st.warning("No matching customer found.")

