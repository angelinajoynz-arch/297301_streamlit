import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pickle
import shap

st.set_page_config(layout="wide")

st.title("Churn Drivers")

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

@st.cache_data
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

features_to_plot = [
    "Lifetime_Value","Cart_Abandonment_Rate", "Email_Open_Rate", 
    "Session_Duration_Avg", "Total_Purchases", "Pages_Per_Session",
    "Credit_Balance"
]

row1_features = features_to_plot[0:3]
row2_features = features_to_plot[3:6]
row3_features = features_to_plot[6:9]

# --- ROW 1 ---
cols_row1 = st.columns(3)
for i, feature in enumerate(row1_features):
    with cols_row1[i]:
        fig, ax = plt.subplots(figsize=(4, 3))  # Smaller size so they fit nicely
        shap.plots.scatter(shap_explains[:, feature, 1], ax=ax, show=False)
        
        if feature == "Cart_Abandonment_Rate":
            ax.axvline(x=70, color='crimson', linestyle='--', linewidth=1, label='70')
        elif feature == "Email_Open_Rate":
            ax.axvline(x=10, color='crimson', linestyle='--', linewidth=1, label='10')
        elif feature == "Lifetime_Value":
            ax.axvline(x=1950, color='crimson', linestyle='--', linewidth=1, label='1950')

        ax.axhline(y=0.0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_title(feature.replace("_", " "), fontsize=16)
        ax.set_ylabel('Risk Factor'); ax.set_xlabel('Feature Value')
        ax.tick_params(axis='both', labelsize=8)
        ax.legend()
        
        st.pyplot(fig)
        plt.close(fig)

# --- ROW 2 ---
cols_row2 = st.columns(3)
for i, feature in enumerate(row2_features):
    with cols_row2[i]:
        fig, ax = plt.subplots(figsize=(4, 3))
        shap.plots.scatter(shap_explains[:, feature, 1], ax=ax, show=False)

        if feature == "Session_Duration_Avg":
            ax.axvline(x=20, color='crimson', linestyle='--', linewidth=1, label='20')
        elif feature == "Total_Purchases":
            ax.axvline(x=6, color='crimson', linestyle='--', linewidth=1, label='6')
        elif feature == "Pages_Per_Session":
            ax.axvline(x=6, color='crimson', linestyle='--', linewidth=1, label='6')
        
        ax.axhline(y=0.0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_title(feature.replace("_", " "), fontsize=16)
        ax.set_ylabel('Risk Factor'); ax.set_xlabel('Feature Value')
        ax.tick_params(axis='both', labelsize=8)
        ax.legend()
        
        st.pyplot(fig)
        plt.close(fig)


# --- ROW 3 ---
cols_row3 = st.columns(3)
for i, feature in enumerate(row3_features):
    with cols_row3[i]:
        fig, ax = plt.subplots(figsize=(4, 3))
        shap.plots.scatter(shap_explains[:, feature, 1], ax=ax, show=False)

        if feature == "Session_Duration_Avg":
            ax.axvline(x=20, color='crimson', linestyle='--', linewidth=1, label='20')
        elif feature == "Total_Purchases":
            ax.axvline(x=6, color='crimson', linestyle='--', linewidth=1, label='6')
        elif feature == "Pages_Per_Session":
            ax.axvline(x=6, color='crimson', linestyle='--', linewidth=1, label='6')
        
        ax.axhline(y=0.0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_title(feature.replace("_", " "), fontsize=16)
        ax.set_ylabel('Risk Factor'); ax.set_xlabel('Feature Value')
        ax.tick_params(axis='both', labelsize=8)
        ax.legend()
        
        st.pyplot(fig)
        plt.close(fig)
