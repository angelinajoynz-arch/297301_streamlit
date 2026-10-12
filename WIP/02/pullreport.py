import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pickle
import shap

st.button("Custom Global Style")

st.set_page_config(layout="wide")

churn_dataset = pd.read_csv('datasets/customer_churn.csv')
export_list = []

if 'export_list' not in st.session_state:
    st.session_state.export_list = []

if 'is_car_selected' not in st.session_state:
    st.session_state.is_car_selected = False

if 'is_total_purchases_selected' not in st.session_state:
    st.session_state.is_total_purchases_selected = False

def toggle_selected_car():
    st.session_state.is_car_selected = not st.session_state.is_car_selected

def toggle_selected_total_purchases():
    st.session_state.is_total_purchases_selected = not st.session_state.is_total_purchases_selected

if st.session_state.is_car_selected:
    st.button('Cart Abandonment Rate', icon="✔", on_click=toggle_selected_car)
    st.session_state.export_list.append('Cart_Abandonment_Rate')
else:
    st.button('Cart Abandonment Rate', on_click=toggle_selected_car)
    if 'Cart_Abandonment_Rate' in st.session_state.export_list:
        st.session_state.export_list.remove('Cart_Abandonment_Rate')

if st.session_state.is_total_purchases_selected:
    st.button('Total Purchases', icon="✔", on_click=toggle_selected_total_purchases)
    st.session_state.export_list.append('Total_Purchases')
else:
    st.button('Total Purchases', on_click=toggle_selected_total_purchases)
    if 'Total_Purchases' in st.session_state.export_list:
        st.session_state.export_list.remove('Total_Purchases')


st.write("Current List:", st.session_state.export_list)