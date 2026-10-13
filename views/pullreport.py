import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pickle

churn_df = pd.read_csv('churn_analysis_results.csv')
churn_dataset = pd.read_csv('datasets/customer_churn.csv')
customer_deets = churn_dataset[['Customer_ID', 'First Name', 'Last Name']]
customer_churn_risk = churn_df[['Churn_Risk']]
starter_frame = pd.concat([customer_deets, customer_churn_risk], axis=1)

selected_tiers = []
selected_filters = []
d_val = 0.0
min_lifetime = d_val
max_lifetime = d_val
min_car = d_val; max_car = d_val;
min_email = d_val; max_email = d_val; min_session = d_val; max_session = d_val;
min_purchases = d_val; max_purchases = d_val;

@st.cache_data
def convert_df(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8")

# Min/Max values in dataset
absolute_vals = {'Lifetime_Value': [(churn_df['Lifetime_Value'].min(), churn_df['Lifetime_Value'].max())],
    'Cart_Abandonment_Rate': [(churn_df['Cart_Abandonment_Rate'].min(), churn_df['Cart_Abandonment_Rate'].max())],
    'Email_Open_Rate': [(churn_df['Email_Open_Rate'].min(), churn_df['Email_Open_Rate'].max())],
    'Session_Duration_Avg': [(churn_df['Session_Duration_Avg'].min(), churn_df['Session_Duration_Avg'].max())],
    'Total_Purchases': [(churn_df['Total_Purchases'].min(), churn_df['Total_Purchases'].max())],
    'Pages_Per_Session': [(churn_df['Pages_Per_Session'].min(), churn_df['Pages_Per_Session'].max())],
    'Credit_Balance': [(churn_df['Credit_Balance'].min(), churn_df['Credit_Balance'].max())]}

min_lifetime = absolute_vals['Lifetime_Value'][0][0]
max_lifetime = absolute_vals['Lifetime_Value'][0][1]
min_car = absolute_vals['Cart_Abandonment_Rate'][0][0]
max_car = absolute_vals['Cart_Abandonment_Rate'][0][1]
min_email = absolute_vals['Email_Open_Rate'][0][0]
max_email = absolute_vals['Email_Open_Rate'][0][1]
min_session = absolute_vals['Session_Duration_Avg'][0][0]
max_session = absolute_vals['Session_Duration_Avg'][0][1]
min_purchases = absolute_vals['Total_Purchases'][0][0]
max_purchases = absolute_vals['Total_Purchases'][0][1]

available_risk_tiers = ['Low', 'Medium', 'High']
available_filters = ['Cart Abandonment Rate', 'Lifetime Value', 'Total Purchases', 'Email Open Rate', 
                     'Session Duration Avg', 'Pages Per Session', 'Credit Balance']

st.markdown("<center><h1 style='font-family: Georgia, serif; font-weight: normal; color: #656280;'>Pull Report</h1></center>", unsafe_allow_html=True)
st.write('You can filter the results by selecting risk levels and features below. There is a minimum of one risk level and one feature required for export.')
st.write('##### Select customer risk level(s):')
cols1 = st.columns(3)
for index, tier in enumerate(available_risk_tiers):
    with cols1[index % 3]:
        if st.checkbox(tier, key=f"cb_{tier}"):
                selected_tiers.append(tier.replace(' ', '_'))

st.write('##### Select feature(s):')
cols2 = st.columns(3)
for index, filter in enumerate(available_filters):
    with cols2[index % 3]:
        if st.checkbox(filter, key=f"cb_{filter}"):
                selected_filters.append(filter.replace(' ', '_'))

if st.toggle("Show Advanced Filters"):
    if 'Lifetime_Value' in selected_filters:
        st.write('###### Lifetime Value:')
        min_lifetime = st.text_input(label="Minimum Lifetime Value:", value=absolute_vals['Lifetime_Value'][0][0])
        max_lifetime = st.text_input(label="Maximum Lifetime Value:", value=absolute_vals['Lifetime_Value'][0][1])

    if 'Cart_Abandonment_Rate' in selected_filters:
        st.write('###### Cart Abandonment Rate:')
        min_car = st.text_input(label="Minimum Cart Abandonment Rate:", value=absolute_vals['Cart_Abandonment_Rate'][0][0])
        max_car = st.text_input(label="Maximum Cart Abandonment Rate:", value=absolute_vals['Cart_Abandonment_Rate'][0][1])

    if 'Email_Open_Rate' in selected_filters:
        st.write('###### Email Open Rate:')
        min_email = st.text_input(label="Minimum Email Open Rate:", value=absolute_vals['Email_Open_Rate'][0][0])
        max_email = st.text_input(label="Maximum Email Open Rate:", value=absolute_vals['Email_Open_Rate'][0][1])

    if 'Session_Duration_Avg' in selected_filters:
        st.write('###### Session Duration Avg:')
        min_session = st.text_input(label="Minimum Session Duration Avg:", value=absolute_vals['Session_Duration_Avg'][0][0])
        max_session = st.text_input(label="Maximum Session Duration Avg:", value=absolute_vals['Session_Duration_Avg'][0][1])

    if 'Total_Purchases' in selected_filters:
        st.write('###### Total Purchases:')
        min_purchases = st.text_input(label="Minimum Total Purchases:", value=absolute_vals['Total_Purchases'][0][0])
        max_purchases = st.text_input(label="Maximum Total Purchases:", value=absolute_vals['Total_Purchases'][0][1])

    if 'Pages_Per_Session' in selected_filters:
        st.write('###### Pages Per Session:')
        min_pages = st.text_input(label="Minimum Pages Per Session:", value=absolute_vals['Pages_Per_Session'][0][0])
        max_pages = st.text_input(label="Maximum Pages Per Session:", value=absolute_vals['Pages_Per_Session'][0][1])

    if 'Credit_Balance' in selected_filters:
        st.write('###### Credit Balance:')
        min_credit = st.text_input(label="Minimum Credit Balance:", value=absolute_vals['Credit_Balance'][0][0])
        max_credit = st.text_input(label="Maximum Credit Balance:", value=absolute_vals['Credit_Balance'][0][1])

building_frame = pd.DataFrame()

# Build df for export
if 'Lifetime_Value' in selected_filters:
    if 'Lifetime_Value' in building_frame.columns:
        building_frame.drop(columns=['Lifetime_Value'], inplace=True)
        lifetime_value = churn_df['Lifetime_Value'].loc[(churn_df['Lifetime_Value'] >= float(min_lifetime)) & (churn_df['Lifetime_Value'] <= float(max_lifetime))]

    else:
        lifetime_value = churn_df['Lifetime_Value'].loc[(churn_df['Lifetime_Value'] >= float(min_lifetime)) & (churn_df['Lifetime_Value'] <= float(max_lifetime))]

    building_frame = pd.concat([building_frame, lifetime_value], axis=1)

if 'Cart_Abandonment_Rate' in selected_filters:
    if 'Cart_Abandonment_Rate' in building_frame.columns:
        building_frame.drop(columns=['Cart_Abandonment_Rate'], inplace=True)
        cart_abandonment_rate = churn_df['Cart_Abandonment_Rate'].loc[(churn_df['Cart_Abandonment_Rate'] >= float(min_car)) & (churn_df['Cart_Abandonment_Rate'] <= float(max_car))]

    else:
        cart_abandonment_rate = churn_df['Cart_Abandonment_Rate'].loc[(churn_df['Cart_Abandonment_Rate'] >= float(min_car)) & (churn_df['Cart_Abandonment_Rate'] <= float(max_car))]
    building_frame = pd.concat([building_frame, cart_abandonment_rate], axis=1)

if 'Email_Open_Rate' in selected_filters:
    if 'Email_Open_Rate' in building_frame.columns:
        building_frame.drop(columns=['Email_Open_Rate'], inplace=True)
        email_opens = churn_df['Email_Open_Rate'].loc[(churn_df['Email_Open_Rate'] >= float(min_email)) & (churn_df['Email_Open_Rate'] <= float(max_email))]
    else:
        email_opens = churn_df['Email_Open_Rate'].loc[(churn_df['Email_Open_Rate'] >= float(min_email)) & (churn_df['Email_Open_Rate'] <= float(max_email))]
    building_frame = pd.concat([building_frame, email_opens], axis=1)

if 'Session_Duration_Avg' in selected_filters:
    if 'Session_Duration_Avg' in building_frame.columns:
        building_frame.drop(columns=['Session_Duration_Avg'], inplace=True)
        session_duration = churn_df['Session_Duration_Avg'].loc[(churn_df['Session_Duration_Avg'] >= float(min_session)) & (churn_df['Session_Duration_Avg'] <= float(max_session))]
    else:
        session_duration = churn_df['Session_Duration_Avg'].loc[(churn_df['Session_Duration_Avg'] >= float(min_session)) & (churn_df['Session_Duration_Avg'] <= float(max_session))]
    building_frame = pd.concat([building_frame, session_duration], axis=1)

if 'Total_Purchases' in selected_filters:
    if 'Total_Purchases' in building_frame.columns:
        building_frame.drop(columns=['Total_Purchases'], inplace=True)
        total_purchases = churn_df['Total_Purchases'].loc[(churn_df['Total_Purchases'] >= float(min_purchases)) & (churn_df['Total_Purchases'] <= float(max_purchases))]
    else:
        total_purchases = churn_df['Total_Purchases'].loc[(churn_df['Total_Purchases'] >= float(min_purchases)) & (churn_df['Total_Purchases'] <= float(max_purchases))]
    building_frame = pd.concat([building_frame, total_purchases], axis=1)

if 'Pages_Per_Session' in selected_filters:
    if 'Pages_Per_Session' in building_frame.columns:
        building_frame.drop(columns=['Pages_Per_Session'], inplace=True)
        pages_per_session = churn_df['Pages_Per_Session'].loc[(churn_df['Pages_Per_Session'] >= float(min_pages)) & (churn_df['Pages_Per_Session'] <= float(max_pages))]
    else:
        pages_per_session = churn_df['Pages_Per_Session'].loc[(churn_df['Pages_Per_Session'] >= float(min_pages)) & (churn_df['Pages_Per_Session'] <= float(max_pages))]
    building_frame = pd.concat([building_frame, pages_per_session], axis=1)

if 'Credit_Balance' in selected_filters:
    if 'Credit_Balance' in building_frame.columns:
        building_frame.drop(columns=['Credit_Balance'], inplace=True)
        credit_balance = churn_df['Credit_Balance'].loc[(churn_df['Credit_Balance'] >= float(min_credit)) & (churn_df['Credit_Balance'] <= float(max_credit))]
    else:
        credit_balance = churn_df['Credit_Balance'].loc[(churn_df['Credit_Balance'] >= float(min_credit)) & (churn_df['Credit_Balance'] <= float(max_credit))]
    building_frame = pd.concat([building_frame, credit_balance], axis=1)

if selected_filters is not None:
    filtered_dfa = pd.concat([starter_frame, building_frame], axis=1)

filtered_df = building_frame.copy()
results_df = pd.concat([starter_frame, filtered_df], axis=1)
results_df.dropna(inplace=True)

if len(selected_tiers) > 0 and len(selected_filters) > 0:
    csv_data = convert_df(results_df)
    st.download_button(
        label="📥 Export to CSV",
        data=csv_data,
        file_name="exported_data.csv",
        mime="text/csv",
    )




