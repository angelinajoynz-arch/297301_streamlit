import streamlit as st
import pandas as pd
import pickle

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #fefefc;
    }

    .stApp {
        background-color: #F9F9F9;
    }
    </style>
    """,
    unsafe_allow_html=True,)

st.sidebar.image("views/masksolutionslogo.png", use_container_width=True)

#home_page = st.Page("views/home.py", title="Home", icon="🏠", default=True)
home_page = st.Page('views/home.py', title='Home', icon='🏠', default=True)
dashboard_page = st.Page('views/dashboard.py', title='Dashboard', icon='📊')
customer_insights = st.Page('views/waterfalls.py', title='Customer Insights', icon='👤')
model_insights = st.Page('views/model_insights.py', title='Churn Drivers', icon='📈')
pull_report = st.Page('views/pullreport.py', title='Pull Report', icon='📄')

pg = st.navigation([home_page, dashboard_page, model_insights, customer_insights, pull_report])

pg.run()

### Function for applying risk labels
def apply_risk_label(row):
    if row['churn_probability'] < 0.3:
        return 'Low'
    elif row['churn_probability'] > 0.3 and row['churn_probability'] <= 0.4:
        return 'Medium'
    else:
        return 'High'

# Import data
churn_dataset = pd.read_csv('datasets/customer_churn.csv')

# Import model
with open('Model/final_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Preprocess Data
feature_list2 = ['Session_Duration_Avg', 'Pages_Per_Session', 'Cart_Abandonment_Rate', 'Total_Purchases', 'Email_Open_Rate', 'Lifetime_Value',
       'Credit_Balance', 'Churned']
churn_df = churn_dataset[feature_list2].copy()

y = churn_df['Churned']
X = churn_df.drop(columns=['Churned']).copy()

# Predict
y_proba = model.predict_proba(X)
churn_df['churn_probability'] = y_proba[:, 1]

# Apply Risk Labels
churn_df['Churn_Risk'] = churn_df.apply(apply_risk_label, axis=1)

churn_df.to_csv('churn_analysis_results.csv', index=False)



