import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="Insurance Analysis Dashboard", layout="wide")
st.title("Insurance Business Analytics Dashboard")

# Data Loading and Preprocessing
@st.cache_data
def load_data():
    # Load your data here
    df = pd.read_csv("AutoInsuranceClaims2024.csv")  # Replace with your data loading method
    return df

df = load_data()

# Sidebar
st.sidebar.title("Filter Options")
selected_location = st.sidebar.multiselect("Select Location", df['Location'].unique())
selected_vehicle = st.sidebar.multiselect("Select Vehicle Class", df['Vehicle_Class'].unique())

# Apply filters
if selected_location:
    df = df[df['Location'].isin(selected_location)]
if selected_vehicle:
    df = df[df['Vehicle_Class'].isin(selected_vehicle)]

# Main dashboard
st.header("Key Performance Indicators")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Average Premium", f"${df['Monthly_Premium_Auto'].mean():.2f}")
with col2:
    st.metric("Total Claims", f"${df['Total_Claim_Amount'].sum():.2f}")
with col3:
    st.metric("Average Claim Amount", f"${df['Total_Claim_Amount'].mean():.2f}")

# Claims Analysis
st.header("Claims Analysis")
col1, col2 = st.columns(2)

with col1:
    # Claims by Vehicle Class
    fig_vehicle = px.bar(
        df.groupby('Vehicle_Class')['Total_Claim_Amount'].mean().reset_index(),
        x='Vehicle_Class',
        y='Total_Claim_Amount',
        title='Average Claim Amount by Vehicle Class',
        color='Vehicle_Class'
    )
    st.plotly_chart(fig_vehicle)
    st.write("Analysis: Shows which vehicle types are associated with higher claim amounts, helping in risk assessment and premium pricing.")

with col2:
    # Claims by Location
    fig_location = px.pie(
        df.groupby('Location')['Total_Claim_Amount'].sum().reset_index(),
        values='Total_Claim_Amount',
        names='Location',
        title='Total Claims Distribution by Location'
    )
    st.plotly_chart(fig_location)
    st.write("Analysis: Reveals geographical distribution of claims, useful for targeted risk management strategies.")

# Sales Channel Performance
st.header("Sales Channel Analysis")
fig_sales = px.scatter(
    df,
    x='Monthly_Premium_Auto',
    y='Total_Claim_Amount',
    color='Sales_Channel',
    size='Monthly_Premium_Auto',
    title='Premium vs Claims by Sales Channel'
)
st.plotly_chart(fig_sales)
st.write("Analysis: Visualizes the relationship between premiums and claims across different sales channels, helping optimize distribution strategy.")

# Policy Type Analysis
st.header("Policy Type Performance")
col1, col2 = st.columns(2)

with col1:
    avg_premium_policy = df.groupby('Policy_Type')['Monthly_Premium_Auto'].mean().reset_index()
    fig_policy_premium = px.bar(
        avg_premium_policy,
        x='Policy_Type',
        y='Monthly_Premium_Auto',
        title='Average Premium by Policy Type',
        color='Policy_Type'
    )
    st.plotly_chart(fig_policy_premium)
    st.write("Analysis: Compares average premiums across different policy types to identify most profitable segments.")

# Time-based Analysis
months_analysis = df.groupby('Months_Since_Last_Claim')['Total_Claim_Amount'].mean().reset_index()
fig_time = px.line(
    months_analysis,
    x='Months_Since_Last_Claim',
    y='Total_Claim_Amount',
    title='Claim Amount Trend Over Time'
)
st.plotly_chart(fig_time)
st.write("Analysis: Shows claim amount patterns over time, useful for understanding claim frequency and severity trends.")

# Insights Section
st.header("Key Business Insights")
st.write("""
1. Premium vs Claims Relationship
   - Analyze correlation between premium pricing and claim amounts
   - Identify potentially underpriced or overpriced segments

2. Channel Performance
   - Compare effectiveness of different sales channels
   - Evaluate customer acquisition costs against premium revenue

3. Risk Assessment
   - Identify high-risk vehicle classes and locations
   - Guide underwriting decisions and premium adjustments
""")
