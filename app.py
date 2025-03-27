import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Page Configuration
st.set_page_config(page_title="Isurance Analysis Dashboard", layout="wide")

# Data Loading and Preprocessing
@st.cache_data
def load_data():
    # Load your data here
    df = pd.read_csv("AutoInsuranceClaims2024.csv")  # Replace with your data loading method
    return df

df = load_data()

# Main Header
st.title("🚗 Insurance Data Analysis Dashboard")

# Sidebar Navigation
page = st.sidebar.selectbox("Select Analysis Page", 
    ["Overview", "EDA", "Customer Analysis", "Policy Analysis", "Modeling", "Insights"])

if page == "Overview":
    st.header("📊 Data Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", len(df))
    with col2:
        st.metric("Average Premium", f"${df['Monthly Premium Auto'].mean():.2f}")
    with col3:
        st.metric("Total Claims", f"${df['Total Claim Amount'].sum():.2f}")
    
    st.subheader("Sample Data")
    st.dataframe(df.head())
    
    st.subheader("Data Summary")
    st.write(df.describe())

elif page == "EDA":
    st.header("📈 Exploratory Data Analysis")
    
    # Distribution Plots
    st.subheader("Distribution Analysis")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    col = st.selectbox("Select Variable for Distribution", numeric_cols)
    
    fig = px.histogram(df, x=col, marginal="box")
    st.plotly_chart(fig)
    
    # Correlation Analysis
    st.subheader("Correlation Analysis")
    corr = df.corr()
    fig = px.imshow(corr, color_continuous_scale='RdBu')
    st.plotly_chart(fig)

elif page == "Customer Analysis":
    st.header("👥 Customer Segmentation")
    
    # Marital Status Distribution
    st.subheader("Customer Demographics")
    fig = px.pie(df, names='Marital Status', title='Distribution by Marital Status')
    st.plotly_chart(fig)
    
    # Location Analysis
    st.subheader("Geographic Distribution")
    fig = px.bar(df['Location'].value_counts().reset_index(), 
                 x='index', y='Location', title='Customer Location Distribution')
    st.plotly_chart(fig)

elif page == "Policy Analysis":
    st.header("📋 Policy Analysis")
    
    # Policy Type Distribution
    st.subheader("Policy Distribution")
    fig = px.pie(df, names='Policy Type', title='Distribution by Policy Type')
    st.plotly_chart(fig)
    
    # Premium vs Claims Analysis
    st.subheader("Premium vs Claims")
    fig = px.scatter(df, x='Monthly Premium Auto', y='Total Claim Amount',
                    color='Policy Type', title='Premium vs Claims by Policy Type')
    st.plotly_chart(fig)

elif page == "Modeling":
    st.header("🤖 Predictive Modeling")
    
    # Feature Selection
    features = ['Monthly Premium Auto', 'Months Since Last Claim', 
                'Number of Policies', 'Vehicle Class Index']
    target = 'Total Claim Amount'
    
    # Model Training
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Feature Importance
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    })
    fig = px.bar(importance_df, x='Feature', y='Importance')
    st.plotly_chart(fig)

elif page == "Insights":
    st.header("💡 Key Insights")
    
    st.subheader("Business Recommendations")
    st.write("""
    **Key Findings:**
    * Premium pricing shows strong correlation with vehicle class
    * Customer retention is highest among married individuals
    * Urban locations show higher claim frequencies
    
    **Recommendations:**
    * Implement risk-based pricing strategy
    * Develop targeted marketing for high-value segments
    * Enhance customer service in high-claim areas
    """)

# Footer
st.markdown("---")
st.markdown("*Insurance Analysis Dashboard - Created with Streamlit*")
