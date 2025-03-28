import streamlit as st
import pandas as pd
from utils.data_loader import load_data

def run_overview():
    st.header("📊 Data Overview")
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("No data available")
        return
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", len(df))
    with col2:
        st.metric("Average Monthly Premium", f"Rs.{df['Monthly Premium Auto'].mean():.2f}")
    with col3:
        st.metric("Total Claims", f"Rs.{df['Total Claim Amount'].sum():.2f}")
    
    # Sample Data
    st.subheader("Sample Data")
    st.dataframe(df.head())
    
    # Data Summary
    st.subheader("Data Summary")
    st.write(df.describe())

# This allows the page to be run when imported
if __name__ == "__main__":
    run_overview()