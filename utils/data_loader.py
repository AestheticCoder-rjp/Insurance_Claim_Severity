import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    """
    Load and preprocess the insurance dataset
    
    Returns:
    pandas.DataFrame: Loaded and preprocessed dataframe
    """
    try:
        # Replace with your actual data loading method
        df = pd.read_csv("AutoInsuranceClaims2024.csv")
        
        # Basic preprocessing (add as needed)
        # Example: handle missing values, convert data types
        df = df.dropna()  # Remove rows with missing values
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()