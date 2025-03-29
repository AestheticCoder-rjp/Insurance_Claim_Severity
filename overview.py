import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.data_loader import load_data

def run_overview():
    st.header("📊 Data Overview")
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("No data available")
        return
    
    # Key Metrics
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        st.metric("Average Premium", f"Rs.{df['Monthly Premium Auto'].mean():.2f}")
    with col3:
        st.metric("Total Claims", f"Rs.{df['Total Claim Amount'].sum():,.2f}")
    with col4:
        st.metric("Avg Customer Value", f"Rs.{df['Customer Lifetime Value'].mean():,.2f}")
    
    # Data Information Tabs
    tab1, tab2, tab3 = st.tabs(["Sample Data", "Data Quality", "Column Information"])
    
    with tab1:
        # Sample Data
        st.subheader("Sample Records")
        
        # Add a slider to control how many rows to show
        num_rows = st.slider("Number of rows to display", 5, 50, 10)
        st.dataframe(df.head(num_rows))
        
        # Add download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Full Dataset as CSV",
            data=csv,
            file_name='insurance_data.csv',
            mime='text/csv',
        )
    
    with tab2:
        # Null Values Analysis
        st.subheader("Null Values Analysis")
        
        # Calculate null values
        null_counts = df.isnull().sum()
        null_percent = (null_counts / len(df)) * 100
        
        # Create a DataFrame to display null values information
        null_df = pd.DataFrame({
            'Column': null_counts.index,
            'Missing Values': null_counts.values,
            'Missing Percentage': null_percent.values
        })
        
        # Sort by missing values (descending)
        null_df = null_df.sort_values('Missing Values', ascending=False)
        
        # Display statistics
        total_missing = null_counts.sum()
        total_cells = np.prod(df.shape)
        overall_missing_percent = (total_missing / total_cells) * 100
        
        st.metric("Overall Data Completeness", f"{100 - overall_missing_percent:.2f}%")
        
        # Create a bar chart for null values
        if null_counts.sum() > 0:
            fig = px.bar(
                null_df, 
                x='Column', 
                y='Missing Percentage',
                text='Missing Values',
                title='Missing Values by Column',
                color='Missing Percentage',
                color_continuous_scale=px.colors.sequential.Blues_r
            )
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(xaxis_title='Column', yaxis_title='Missing Percentage (%)')
            st.plotly_chart(fig)
            
            # Display the null values table
            st.dataframe(null_df)
        else:
            st.success("✅ No missing values in the dataset!")
        
        # Data types information
        st.subheader("Data Types")
        
        # Create a DataFrame with data types - FIXED: Convert data types to strings
        dtype_dict = {col: str(dtype) for col, dtype in zip(df.dtypes.index, df.dtypes.values)}
        dtype_df = pd.DataFrame({
            'Column': list(dtype_dict.keys()),
            'Data Type': list(dtype_dict.values())
        })
        
        # Count of each data type - FIXED: Convert to strings first
        dtype_counts = pd.Series(dtype_dict.values()).value_counts().reset_index()
        dtype_counts.columns = ['Data Type', 'Count']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(dtype_df)
        
        with col2:
            # Pie chart of data types
            fig = px.pie(
                dtype_counts, 
                values='Count', 
                names='Data Type',
                title='Distribution of Data Types',
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig)
    
    with tab3:
        # Column Information
        st.subheader("Column Information")
        
        # Create a DataFrame with column information
        column_info = []
        for col in df.columns:
            col_type = str(df[col].dtype)  # Convert dtype to string
            unique_values = df[col].nunique()
            
            # Determine if categorical or numerical
            if col_type in ['object', 'category'] or 'string' in col_type.lower():
                col_category = "Categorical"
                # Get examples of categorical values (up to 5)
                examples = ", ".join(df[col].dropna().unique()[:5].astype(str))
                if len(df[col].dropna().unique()) > 5:
                    examples += "..."
            else:
                col_category = "Numerical"
                # Convert to Python native types to avoid JSON serialization issues
                min_val = float(df[col].min()) if not pd.isna(df[col].min()) else "N/A"
                max_val = float(df[col].max()) if not pd.isna(df[col].max()) else "N/A"
                examples = f"Min: {min_val}, Max: {max_val}"
            
            column_info.append({
                "Column": col,
                "Type": col_type,
                "Category": col_category,
                "Unique Values": int(unique_values),  # Convert to native Python int
                "Examples": examples
            })
        
        column_df = pd.DataFrame(column_info)
        st.dataframe(column_df)
        
        # Summary statistics
        st.subheader("Statistical Summary")
        st.write(df.describe())

# This allows the page to be run when imported
if __name__ == "__main__":
    run_overview()