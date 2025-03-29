import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from utils.data_loader import load_data

def run_eda_analysis():
    st.header("Exploratory Data Analysis & Insights")
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("No data available")
        return
    
    # Tabs for different analysis sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Distribution Analysis", 
        "Correlation Analysis", 
        "Customer Segmentation", 
        "Policy & Claims Analysis",
        "Advanced Visualizations"
    ])
    
    with tab1:
        
        # Distribution Analysis
        st.subheader("Distribution Analysis")

        # Define specific columns to include
        selected_numeric_cols = ['Monthly Premium Auto', 'Income', 'Customer Lifetime Value', 'Months Since Last Claim', 'Total Claim Amount']

        # Ensure the columns exist in the dataset (to avoid errors)
        numeric_cols = [col for col in selected_numeric_cols if col in df.columns]

        col = st.selectbox("Select Variable for Distribution", numeric_cols)

        fig_dist = px.histogram(df, x=col, marginal="box", title=f"Distribution of {col}")
        st.plotly_chart(fig_dist)

        
        # Additional insights
        st.subheader("Statistical Summary")
        col_summary = df[col]
        summary_stats = {
            "Mean": col_summary.mean(),
            "Median": col_summary.median(),
            "Standard Deviation": col_summary.std(),
            "Minimum": col_summary.min(),
            "Maximum": col_summary.max()
        }
        st.write(summary_stats)
    
    with tab2:
        # Correlation Analysis
        st.subheader("Correlation Analysis")

        # Define specific numeric columns to include
        selected_numeric_cols = ['Monthly Premium Auto', 'Income', 'Customer Lifetime Value', 'Months Since Last Claim', 'Total Claim Amount']

        # Filter the dataset to only include the selected columns
        df_numeric = df[selected_numeric_cols]

        # Compute correlation
        corr = df_numeric.corr()

        # Create a heatmap using Plotly
        fig_corr = px.imshow(corr, 
                            color_continuous_scale='RdBu', 
                            title="Correlation Heatmap",
                            labels=dict(color="Correlation"),
                            x=df_numeric.columns,
                            y=df_numeric.columns)

        st.plotly_chart(fig_corr)

        # Detailed Correlation Insights
        st.subheader("Top Correlations")

        # Extract top correlations (excluding self-correlations)
        corr_pairs = [
            (col1, col2, abs(corr.loc[col1, col2]))
            for i, col1 in enumerate(corr.columns)
            for j, col2 in enumerate(corr.columns)
            if i < j  # Avoid self-correlations
        ]

        # Sort by absolute correlation value and get the top 5
        top_correlations = sorted(corr_pairs, key=lambda x: x[2], reverse=True)[:5]

        # Display top correlations
        st.write("### Strongest Correlations")
        for col1, col2, value in top_correlations:
            st.write(f" **{col1}** & **{col2}** → Correlation: `{value:.2f}`")

    
    with tab3:
        # Customer Segmentation
        st.subheader("Customer Demographics")
        
        # Marital Status Distribution
        if 'Marital Status' in df.columns:
            fig_marital = px.pie(df, names='Marital Status', 
                                  title='Distribution by Marital Status')
            st.plotly_chart(fig_marital)
        
        # Geographic Distribution
        st.subheader("Geographic Distribution")
        if 'Location' in df.columns:
            # Location Distribution
            location_counts = df['Location'].value_counts()
            
            # Bar chart for location distribution
            fig_location = px.bar(
                x=location_counts.index, 
                y=location_counts.values,
                labels={'x': 'Location', 'y': 'Count'},
                title='Location Distribution'
            )
            st.plotly_chart(fig_location)
            
            # Breakdown of locations
            st.subheader("Location Breakdown")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Suburban", location_counts.get('Suburban', 0))
            with col2:
                st.metric("Rural", location_counts.get('Rural', 0))
            with col3:
                st.metric("Urban", location_counts.get('Urban', 0))
    
    with tab4:
        # Policy and Claims Analysis
        st.subheader("Policy Type Distribution")
        # Policy Type Distribution
        if 'Policy Type' in df.columns:
            fig_policy = px.pie(df, names='Policy Type', 
                                 title='Distribution by Policy Type')
            st.plotly_chart(fig_policy)
        
        # Premium vs Claims Analysis
        st.subheader("Premium vs Claims Analysis")
        fig_premium = px.scatter(df, 
                                  x='Monthly Premium Auto', 
                                  y='Total Claim Amount',
                                  color='Policy Type', 
                                  title='Premium vs Claims by Policy Type')
        st.plotly_chart(fig_premium)
        
        # NEW: Adding Box Plot for Claim Amount by Vehicle Class
        st.subheader("Claim Amount Analysis by Categories")
        
        claim_analysis_type = st.radio(
            "Select Analysis Type:",
            ["Vehicle Class", "Coverage Type"]
        )
        
        if claim_analysis_type == "Vehicle Class":
            # Plot 6: Total Claim Amount by Vehicle Class (Box plot)
            fig6 = px.box(df, x='Vehicle Class', y='Total Claim Amount', title='Claim Amount by Vehicle Class')
            fig6.update_layout(title_font_size=18, xaxis_title='Vehicle Class', yaxis_title='Claim Amount')
            st.plotly_chart(fig6)
            st.write("This plot demonstrates differences in claim amounts across vehicle classes, indicating risk based on vehicle type.")
        else:
            # Plot 10: Coverage Type vs. Total Claim Amount
            fig10 = px.box(df, x='Coverage', y='Total Claim Amount', title='Claim Amount by Coverage Type')
            fig10.update_layout(title_font_size=18, xaxis_title='Coverage Type', yaxis_title='Claim Amount')
            st.plotly_chart(fig10)
            st.write("This plot evaluates how different coverage types correlate with claim costs, indicating risk based on coverage.")
    
    with tab5:
        # Advanced Visualizations Tab
        st.subheader("Hierarchical Visualizations")
        
        viz_type = st.selectbox(
            "Select Visualization Type:",
            ["Sunburst Plot", "Treemap"]
        )
        
        if viz_type == "Sunburst Plot":
            # 2. Sunburst plot: Hierarchical breakdown by State and Vehicle Class
            if 'State' in df.columns and 'Vehicle Class' in df.columns:
                fig2 = px.sunburst(df, path=['State', 'Vehicle Class'], values='Total Claim Amount', 
                                  title='Sunburst of Claim Amount by State and Vehicle Class')
                fig2.update_layout(title_font_size=18)
                st.plotly_chart(fig2)
                st.write("This sunburst plot shows the hierarchical breakdown of claim amounts first by state and then by vehicle class within each state.")
            else:
                st.warning("State or Vehicle Class columns not found in the dataset.")
        else:
            # 3. Treemap: Breakdown of Total Claim Amount by Policy Type and Coverage
            if 'Policy Type' in df.columns and 'Coverage' in df.columns:
                fig3 = px.treemap(df, path=['Policy Type', 'Coverage'], values='Total Claim Amount', 
                                 title='Treemap of Claim Amount by Policy Type and Coverage')
                fig3.update_layout(title_font_size=18)
                st.plotly_chart(fig3)
                st.write("This treemap visualizes the breakdown of total claim amounts by policy type and coverage, helping identify which combinations result in higher claims.")
            else:
                st.warning("Policy Type or Coverage columns not found in the dataset.")

# This allows the page to be run when imported
if __name__ == "__main__":
    run_eda_analysis()