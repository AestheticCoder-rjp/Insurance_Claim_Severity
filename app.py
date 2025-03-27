import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objs as go

# Page configuration
st.set_page_config(page_title='Insurance Data Analysis', layout='wide')

# Load data
@st.cache_data
def load_data():
    # Replace with your actual data loading method
    data = pd.read_csv('insurance_data.csv')
    return data

# Data Preprocessing
def preprocess_data(df):
    # Encode categorical variables
    categorical_columns = ['Location', 'Marital Status', 'Policy Type', 'Policy', 
                           'Renew Offer Type', 'Sales Channel', 'Vehicle Class', 'Vehicle Size']
    
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[f'{col}_Encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    return df, label_encoders

# EDA Section
def eda_section(df):
    st.header('🔍 Exploratory Data Analysis')
    
    # Summary Statistics
    st.subheader('Summary Statistics')
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(df.describe())
    
    with col2:
        # Categorical Distribution
        categorical_cols = ['Location', 'Marital Status', 'Policy Type', 'Sales Channel', 'Vehicle Class']
        selected_cat = st.selectbox('Select Categorical Column', categorical_cols)
        
        plt.figure(figsize=(10, 6))
        df[selected_cat].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {selected_cat}')
        st.pyplot(plt)
    
    # Correlation Heatmap
    st.subheader('Correlation Heatmap')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    st.pyplot(plt)
    
    # Interactive Scatter Plot
    st.subheader('Interactive Scatter Plot')
    x_col = st.selectbox('X-axis', numeric_cols)
    y_col = st.selectbox('Y-axis', numeric_cols, index=1)
    
    fig = px.scatter(df, x=x_col, y=y_col, color='Policy Type', 
                     hover_data=['Location', 'Vehicle Class'])
    st.plotly_chart(fig)

# Modeling Section
def modeling_section(df):
    st.header('🤖 Predictive Modeling')
    
    # Preprocess data
    df_processed, label_encoders = preprocess_data(df)
    
    # Select features and target
    feature_columns = ['Location_Encoded', 'Marital Status_Encoded', 'Monthly Premium Auto', 
                       'Months Since Last Claim', 'Number of Policies', 
                       'Policy Type_Encoded', 'Vehicle Class_Encoded']
    
    X = df_processed[feature_columns]
    y = df_processed['Policy Type_Encoded']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = rf_classifier.predict(X_test_scaled)
    
    # Model Performance
    st.subheader('Model Performance')
    col1, col2 = st.columns(2)
    
    with col1:
        st.text('Classification Report')
        st.text(classification_report(y_test, y_pred))
    
    with col2:
        # Feature Importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf_classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance')
        st.pyplot(plt)

# Insights Section
def insights_section(df):
    st.header('💡 Key Insights')
    
    # Average Monthly Premium by Location and Policy Type
    st.subheader('Average Monthly Premium')
    premium_pivot = df.pivot_table(values='Monthly Premium Auto', 
                                   index='Location', 
                                   columns='Policy Type', 
                                   aggfunc='mean')
    st.dataframe(premium_pivot)
    
    # Claim Analysis
    st.subheader('Claim Analysis')
    claim_analysis = df.groupby('Policy Type')['Total Claim Amount'].agg(['mean', 'max', 'count'])
    st.dataframe(claim_analysis)
    
    # Policy Distribution
    st.subheader('Policy Distribution')
    policy_dist = df['Policy Type'].value_counts()
    fig = px.pie(values=policy_dist.values, names=policy_dist.index, 
                 title='Policy Type Distribution')
    st.plotly_chart(fig)

# Main App
def main():
    st.title('🚗 Insurance Data Analysis Dashboard')
    
    # Load data
    df = load_data()
    
    # Sidebar Navigation
    page = st.sidebar.selectbox('Navigate', 
                                ['EDA', 'Modeling', 'Insights'])
    
    if page == 'EDA':
        eda_section(df)
    elif page == 'Modeling':
        modeling_section(df)
    else:
        insights_section(df)

if __name__ == '__main__':
    main()