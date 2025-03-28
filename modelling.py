import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    r2_score, 
    mean_absolute_error, 
    mean_squared_error, 
    mean_absolute_percentage_error
)

# Optional XGBoost (if installed)
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def run_advanced_modeling():
    st.header("🤖 Advanced Predictive Modeling")
    
    # Load data
    @st.cache_data
    def load_data():
        # Replace with your actual data loading method
        df = pd.read_csv("AutoInsuranceClaims2024.csv")
        return df
    
    df = load_data()
    
    # Model Configuration Section
    st.sidebar.header("🛠 Modeling Configuration")
    
    # Feature Selection
    st.sidebar.subheader("Select Features")
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    selected_numerical = st.sidebar.multiselect(
        "Numerical Features", 
        numerical_features, 
        #default=numerical_features[1]
    )
    
    selected_categorical = st.sidebar.multiselect(
        "Categorical Features", 
        categorical_features, 
        #default=categorical_features[5:7]
    )
    
    # Target Variable Selection
    target = st.sidebar.selectbox(
        "Select Target Variable", 
        df.select_dtypes(include=['int64', 'float64']).columns
    )
    
    # Prepare Data
    X = df[selected_numerical + selected_categorical]
    y = df[target]
    
    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), selected_numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), selected_categorical)
        ])
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define Models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42)
    
    # Tabs for different visualizations
    tab1, tab2, tab3 = st.tabs([
        "Model Performance", 
        "Feature Importance", 
        "Prediction vs Actual"
    ])
    
    with tab1:
        st.subheader("Model Comparison")
        
        # Train and evaluate models
        results = {}
        model_predictions = {}
        
        for name, model in models.items():
            # Create pipeline
            pipeline = Pipeline([
                ("preprocessor", preprocessor), 
                ("model", model)
            ])
            
            # Fit model
            pipeline.fit(X_train, y_train)
            
            # Predictions
            y_pred = pipeline.predict(X_test)
            model_predictions[name] = y_pred
            
            # Compute metrics
            results[name] = {
                "R²": r2_score(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "MAPE": mean_absolute_percentage_error(y_test, y_pred)
            }
        
        # Convert results to DataFrame for visualization
        results_df = pd.DataFrame(results).T
        
        # Plotly Bar Chart for Metrics
        fig_metrics = go.Figure()
        metrics = ['R²', 'MAE', 'RMSE', 'MAPE']
        
        for metric in metrics:
            fig_metrics.add_trace(go.Bar(
                name=metric,
                x=results_df.index,
                y=results_df[metric],
                text=[f'{val:.4f}' for val in results_df[metric]],
                textposition='auto'
            ))
        
        fig_metrics.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Metric Value",
            barmode='group'
        )
        
        st.plotly_chart(fig_metrics)
        
        # Detailed Results Table
        st.dataframe(results_df)
    
    with tab2:
        st.subheader("Feature Importance")
        
        # Feature Importance for Random Forest
        rf_model = models["Random Forest"]
        pipeline_rf = Pipeline([
            ("preprocessor", preprocessor), 
            ("model", rf_model)
        ])
        pipeline_rf.fit(X_train, y_train)
        
        # Extract feature names after preprocessing
        feature_names = (
            preprocessor.named_transformers_['num'].get_feature_names_out(selected_numerical).tolist() +
            preprocessor.named_transformers_['cat'].get_feature_names_out(selected_categorical).tolist()
        )
        
        # Get feature importances
        importances = pipeline_rf.named_steps['model'].feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plotly Feature Importance
        fig_importance = go.Figure(
            data=[go.Bar(
                x=[feature_names[i] for i in indices[:10]],
                y=importances[indices[:10]],
                text=[f'{imp:.4f}' for imp in importances[indices[:10]]],
                textposition='auto'
            )]
        )
        fig_importance.update_layout(
            title="Top 10 Feature Importances",
            xaxis_title="Features",
            yaxis_title="Importance"
        )
        st.plotly_chart(fig_importance)
    
    with tab3:
        st.subheader("Prediction vs Actual Comparison")
        
        # Select a model for detailed visualization
        selected_model = st.selectbox(
            "Select Model for Visualization", 
            list(models.keys())
        )
        
        # Scatter plot of predictions vs actual values
        fig_pred_actual = px.scatter(
            x=y_test, 
            y=model_predictions[selected_model],
            labels={'x': 'Actual Values', 'y': 'Predicted Values'},
            title=f'Predictions vs Actual Values - {selected_model}'
        )
        
        # Add diagonal line for perfect predictions
        fig_pred_actual.add_trace(
            go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            )
        )
        
        st.plotly_chart(fig_pred_actual)

# This allows the page to be run when imported
if __name__ == "__main__":
    run_advanced_modeling()