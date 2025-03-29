import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp
import pickle
import os
from datetime import datetime

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score, 
    mean_absolute_error, 
    mean_squared_error
)

# XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def run_advanced_modeling():
    st.header("🚗 Auto Insurance Claim Amount Predictor")
    
    # Load data
    @st.cache_data
    def load_data():
        df = pd.read_csv("AutoInsuranceClaims2024.csv")
        return df

    df = load_data()
    
    # Identify columns for preprocessing
    numerical_features = ['Monthly Premium Auto', 'Income', 'Customer Lifetime Value', 'Months Since Last Claim']
    categorical_features = ['Location', 'Gender', 'Marital Status']
    target_column = 'Total Claim Amount'
    
    # Preprocessing function with one-hot encoding
    def preprocess_data(df):
        # Create a copy of the dataframe
        processed_df = df.copy()
        
        # One-hot encode categorical features
        for feature in categorical_features:
            # Create one-hot encoded columns
            one_hot = pd.get_dummies(processed_df[feature], prefix=feature)
            processed_df = pd.concat([processed_df, one_hot], axis=1)
            
            # Drop original categorical column
            processed_df.drop(columns=[feature], inplace=True)
        
        return processed_df
    
    # Define pickle file paths
    models_file = 'trained_models.pkl'
    feature_selection_file = 'feature_selection.pkl'
    model_predictions_file = 'model_predictions.pkl'
    results_file = 'model_results.pkl'
    
    # Add sidebar info and controls for model training
    st.sidebar.title("Model Controls")
    
    # Check if models exist
    models_exist = (os.path.exists(models_file) and 
                    os.path.exists(feature_selection_file) and 
                    os.path.exists(model_predictions_file) and 
                    os.path.exists(results_file))
    
    if models_exist:
        model_time = datetime.fromtimestamp(os.path.getmtime(models_file))
        st.sidebar.success(f"Using saved models (trained on: {model_time.strftime('%Y-%m-%d %H:%M:%S')})")
        force_retrain = st.sidebar.button("Retrain Models")
    else:
        st.sidebar.warning("No trained models found. Will train new models.")
        force_retrain = True
    
    # Function to perform feature selection
    def perform_feature_selection(df, target_column):
        # Identify features after one-hot encoding
        one_hot_features = [col for col in df.columns if col.startswith(('Location_', 'Gender_', 'Marital Status_'))]
        features_for_model = numerical_features + one_hot_features
        
        # Prepare data
        X = df[features_for_model]
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', 'passthrough', one_hot_features)
            ]
        )
        
        # Transform data
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Combine feature names
        feature_names = numerical_features + one_hot_features
        
        # Train Random Forest for feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train_transformed, y_train)
        
        # Get feature importance
        feature_importance_df = pd.DataFrame({
            "Feature": feature_names, 
            "Importance": rf.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        
        return {
            'preprocessor': preprocessor,
            'feature_names': feature_names,
            'feature_importance': feature_importance_df,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_transformed': X_train_transformed,
            'X_test_transformed': X_test_transformed,
            'original_columns': list(X.columns)
        }
    
    # Load or train models based on existence and force_retrain flag
    if not models_exist or force_retrain:
        # Show training progress
        with st.sidebar.status("Training models..."):
            # Preprocess the data
            processed_df = preprocess_data(df)
            
            # Perform Feature Selection
            feature_selection_results = perform_feature_selection(processed_df, target_column)
            st.sidebar.write("Feature selection complete")
            
            # Define Models
            models = {}
            
            # XGBoost Model
            if XGBOOST_AVAILABLE:
                xgb_best = XGBRegressor(
                    learning_rate=0.1, 
                    max_depth=5, 
                    n_estimators=100, 
                    subsample=0.8,
                    objective='reg:squarederror', 
                    random_state=42
                )
                models["XGBoost"] = xgb_best
            
            # Random Forest Model
            rf_best = RandomForestRegressor(
                max_depth=20, 
                min_samples_leaf=2, 
                min_samples_split=2, 
                n_estimators=200,
                random_state=42
            )
            models["Random Forest"] = rf_best
            
            # Results storage
            results = {}
            model_predictions = {}
            
            # Train and evaluate models
            for name, model in models.items():
                st.sidebar.write(f"Training {name} model...")
                # Fit model
                model.fit(feature_selection_results['X_train_transformed'], feature_selection_results['y_train'])
                
                # Predictions
                y_pred = model.predict(feature_selection_results['X_test_transformed'])
                model_predictions[name] = y_pred
                
                # Compute metrics
                results[name] = {
                    "R² Score": r2_score(feature_selection_results['y_test'], y_pred),
                    "Mean Absolute Error": mean_absolute_error(feature_selection_results['y_test'], y_pred),
                    "Root Mean Square Error": np.sqrt(mean_squared_error(feature_selection_results['y_test'], y_pred))
                }
            
            # Save models and results
            st.sidebar.write("Saving models to disk...")
            with open(models_file, 'wb') as f:
                pickle.dump(models, f)
            
            with open(feature_selection_file, 'wb') as f:
                pickle.dump(feature_selection_results, f)
            
            with open(model_predictions_file, 'wb') as f:
                pickle.dump(model_predictions, f)
            
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)
            
            st.sidebar.success("Models trained and saved successfully!")
    else:
        # Load models if they exist
        with st.sidebar.status("Loading saved models..."):
            with open(models_file, 'rb') as f:
                models = pickle.load(f)
            
            with open(feature_selection_file, 'rb') as f:
                feature_selection_results = pickle.load(f)
            
            with open(model_predictions_file, 'rb') as f:
                model_predictions = pickle.load(f)
            
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
            
            st.sidebar.success("Models loaded successfully!")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame({name: metrics for name, metrics in results.items()}).T
    
    # Tabs for Visualization
    tab1, tab2, tab3, tab4 = st.tabs([
        "Model Performance", 
        "Feature Importance", 
        "Prediction vs Actual",
        "Predict Claim Amount"
    ])
    
    with tab1:
        st.subheader("Model Performance Metrics")
        
        # Define a color palette
        metric_colors = {
            "R² Score": "#1E90FF",
            "Mean Absolute Error": "#2E8B57",
            "Root Mean Square Error": "#8A4FFF"
        }
        
        # Create subplot for all three metrics
        from plotly.subplots import make_subplots
        
        fig = make_subplots(rows=1, cols=3, subplot_titles=list(results_df.columns))
        
        for i, metric in enumerate(results_df.columns):
            fig.add_trace(
                go.Bar(
                    x=results_df.index,
                    y=results_df[metric],
                    text=[f'{val:.4f}' for val in results_df[metric]],
                    textposition='outside',
                    marker_color=metric_colors[metric],
                    textfont=dict(size=11, color='#333333')
                ),
                row=1, col=i+1
            )
        
        # Update layout for a professional look
        fig.update_layout(
            title={
                'text': " ",
                'font': dict(size=16, color='#333333'),
                'x': 0.5,
                'xanchor': 'center'
            },
            height=400,
            width=900,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12, color='#333333'),
            margin=dict(l=50, r=50, t=60, b=50)
        )
        
        # Customize axes
        fig.update_xaxes(showgrid=False, gridwidth=1, gridcolor='lightgray', tickfont=dict(size=10))
        fig.update_yaxes(showgrid=False, gridwidth=1, gridcolor='lightgray', tickfont=dict(size=10))
        
        # Display the subplot
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a styled dataframe display
        styled_df = results_df.style.set_properties(**{
            'background-color': 'white',
            'color': 'black',
            'border-color': 'lightgray'
        }).format("{:.4f}").highlight_max(
            subset=["R² Score"],
            color='lightgreen'
        ).highlight_min(
            subset=["Mean Absolute Error", "Root Mean Square Error"],
            color='lightsalmon'
        )
        
        # Display the results table
        st.dataframe(styled_df)
        
    with tab2:
        st.subheader("Feature Importance Analysis")
        
        # Plotly Feature Importance Visualization
        fig_importance = go.Figure(
            data=[go.Bar(
                x=feature_selection_results['feature_importance']['Feature'],
                y=feature_selection_results['feature_importance']['Importance'],
                text=[f'{imp:.4f}' for imp in feature_selection_results['feature_importance']['Importance']],
                textposition='auto'
            )]
        )
        fig_importance.update_layout(
            title="Feature Importances",
            xaxis_title="Features",
            yaxis_title="Importance"
        )
        
        st.plotly_chart(fig_importance)
        st.dataframe(feature_selection_results['feature_importance'])
    
    with tab3:
        st.subheader("Prediction vs Actual Comparison")
        
        # Model selection for visualization
        selected_model = st.selectbox(
            "Select Model for Visualization", 
            list(models.keys())
        )
        
        # Scatter plot of predictions vs actual values
        fig_pred_actual = px.scatter(
            x=feature_selection_results['y_test'], 
            y=model_predictions[selected_model],
            labels={'x': 'Actual Values', 'y': 'Predicted Values'},
            title=f'Predictions vs Actual Values - {selected_model}'
        )
        
        # Add diagonal line for perfect predictions
        fig_pred_actual.add_trace(
            go.Scatter(
                x=[feature_selection_results['y_test'].min(), feature_selection_results['y_test'].max()],
                y=[feature_selection_results['y_test'].min(), feature_selection_results['y_test'].max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            )
        )
        
        st.plotly_chart(fig_pred_actual)
    
    with tab4:
        st.subheader("Predict Total Claim Amount")
        
        # Create input widgets for each feature
        user_inputs = {}
        
        # Numerical features with sliders
        st.write("Numerical Features")
        user_inputs['Monthly Premium Auto'] = st.slider(
            "Monthly Premium Auto", 
            min_value=float(df['Monthly Premium Auto'].min()), 
            max_value=float(df['Monthly Premium Auto'].max()), 
            value=float(df['Monthly Premium Auto'].mean())
        )
        user_inputs['Income'] = st.slider(
            "Income", 
            min_value=float(df['Income'].min()), 
            max_value=float(df['Income'].max()), 
            value=float(df['Income'].mean())
        )
        user_inputs['Customer Lifetime Value'] = st.slider(
            "Customer Lifetime Value", 
            min_value=float(df['Customer Lifetime Value'].min()), 
            max_value=float(df['Customer Lifetime Value'].max()), 
            value=float(df['Customer Lifetime Value'].mean())
        )
        user_inputs['Months Since Last Claim'] = st.slider(
            "Months Since Last Claim", 
            min_value=int(df['Months Since Last Claim'].min()), 
            max_value=int(df['Months Since Last Claim'].max()), 
            value=int(df['Months Since Last Claim'].mean())
        )
        
        # Categorical features with selectboxes
        st.write("Categorical Features")
        # Location
        location_options = ['Location_Suburban', 'Location_Rural', 'Location_Urban']
        user_inputs['Location'] = st.selectbox("Location", location_options)
        
        # Gender
        gender_options = ['Gender_M', 'Gender_F']
        user_inputs['Gender'] = st.selectbox("Gender", gender_options)
        
        # Marital Status
        marital_options = ['Marital Status_Single', 'Marital Status_Married', 'Marital Status_Divorced']
        user_inputs['Marital Status'] = st.selectbox("Marital Status", marital_options)
        
        # Prediction button
        if st.button("Predict Claim Amount"):
            # Prepare input data
            input_data = {
                'Monthly Premium Auto': user_inputs['Monthly Premium Auto'],
                'Income': user_inputs['Income'],
                'Customer Lifetime Value': user_inputs['Customer Lifetime Value'],
                'Months Since Last Claim': user_inputs['Months Since Last Claim']
            }
            
            # Add one-hot encoded categorical features
            input_data[user_inputs['Location']] = 1
            input_data[user_inputs['Gender']] = 1
            input_data[user_inputs['Marital Status']] = 1
            
            # Convert to DataFrame
            user_df = pd.DataFrame([input_data])
            
            # Ensure all columns are in the correct order
            user_df = user_df.reindex(columns=feature_selection_results['original_columns'], fill_value=0)
            
            # Transform input
            user_transformed = feature_selection_results['preprocessor'].transform(user_df)
            
            # Predict using both models
            predictions = {}
            for name, model in models.items():
                predictions[name] = model.predict(user_transformed)[0]
            
            # Display predictions
            st.subheader("Predicted Total Claim Amounts")
            pred_df = pd.DataFrame.from_dict(predictions, orient='index', columns=['Predicted Amount'])
            st.dataframe(pred_df)

# Run the app
if __name__ == "__main__":
    run_advanced_modeling()