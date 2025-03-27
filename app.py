import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
def load_data():
    file_path = "AutoInsuranceClaims2024.csv"
    df = pd.read_csv(file_path)
    return df

def display_eda(df):
    st.subheader("Exploratory Data Analysis")
    st.write("### Data Preview")
    st.dataframe(df.head())
    
    st.write("### Summary Statistics")
    st.write(df.describe())
    
    st.write("### Missing Values")
    st.write(df.isnull().sum())
    
    st.write("### Distribution of Target Variable")
    if 'ClaimStatus' in df.columns:
        fig, ax = plt.subplots()
        sns.countplot(x=df['ClaimStatus'], ax=ax)
        st.pyplot(fig)


def generate_insights(df):
    st.subheader("Key Insights")
    
    st.write("### Correlation Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    st.write("### High Claim Amount Cases")
    if 'ClaimAmount' in df.columns:
        high_claims = df[df['ClaimAmount'] > df['ClaimAmount'].quantile(0.95)]
        st.write(high_claims[['ClaimAmount', 'PolicyType', 'VehicleAge']].head(10))

def train_model(df):
    st.subheader("Modeling")
    
    if 'ClaimStatus' not in df.columns:
        st.error("Target variable 'ClaimStatus' not found in dataset.")
        return
    
    # Preprocess data
    df = df.dropna()
    X = df.drop(columns=['ClaimStatus'])
    y = df['ClaimStatus']
    
    # Convert categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Display results
    st.write("### Model Accuracy: ", accuracy_score(y_test, y_pred))
    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))

def main():
    st.title("Auto Insurance Claims Analysis")
    
    df = load_data()
    
    menu = st.sidebar.radio("Navigation", ["EDA", "Analysis", "Modeling"])
    
    if menu == "EDA":
        display_eda(df)
    elif menu == "Analysis":
        generate_insights(df)
    elif menu == "Modeling":
        train_model(df)
    
if __name__ == "__main__":
    main()
