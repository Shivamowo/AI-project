import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the pre-trained model
@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_model.pkl")

model = load_model()

# Load dataset for reference (not training)
@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")

data = load_data()

# Title and Navigation
st.title("Credit Card Fraud Detection")
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Menu", ["Overview", "EDA", "Predictions"])

if menu == "Overview":
    st.header("Dataset Overview")
    st.write("This dataset contains credit card transactions labeled as fraudulent or genuine.")
    st.write(data.head())
    st.write(f"Shape of the dataset: {data.shape}")
    st.write(data.describe())

elif menu == "EDA":
    st.header("Exploratory Data Analysis")
    
    # Class Distribution
    if st.checkbox("Show Class Distribution"):
        st.write(data["Class"].value_counts())
        fig, ax = plt.subplots()
        data["Class"].value_counts().plot(kind="bar", ax=ax, color=['blue', 'orange'])
        ax.set_title("Class Distribution")
        ax.set_xlabel("Class")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    # Correlation Heatmap
    if st.checkbox("Show Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data.corr(), cmap="coolwarm", annot=False, ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
    
    # Time and Amount Distribution
    if st.checkbox("Show Time and Amount Distribution"):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        
        sns.histplot(data['Time'], bins=50, color='green', ax=ax[0])
        ax[0].set_title("Transaction Time Distribution")
        ax[0].set_xlabel("Time")
        
        sns.histplot(data['Amount'], bins=50, color='purple', ax=ax[1])
        ax[1].set_title("Transaction Amount Distribution")
        ax[1].set_xlabel("Amount")
        
        st.pyplot(fig)

elif menu == "Predictions":
    st.header("Predict Fraudulent Transactions")
    
    st.write("Enter transaction details for prediction:")
    input_data = {col: st.number_input(col, value=0.0) for col in data.columns[:-1]}
    
    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        result = "Fraudulent" if prediction[0] == 1 else "Genuine"
        st.success(f"The transaction is {result}.")
