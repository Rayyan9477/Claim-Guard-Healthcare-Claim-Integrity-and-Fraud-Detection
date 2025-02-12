import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from model_training import load_data, preprocess_training_data, train_model, save_model, train_and_save_model

# Function to predict claim status
def predict_claim_status(model, patient_age, gender, insurance_provider, service_date, billing_code, diagnosis_code, place_of_service, claim_amount, submitted_charges, allowed_amount, copay_amount, deductible_amount):
    # Create input data as a DataFrame
    input_data = pd.DataFrame({
        'patient_age': [patient_age],
        'gender': [gender],
        'insurance_provider': [insurance_provider],
        'service_date': [service_date],
        'billing_code': [billing_code],
        'diagnosis_code': [diagnosis_code],
        'place_of_service': [place_of_service],
        'claim_amount': [claim_amount],
        'submitted_charges': [submitted_charges],
        'allowed_amount': [allowed_amount],
        'copay_amount': [copay_amount],
        'deductible_amount': [deductible_amount]
    })

    # Make prediction using the loaded model
    prediction = model.predict(input_data)

    # Return prediction result
    return "Denied" if prediction[0] == 1 else "Approved"

# Streamlit app
st.title("Claims Optimization System")
st.write("Enter claim details to predict approval status.  Adjust the input values below to match the claim you want to evaluate.")

# File paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(os.path.dirname(current_dir), 'data', 'dataset.csv')
model_path = os.path.join(os.path.dirname(current_dir), 'models', 'model.pkl')

# Check if the model exists
if not os.path.exists(model_path):
    model = train_and_save_model(data_path, model_path)
    st.success(f"Model saved to {model_path}")
else:
    model = joblib.load(model_path)

# Input fields
patient_age = st.number_input("Patient Age", value=50)
gender = st.selectbox("Gender", ["Male", "Female"])
insurance_provider = st.selectbox("Insurance Provider", ["Medicare", "Blue Cross", "Aetna", "United Healthcare", "Cigna"])
service_date = st.date_input("Service Date")
billing_code = st.text_input("Billing Code (CPT/HCPCS)", value="99214")
diagnosis_code = st.text_input("Diagnosis Code (ICD-10)", value="I10")
place_of_service = st.selectbox("Place of Service", ["Office", "Outpatient", "Inpatient"])
claim_amount = st.number_input("Claim Amount", value=250.00)
submitted_charges = st.number_input("Submitted Charges", value=250.00)
allowed_amount = st.number_input("Allowed Amount", value=200.00)
copay_amount = st.number_input("Copay Amount", value=20.00)
deductible_amount = st.number_input("Deductible Amount", value=50.00)

# Prediction button
if st.button("Predict Claim Status"):
    prediction = predict_claim_status(model, patient_age, gender, insurance_provider, str(service_date), billing_code, diagnosis_code, place_of_service, claim_amount, submitted_charges, allowed_amount, copay_amount, deductible_amount)
    st.write(f"Prediction: {prediction}")

    # Add an explanation and recommendations based on the prediction
    if prediction == "Denied":
        st.warning("This claim has a high risk of denial. Please review the following:")
        st.write("- Ensure all necessary documentation is attached.")
        st.write("- Check for prior authorization requirements.")
    else:
        st.success("This claim is likely to be approved.")

# --- Data Analysis and Visualizations ---
st.sidebar.header("Data Analysis")
show_eda = st.sidebar.checkbox("Show Exploratory Data Analysis", value=True)  # Set default to True

if show_eda:
    st.header("Exploratory Data Analysis")

    # Load the dataset
    data = load_data(data_path)

    # Display claim amount distribution
    st.subheader("Claim Amount Distribution")
    fig_claim_amount, ax_claim_amount = plt.subplots()
    sns.histplot(data['claim_amount'], bins=30, kde=True, ax=ax_claim_amount)
    st.pyplot(fig_claim_amount)

    # Display claim status counts
    st.subheader("Claim Status Counts")
    fig_claim_status, ax_claim_status = plt.subplots()
    sns.countplot(x='claim_status', data=data, ax=ax_claim_status)
    st.pyplot(fig_claim_status)

    # Display Insurance Provider Distribution
    st.subheader("Insurance Provider Distribution")
    fig_insurance, ax_insurance = plt.subplots()
    sns.countplot(x='insurance_provider', data=data, ax=ax_insurance)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig_insurance)

    # Display Place of Service Distribution
    st.subheader("Place of Service Distribution")
    fig_pos, ax_pos = plt.subplots()
    sns.countplot(x='place_of_service', data=data, ax=ax_pos)
    st.pyplot(fig_pos)

    # Claim Amount vs. Claim Status (Boxplot)
    st.subheader("Claim Amount vs. Claim Status")
    fig_boxplot, ax_boxplot = plt.subplots()
    sns.boxplot(x='claim_status', y='claim_amount', data=data, ax=ax_boxplot)
    st.pyplot(fig_boxplot)

    # Denial Reason Distribution
    st.subheader("Denial Reason Distribution")
    fig_denial, ax_denial = plt.subplots()
    sns.countplot(x='denial_reason', data=data, ax=ax_denial)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig_denial)