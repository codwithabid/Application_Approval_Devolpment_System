import streamlit as st
import pandas as pd
from custom_transformers import KNNImputerCustom, OneHotEncoderCustom, StandardScalerWithExclusion, OutlierImputer
import joblib

# Load the preprocessing pipeline and trained model
full_pipeline = joblib.load('Preprocessing_Pipeline.pkl')  # Preprocessing pipeline
loaded_model = joblib.load('Application_Approval_Model.pkl')  # Trained model

# Function to get user input
def get_user_input():
    CoBorrowerTotalMonthlyIncome = st.number_input("Enter Co-Borrower Total Monthly Income: ", min_value=0.0, value=0.0)
    CoBorrowerAge = st.number_input("Enter Co-Borrower Age: ", min_value=0, value=0)
    CoBorrowerYearsInSchool = st.number_input("Enter Co-Borrower Years In School: ", min_value=0, value=0)
    BorrowerTotalMonthlyIncome = st.number_input("Enter Borrower Total Monthly Income: ", min_value=0.0, value=0.0)
    BorrowerAge = st.number_input("Enter Borrower Age: ", min_value=0, value=0)
    DTI = st.number_input("Enter DTI (Debt-to-Income Ratio): ", min_value=0.0, value=0.0)
    CLTV = st.number_input("Enter CLTV (Loan-to-Value Ratio): ", min_value=0.0, value=0.0)
    CreditScore = st.number_input("Enter Credit Score: ", min_value=0, value=0)
    TotalLoanAmount = st.number_input("Enter Total Loan Amount: ", min_value=0.0, value=0.0)
    LeadSourceGroup = st.selectbox("Select Lead Source Group", ["TV", "Self Sourced", "Internet", "Radio", "Referral", "Repeat Client", "Direct Mail", "3rd Party", "Social Media"])
    Group = st.text_input("Enter Group (e.g., Admin, Loan Coordinator, Refinance Team - #number): ")
    LoanPurpose = st.selectbox("Select Loan Purpose", ["Purchase", "VA IRRRL", "Refinance Cash-out", "FHA Streamlined Refinance"])

    return {
        "CoBorrowerTotalMonthlyIncome": CoBorrowerTotalMonthlyIncome,
        "CoBorrowerAge": CoBorrowerAge,
        "CoBorrowerYearsInSchool": CoBorrowerYearsInSchool,
        "BorrowerTotalMonthlyIncome": BorrowerTotalMonthlyIncome,
        "BorrowerAge": BorrowerAge,
        "DTI": DTI,
        "CLTV": CLTV,
        "CreditScore": CreditScore,
        "TotalLoanAmount": TotalLoanAmount,
        "LeadSourceGroup": LeadSourceGroup,
        "Group": Group,
        "LoanPurpose": LoanPurpose
    }

# Streamlit UI layout
st.set_page_config(page_title="Application Approval System", layout="centered")

st.title("Application Approval System")
st.write("Please enter the details to check the Loan Application Approval status.")

# Get user input
user_input = get_user_input()

# Add a button to check the loan status
if st.button("Check Status"):
    # Convert user input into a DataFrame
    user_input_df = pd.DataFrame([user_input])

    # Preprocess the input using the saved pipeline
    user_input_transformed = full_pipeline.transform(user_input_df)

    # Make a prediction using the trained model
    prediction = loaded_model.predict(user_input_transformed)

    # Show the prediction result
    if prediction == 1:
        st.success("Loan Approved!")
    else:
        st.error("Loan Denied!")
