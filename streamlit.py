import streamlit as st
import joblib
import numpy as np

st.header=("Loan Approval Classification using Decision Tree")

init_payment=st.number_input("Enter the initial payment amount")
last_payment=st.number_input("Enter the last payment amount")
credit_score=st.number_input("Enter the Credit score")

model = joblib.load(r"C:\Users\hp\ML file\Loan_clissification\loan_model_DT.pkl")


input_array=np.array([[init_payment,last_payment,credit_score]])

button=st.button("SUBMIT")

if button:
    output=model.predict(input_array)
    st.info(f"Your loan status is: {output}")
