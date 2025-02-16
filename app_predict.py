import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model and scaler
model = joblib.load("VotingClassifier.joblib")
scaler = joblib.load("scaler.joblib")

# Define input features
feature_names = [
    "Age",
    "Gender",
    "Diabetes mellitus",
    "Chronic kidney disease",
    "Burning Micturition",
    "Fever",
    "Nausea or vomiting",
    "WBC count(cells/cumm)",
    "Urine leucocytes",
    "Fatigue",
    "Lower abdominal pain",
    "Serum Creatinine (mg/dL)",
    "Indwelling foley catheter",
    "Dysuria(painful urination)",
]

# Streamlit UI
st.title("UTI Prediction")
st.write("Enter the required details to predict whether UTI is present or not.")

# User inputs
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.radio("Gender", ("Male", "Female"))
diabetes = st.radio("Diabetes Mellitus", ("Yes", "No"))
kidney_disease = st.radio("Chronic Kidney Disease", ("Yes", "No"))
burning_micturition = st.radio("Burning Micturition", ("Yes", "No"))
fever = st.radio("Fever", ("Yes", "No"))
nausea = st.radio("Nausea or Vomiting", ("Yes", "No"))
wbc_count = st.number_input("WBC Count (cells/cumm)", min_value=0.0, value=0.0)
urine_leucocytes = st.number_input("Urine Leucocytes", min_value=0.0, value=0.0)
fatigue = st.radio("Fatigue", ("Yes", "No"))
lower_abd_pain = st.radio("Lower Abdominal Pain", ("Yes", "No"))
serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, value=0.0)
indwelling_catheter = st.radio("Indwelling Foley Catheter", ("Yes", "No"))
dysuria = st.radio("Dysuria (Painful Urination)", ("Yes", "No"))

# Convert inputs
input_data = np.array(
    [
        age,
        1 if gender == "Male" else 0,
        1 if diabetes == "Yes" else 0,
        1 if kidney_disease == "Yes" else 0,
        1 if burning_micturition == "Yes" else 0,
        1 if fever == "Yes" else 0,
        1 if nausea == "Yes" else 0,
        wbc_count,
        urine_leucocytes,
        1 if fatigue == "Yes" else 0,
        1 if lower_abd_pain == "Yes" else 0,
        serum_creatinine,
        1 if indwelling_catheter == "Yes" else 0,
        1 if dysuria == "Yes" else 0,
    ]
).reshape(1, -1)

# Scale input data
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    result = "UTI Present" if prediction == 1 else "UTI Absent"
    st.write(f"Prediction: **{result}**")
