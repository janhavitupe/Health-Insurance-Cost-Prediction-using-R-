import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load saved objects
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
le_gender = joblib.load("label_encoder_gender.pkl")
le_smoker = joblib.load("label_encoder_smoker.pkl")
le_diabetic = joblib.load("label_encoder_diabetic.pkl")

# Page setup
st.set_page_config(page_title="🏥 Health Insurance Cost Predictor", layout="centered")
st.title("🏥 Health Insurance Cost Prediction")
st.write("Enter the details below to predict the estimated health insurance cost.")

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    with col2:
        bloodpressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=120)
        gender = st.selectbox("Gender", options=list(le_gender.classes_))
        smoker = st.selectbox("Smoker", options=list(le_smoker.classes_))
        diabetic = st.selectbox("Diabetic", options=list(le_diabetic.classes_))
    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    try:
        # Encode categorical values
        gender_enc = le_gender.transform([gender])[0]
        smoker_enc = le_smoker.transform([smoker])[0]
        diabetic_enc = le_diabetic.transform([diabetic])[0]

        # Create DataFrame in correct order
        input_data = pd.DataFrame([[
            age,
            gender_enc,
            bmi,
            bloodpressure,
            diabetic_enc,
            children,
            smoker_enc
        ]], columns=["age", "gender", "bmi", "bloodpressure", "diabetic", "children", "smoker"])

        # Scale only numerical columns
        num_cols = ["age", "bmi", "bloodpressure", "children"]
        input_data[num_cols] = scaler.transform(input_data[num_cols])

        # Predict
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Predicted Health Insurance Cost: **${prediction:,.2f}**")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
