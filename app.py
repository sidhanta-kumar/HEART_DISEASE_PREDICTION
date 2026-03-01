import streamlit as st
import numpy as np
import joblib

st.title("Heart Disease Prediction App")

# Load saved files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.header("Enter Patient Details")

age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200)
chol = st.number_input("Cholesterol", 100, 400)
thalach = st.number_input("Max Heart Rate", 60, 220)

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, thalach]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")