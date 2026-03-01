import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model, scaler and training columns
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")  # make sure you saved this during training

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction App")

st.warning(
    "⚠️ Disclaimer: This model is trained on a small academic dataset "
    "and is for educational purposes only. It is NOT a medical diagnosis."
)

st.info("📊 Model Test Accuracy: 83.6%")

st.write("Enter patient details below:")

# ------------------ INPUT FIELDS ------------------

age = st.number_input("Age", 1, 120)

sex = st.selectbox("Sex", ["Female", "Male"])

cp = st.selectbox("Chest Pain Type", [
    "Typical Angina",
    "Atypical Angina",
    "Non-anginal Pain",
    "Asymptomatic"
])

trestbps = st.number_input("Resting Blood Pressure", 50, 250)

chol = st.number_input("Cholesterol", 100, 600)

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

restecg = st.selectbox("Rest ECG", [
    "Normal",
    "ST-T Wave Abnormality",
    "Left Ventricular Hypertrophy"
])

thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220)

exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])

oldpeak = st.number_input("Oldpeak", 0.0, 10.0, step=0.1)

slope = st.selectbox("Slope", [
    "Upsloping",
    "Flat",
    "Downsloping"
])

ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])

thal = st.selectbox("Thal", [
    "Normal",
    "Fixed Defect",
    "Reversible Defect"
])

# ------------------ PREDICTION ------------------

if st.button("Predict"):

    # -------- Convert Objects to Numeric --------
    sex = 1 if sex == "Male" else 0

    cp_map = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }
    cp = cp_map[cp]

    fbs = 1 if fbs == "Yes" else 0

    restecg_map = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }
    restecg = restecg_map[restecg]

    exang = 1 if exang == "Yes" else 0

    slope_map = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }
    slope = slope_map[slope]

    thal_map = {
        "Normal": 1,
        "Fixed Defect": 2,
        "Reversible Defect": 3
    }
    thal = thal_map[thal]

    # -------- Create DataFrame --------
    input_data = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak,
        slope, ca, thal
    ]], columns=[
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal"
    ])

    # -------- Apply same preprocessing --------

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    # -------- Output --------
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")

    st.write(f"Risk Probability: {probability*100:.2f}%")

    if probability < 0.3:
        st.write("🟢 Risk Level: Low")
    elif probability < 0.7:
        st.write("🟡 Risk Level: Moderate")
    else:
        st.write("🔴 Risk Level: High")

