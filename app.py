import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Dark theme setup using Streamlit config
st.set_page_config(page_title="Diabetes Predictor", layout="wide", page_icon="🧬")

# Fully override the white background
st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: white;
    }
    section[data-testid="stSidebar"] {
        background-color: #1c1c1c;
    }
    h1, h2, h3, h4, h5, h6, .css-10trblm {
        color: white;
    }
    .css-1d391kg, .css-1v0mbdj, .css-qrbaxs {
        color: white;
    }
    .css-1ec096l, .css-1y4p8pa, .css-hyum1k {
        color: white !important;
    }
    .stButton>button {
        background-color: #00b894;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stProgress > div > div > div > div {
        background-color: #00ff99;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🧬 Diabetes Prediction System")

# Load trained model
with open("trained_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]
model_name = data["model_name"]

st.write(f"This app uses **{model_name}** to predict if a patient is likely diabetic based on medical parameters.")

# Sidebar input
st.sidebar.header("🩺 Enter Patient Medical Data")
def user_input():
    return pd.DataFrame({
        "Pregnancies": [st.sidebar.slider("Pregnancies", 0, 20, 1)],
        "Glucose": [st.sidebar.slider("Glucose", 0, 200, 120)],
        "BloodPressure": [st.sidebar.slider("Blood Pressure", 0, 140, 70)],
        "SkinThickness": [st.sidebar.slider("Skin Thickness", 0, 100, 20)],
        "Insulin": [st.sidebar.slider("Insulin", 0, 900, 80)],
        "BMI": [st.sidebar.slider("BMI", 0.0, 70.0, 25.0)],
        "DiabetesPedigreeFunction": [st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)],
        "Age": [st.sidebar.slider("Age", 1, 100, 30)],
    })

input_df = user_input()

# Prediction
if st.button("Predict"):
    if scaler:
        input_scaled = scaler.transform(input_df)
    else:
        input_scaled = input_df
    prediction = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled) if hasattr(model, "predict_proba") else None

    # Result section
    st.subheader("🔍 Prediction Result")
    if prediction[0] == 1:
        st.markdown("### 🔴 The patient is likely **Diabetic** ❗", unsafe_allow_html=True)
    else:
        st.markdown("### 🟢 The patient is likely **Not Diabetic** ✅", unsafe_allow_html=True)

    # Probabilities
    if proba is not None:
        st.subheader("📊 Prediction Probabilities")
        st.progress(proba[0][1])
        st.markdown(f"**🔴 Diabetic Probability:** `{proba[0][1]:.2f}`", unsafe_allow_html=True)
        st.markdown(f"**🟢 Not Diabetic Probability:** `{proba[0][0]:.2f}`", unsafe_allow_html=True)
