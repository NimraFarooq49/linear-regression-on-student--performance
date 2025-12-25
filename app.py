import streamlit as st
import pandas as pd
import pickle

# Load model bundle
with open("student_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
FEATURES = data["features"]

st.title("🎓 Student Performance Predictor")

# Inputs
hours = st.number_input("Hours Studied", 0.0, 24.0)
previous = st.number_input("Previous Scores", 0.0, 100.0)

extra = st.selectbox(
    "Extracurricular Activities",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

sleep = st.number_input("Sleep Hours", 0.0, 24.0)
papers = st.number_input("Sample Question Papers Practiced", 0)

# Create DataFrame (KEY FIX 🔥)
input_df = pd.DataFrame([[
    hours, previous, extra, sleep, papers
]], columns=FEATURES)

if st.button("Predict Performance"):
    prediction = model.predict(input_df)
    st.success(f"📊 Predicted Performance Index: {prediction[0]:.2f}")

