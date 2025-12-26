import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ================================
# Load Trained Model
# ================================
model_path = "linear_regression_student_performance.pkl"

with open(model_path, "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Prediction App")
st.write("Linear Regression model based on student performance dataset")

# ================================
# User Input Section
# ================================
st.header("Enter Student Details")

# ⚠️ IMPORTANT:
# Ye input columns tumhare dataset ke numeric columns ke naam ke hisaab se hone chahiye
# Example values common Kaggle dataset ke basis par diye gaye hain

hours_studied = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, value=5.0)
previous_scores = st.number_input("Previous Scores", min_value=0.0, max_value=100.0, value=70.0)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=80.0)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=12.0, value=7.0)
sample_papers = st.number_input("Sample Question Papers Practiced", min_value=0.0, max_value=50.0, value=10.0)

# ================================
# Prediction
# ================================
if st.button("Predict Performance"):
    input_data = np.array([[hours_studied, previous_scores, attendance, sleep_hours, sample_papers]])
    prediction = model.predict(input_data)

    st.success(f"📊 Predicted Performance Index: **{prediction[0]:.2f}**")
