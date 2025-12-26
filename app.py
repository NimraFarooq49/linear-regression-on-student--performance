import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Model file ka rasta
model_path = "linear_regression_student_performance.pkl"

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Prediction App")
st.write("Linear Regression model based on student performance dataset")

# Check karna ke model file exist karti hai ya nahi
if not os.path.exists(model_path):
    st.error(f"⚠️ Error: '{model_path}' file nahi mili! Pehle training script run karke model save karein aur GitHub par upload karein.")
else:
    # Load Trained Model
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    st.header("Enter Student Details")

    # Input fields
    hours_studied = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, value=5.0)
    previous_scores = st.number_input("Previous Scores", min_value=0.0, max_value=100.0, value=70.0)
    attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=80.0)
    sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=12.0, value=7.0)
    sample_papers = st.number_input("Sample Question Papers Practiced", min_value=0.0, max_value=50.0, value=10.0)

    if st.button("Predict Performance"):
        # Input data array
        input_data = np.array([[hours_studied, previous_scores, attendance, sleep_hours, sample_papers]])
        prediction = model.predict(input_data)
        st.success(f"📊 Predicted Performance Index: **{prediction[0]:.2f}**")
