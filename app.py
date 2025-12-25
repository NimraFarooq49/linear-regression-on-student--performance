import streamlit as st
import numpy as np
import pickle

# ================================
# Load Trained Model
# ================================
model_path = "linear_regression_student_performance.pkl"

with open(model_path, "rb") as file:
    model = pickle.load(file)

# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="Student Performance Prediction", layout="centered")

st.title("🎓 Student Performance Prediction App")
st.write("Enter student details to predict performance score")

# ================================
# Input Fields (MATCH DATASET)
# ================================
hours_studied = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, step=0.5)
previous_scores = st.number_input("Previous Scores", min_value=0.0, max_value=100.0, step=1.0)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, step=1.0)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, step=0.5)
practice_tests = st.number_input("Number of Practice Tests", min_value=0, step=1)

# ================================
# Prediction
# ================================
if st.button("Predict Performance"):
    input_data = np.array([[hours_studied, previous_scores, attendance, sleep_hours, practice_tests]])
    prediction = model.predict(input_data)

    st.success(f"📊 Predicted Performance Index: {prediction[0]:.2f}")
