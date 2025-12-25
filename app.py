import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load Model
@st.cache_resource
def load_model():
    # File name aapki uploaded file ke mutabiq hai
    with open("linear_regression_student_performance.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("🎓 Student GPA Predictor")

with st.form("main_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        s_id = st.number_input("Student ID", value=1001)
        age = st.number_input("Age", 15, 20, 18)
        gender = st.selectbox("Gender (0=F, 1=M)", [0, 1])
        eth = st.selectbox("Ethnicity (0-3)", [0, 1, 2, 3])
        p_edu = st.selectbox("Parental Education (0-4)", [0, 1, 2, 3, 4])
        study = st.number_input("Study Time Weekly (Hours)", 0, 40, 10)
        absent = st.number_input("Absences", 0, 30, 0)

    with col2:
        tutor = st.selectbox("Tutoring (0=No, 1=Yes)", [0, 1])
        p_supp = st.selectbox("Parental Support (0-4)", [0, 1, 2, 3, 4])
        extra = st.selectbox("Extracurricular (0/1)", [0, 1])
        sports = st.selectbox("Sports (0/1)", [0, 1])
        music = st.selectbox("Music (0/1)", [0, 1])
        vol = st.selectbox("Volunteering (0/1)", [0, 1])
        dummy_gpa = 0.0 # Ye 14th feature hai jo model ko chahiye

    submit = st.form_submit_button("Predict Now")

if submit:
    # Model expects 14 features in this exact order 
    features = np.array([[s_id, age, gender, eth, p_edu, study, absent, 
                          tutor, p_supp, extra, sports, music, vol, dummy_gpa]])
    
    prediction = model.predict(features)
    st.success(f"### Predicted GPA: {prediction[0]:.2f}")
