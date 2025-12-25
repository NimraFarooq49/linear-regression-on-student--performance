import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page Configuration
st.set_page_config(page_title="Student GPA Predictor", layout="centered")

# Load the saved model data
@st.cache_resource
def load_model():
    try:
        with open("linear_regression_student_performance.pkl", "rb") as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        st.error("Model file (.pkl) nahi mili. Please check the file name.")
        return None

data = load_model()

if data:
    # Agar model dictionary hai toh model aur features nikaalein
    if isinstance(data, dict):
        model = data.get("model")
        # .pkl file ke mutabiq exact features 
        features = data.get("features", [
            'StudentID', 'Age', 'Gender', 'Ethnicity', 'ParentalEducation', 
            'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 
            'Extracurricular', 'Sports', 'Music', 'Volunteering', 'GPA'
        ])
    else:
        model = data
        features = getattr(model, "feature_names_in_", None)

    st.title("🎓 Student GPA Prediction App")
    st.write("Student ki details enter karein aur GPA predict karein.")

    st.divider()

    # Form for User Input
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=15, max_value=20, value=18)
            gender = st.selectbox("Gender (0=Female, 1=Male)", [0, 1])
            ethnicity = st.selectbox("Ethnicity (0-3)", [0, 1, 2, 3])
            parental_edu = st.selectbox("Parental Education (0-4)", [0, 1, 2, 3, 4])
            study_time = st.number_input("Study Time Weekly (Hours)", min_value=0, max_value=40, value=10)
            absences = st.number_input("Absences", min_value=0, max_value=30, value=0)
            tutoring = st.selectbox("Tutoring (0=No, 1=Yes)", [0, 1])

        with col2:
            parental_support = st.selectbox("Parental Support (0-4)", [0, 1, 2, 3, 4])
            extracurricular = st.selectbox("Extracurricular (0=No, 1=Yes)", [0, 1])
            sports = st.selectbox("Sports (0=No, 1=Yes)", [0, 1])
            music = st.selectbox("Music (0=No, 1=Yes)", [0, 1])
            volunteering = st.selectbox("Volunteering (0=No, 1=Yes)", [0, 1])
            student_id = st.number_input("Student ID (System Ref)", value=1001)
            # Input features must match model expectations 
            dummy_gpa = 0.0 # Standard input ke liye

        submit = st.form_submit_button("Predict GPA")

    if submit:
        # Features array banayein (Exactly usi sequence mein jo .pkl file mein hai) 
        input_data = np.array([[
            student_id, age, gender, ethnicity, parental_edu, 
            study_time, absences, tutoring, parental_support, 
            extracurricular, sports, music, volunteering, dummy_gpa
        ]])

        try:
            prediction = model.predict(input_data)
            st.success(f"### 🔮 Predicted GPA: {prediction[0]:.2f}")
            
            # Progress bar for visual GPA representation
            gpa_val = float(prediction[0])
            st.progress(min(max(gpa_val / 4.0, 0.0), 1.0))
        except Exception as e:
            st.error(f"Prediction Error: {e}")