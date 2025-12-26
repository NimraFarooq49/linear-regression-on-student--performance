import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os

# Page Settings
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

# --- STEP 1: DATA LOADING & MODEL TRAINING ---
# Hum app ke andar hi model train kar rahe hain taake .pkl file ka lafda khatam ho jaye
@st.cache_resource
def train_model():
    # Dataset ka rasta (Ensure karein ke ye file aapke GitHub par ho)
    data_path = "Student_performance_data _.csv" 
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        # Sirf numeric columns le rahe hain
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Target variable selection
        target = "Performance Index" if "Performance Index" in numeric_df.columns else numeric_df.columns[-1]
        
        X = numeric_df.drop(columns=[target])
        y = numeric_df[target]
        
        # Model Training
        model = LinearRegression()
        model.fit(X, y)
        return model, X.columns.tolist()
    else:
        return None, None

model, feature_names = train_model()

# --- STEP 2: USER INTERFACE ---
st.title("🎓 Student Performance Predictor")
st.write("Yeh app aapki performance index predict karti hai.")

if model is None:
    st.error("⚠️ Dataset file (Student_performance_data _.csv) nahi mili! Isse GitHub par upload karein.")
else:
    st.header("Input Student Details")
    
    # Dynamic Input Fields based on Dataset
    user_inputs = []
    cols = st.columns(2)
    
    for i, col_name in enumerate(feature_names):
        with cols[i % 2]:
            val = st.number_input(f"Enter {col_name}", value=0.0)
            user_inputs.append(val)

    if st.button("Predict Performance"):
        prediction = model.predict([user_inputs])
        st.success(f"🎯 Predicted Result: {prediction[0]:.2f}")

# --- EDA SECTION (Optional) ---
if st.checkbox("Show Data Analysis (EDA)"):
    df = pd.read_csv("Student_performance_data _.csv")
    st.subheader("Data Distribution")
    st.bar_chart(df.iloc[:, :5]) # Pehle 5 columns ka chart