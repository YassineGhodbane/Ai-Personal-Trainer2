import streamlit as st
import pandas as pd
import joblib
import os
from pyngrok import ngrok

# Start ngrok tunnel for port 8501
if "PUBLIC_URL" not in st.session_state:
    public_url = ngrok.connect(8501)
    st.session_state.PUBLIC_URL = public_url
    st.write(f"🌐 Public URL: [Click here to access]( {public_url} )")

st.title("🏋️‍♂️ Fitness Goal Predictor")
st.write("Enter your personal details to predict your ideal fitness goal.")

# Load model and feature names
model_path = 'fitness_goal_model.pkl'
features_path = 'feature_names.pkl'

if not os.path.exists(model_path) or not os.path.exists(features_path):
    st.error("🚫 Model files not found. Ensure 'fitness_goal_model.pkl' and 'feature_names.pkl' exist.")
    st.stop()

model = joblib.load(model_path)
feature_names = joblib.load(features_path)

# Input form
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
age = st.slider("Age", 1, 100, 25)
height = st.slider("Height (cm)", 100, 250, 175)
weight = st.slider("Weight (kg)", 30, 300, 70)
hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
level = st.selectbox("Fitness Level", [0, 1, 2], format_func=lambda x: ["Beginner", "Intermediate", "Advanced"][x])

# BMI Calculation
bmi = weight / ((height / 100) ** 2)
st.write(f"📊 Your calculated BMI: **{bmi:.2f}**")

# Prediction
if st.button("🔍 Predict Fitness Goal"):
    input_data = {
        'Sex': sex,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'Hypertension': hypertension,
        'Diabetes': diabetes,
        'BMI': bmi,
        'Level': level
    }
    input_df = pd.DataFrame([input_data])

    try:
        input_df = input_df[feature_names]
        prediction = model.predict(input_df)
        label_map = {0: "Weight Loss", 1: "Weight Gain"}
        predicted_label = label_map.get(prediction[0], "Unknown")
        st.success(f"🏁 **Predicted Fitness Goal:** {predicted_label}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
