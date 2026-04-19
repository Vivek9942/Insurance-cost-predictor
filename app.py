import streamlit as st
import numpy as np
import pickle
import os

# Set page config FIRST
st.set_page_config(page_title="Insurance Cost Predictor", layout="centered")

# Safe path handling
BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

# Load files safely
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

st.title("💰 Insurance Cost Prediction App")
st.write("Enter details to predict insurance cost")

# -------- INPUTS --------
age = st.slider("Age", 18, 65, 25)

gender = st.selectbox("Gender", ["Male", "Female"])
is_female = 1 if gender == "Female" else 0

bmi = st.slider("BMI", 15.0, 55.0, 25.0)

children = st.slider("Children", 0, 5, 0)

smoker = st.selectbox("Smoker", ["No", "Yes"])
is_smoker = 1 if smoker == "Yes" else 0

region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])
region_southeast = 1 if region == "Southeast" else 0

# BMI category
bmi_category_obesity = 1 if bmi > 30 else 0

# -------- SCALING --------
input_data = np.array([[age, bmi, children]])
input_scaled = scaler.transform(input_data)

age_s, bmi_s, children_s = input_scaled[0]

# Final input
final_input = np.array([[age_s, is_female, bmi_s, children_s,
                         is_smoker, region_southeast,
                         bmi_category_obesity]])

# -------- PREDICTION --------
if st.button("Predict"):
    prediction = model.predict(final_input)[0]
    st.success(f"💵 Estimated Insurance Cost: ₹ {round(prediction, 2)}")
