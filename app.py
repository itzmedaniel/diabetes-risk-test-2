import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("diabetes_model_rf.pkl")

st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Risk Prediction")

st.markdown("""
This tool estimates your risk of diabetes based on your health data.
Please enter your information below.
""")

# Systolic/Diastolic Blood Pressure Input
st.subheader("🩺 Blood Pressure")
systolic = st.slider("Systolic (SYS) mmHg", 80, 200, 120)
diastolic = st.slider("Diastolic (DIA) mmHg", 40, 130, 80)
pulse = st.slider("Pulse (bpm)", 40, 160, 70)
st.markdown("Normal SYS: 90–120, DIA: 60–80, Pulse: 60–100")

# Cholesterol Input
st.subheader("🧪 Cholesterol")
cholesterol = st.slider("Total Cholesterol (mg/dL)", 100, 400, 180)
st.markdown("Desirable: <200 mg/dL | Borderline high: 200–239 | High: ≥240")

# Heart Disease
st.subheader("❤️ Heart Disease")
heart_disease = st.radio("Do you have heart disease?", ["Yes", "No"])
heart_disease_type = ""
if heart_disease == "Yes":
    heart_disease_type = st.text_input("If known, specify the type (optional):")

# General Inputs
st.subheader("👤 Personal & Lifestyle")

age = st.slider("Age (years)", 13, 100, 30)
sex = st.selectbox("Sex", ["Male", "Female"])
bmi = st.slider("Body Mass Index (BMI)", 10.0, 60.0, 22.0)
physical_activity = st.selectbox("How often do you exercise?", ["None", "Rarely", "Sometimes", "Often", "Everyday"])
smoking = st.radio("Do you smoke?", ["Yes", "No"])
alcohol = st.radio("Do you drink alcohol?", ["Yes", "No"])
fruit_veg = st.selectbox("Do you eat fruits/vegetables daily?", ["No", "Sometimes", "Yes"])
mental_health = st.slider("Poor mental health days (past 30 days)", 0, 30, 0)
physical_health = st.slider("Poor physical health days (past 30 days)", 0, 30, 0)
sleep = st.slider("Average sleep per night (hours)", 0.0, 24.0, 7.0)

# Healthcare Access
healthcare = st.radio("Do you have access to healthcare?", ["Yes", "No"])

# Income
income = st.selectbox("Monthly Income", ["< RM1,000", "RM1,000–RM3,000", "RM3,001–RM5,000", "RM5,001–RM10,000", "> RM10,000"])

# Mapping user input into model features
def prepare_features():
    # Map lifestyle responses to scale
    activity_map = {"None": 0, "Rarely": 2.5, "Sometimes": 5, "Often": 7.5, "Everyday": 10}
    yesno_map = {"Yes": 1, "No": 0}
    fruitveg_map = {"No": 0, "Sometimes": 0.5, "Yes": 1}
    sex_map = {"Male": 1, "Female": 0}
    income_map = {
        "< RM1,000": 1,
        "RM1,000–RM3,000": 2,
        "RM3,001–RM5,000": 3,
        "RM5,001–RM10,000": 4,
        "> RM10,000": 5
    }

    features = [
        1 if systolic >= 130 or diastolic >= 80 else 0,  # HighBP
        1 if cholesterol >= 240 else 0,                  # HighChol
        yesno_map.get(smoking),                          # Smoker
        yesno_map.get(alcohol),                          # HvyAlcoholConsump
        fruitveg_map.get(fruit_veg),                     # FruitsVeggies
        bmi,                                             # BMI
        activity_map.get(physical_activity),             # PhysActivity
        yesno_map.get(heart_disease),                    # HeartDiseaseorAttack
        mental_health,                                   # MentHlth
        physical_health,                                 # PhysHlth
        sleep,                                           # AvgDailySleep
        yesno_map.get(healthcare),                       # DiffWalk (using as a proxy)
        income_map.get(income),                          # Income
        age,                                             # Age
        sex_map.get(sex),                                # Sex
        systolic,                                        # Custom: Systolic
        diastolic,                                       # Custom: Diastolic
        pulse,                                           # Custom: Pulse
        cholesterol,                                     # Custom: Cholesterol
    ]

    # Fill to 21 features if needed
    while len(features) < 21:
        features.append(0)

    return np.array(features).reshape(1, -1)

# Predict button
if st.button("🔍 Predict Diabetes Risk"):
    features = prepare_features()
    prediction = int(model.predict(features)[0])
    prob = model.predict_proba(features)[0]

    if prediction == 1:
        st.error("⚠️ High risk of diabetes.")
    else:
        st.success("✅ Low risk of diabetes.")
    
    st.markdown(f"**Confidence:** {prob[prediction]*100:.2f}%")

