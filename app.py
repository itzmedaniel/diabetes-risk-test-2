import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("diabetes_model_rf.pkl")

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Risk Prediction")

st.markdown("This app uses health indicators to predict your risk of diabetes. Please enter accurate information.")

with st.sidebar:
    st.header("Doctor Panel")
    show_info = st.checkbox("Show medical guidance", value=True)

# --- 1. Vitals: BP, Cholesterol, Pulse
st.subheader("1. Blood Pressure & Cholesterol")

systolic = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=200, value=120)
diastolic = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=140, value=80)
pulse = st.number_input("Resting Pulse (bpm)", min_value=40, max_value=180, value=75)

if show_info:
    st.markdown("""
    - **Normal BP**: below 120/80 mmHg  
    - **Elevated**: 120–129/<80  
    - **High**: ≥130 systolic or ≥80 diastolic  
    """)

# Convert to HighBP binary feature
high_bp = 1 if systolic >= 130 or diastolic >= 80 else 0

cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=180)

if show_info:
    st.markdown("""
    - **Desirable**: < 200 mg/dL  
    - **Borderline high**: 200–239  
    - **High**: ≥ 240 mg/dL  
    """)

# Convert to HighChol binary feature
high_chol = 1 if cholesterol >= 200 else 0

chol_check = st.radio("Had cholesterol checked in past 5 years?", ["No", "Yes"])

bmi = st.number_input("Body Mass Index (BMI)", 10.0, 60.0, 22.0)

# --- 2. Lifestyle

st.subheader("2. Lifestyle")

smoker = st.radio("Do you currently smoke?", ["No", "Yes"])
stroke = st.radio("History of stroke?", ["No", "Yes"])
heart_disease = st.radio("Coronary heart disease / heart attack history?", ["No", "Yes"])
phys_activity = st.radio("Physical activity outside work?", ["No", "Yes"])
fruits = st.radio("Eat fruits daily?", ["No", "Yes"])
veggies = st.radio("Eat vegetables daily?", ["No", "Yes"])
alcohol = st.radio("Heavy drinker? (Men >14 / Women >7 drinks per week)", ["No", "Yes"])

# --- 3. Healthcare Access

st.subheader("3. Healthcare Access")

any_healthcare = st.radio("Do you have healthcare coverage?", ["No", "Yes"])
no_doc_cost = st.radio("Unable to see doctor due to cost in past year?", ["No", "Yes"])

# --- 4. General & Mental Health

st.subheader("4. General & Mental Health")

genhlth = st.slider("General health (1=Excellent, 5=Poor)", 1, 5, 3)
menthlth = st.slider("Days mental health was not good (past 30 days)", 0, 30, 0)
physhlth = st.slider("Days physical health was not good (past 30 days)", 0, 30, 0)
diffwalk = st.radio("Serious difficulty walking/climbing stairs?", ["No", "Yes"])

# --- 5. Demographics

st.subheader("5. Demographics")

sex = st.radio("Biological Sex", ["Female", "Male"])
age = st.slider("Age", min_value=13, max_value=100, value=25, help="Select your current age (13–100 years old)")
education = st.slider("Education level (1=No school, 6=College grad)", 1, 6, 4)
income = st.selectbox(
    "Monthly Income (in RM)", 
    [
        "Less than RM1,000",
        "RM1,000 - RM1,999",
        "RM2,000 - RM2,999",
        "RM3,000 - RM3,999",
        "RM4,000 - RM4,999",
        "RM5,000 and above"
    ],
    help="Please estimate your gross monthly income in Malaysian Ringgit"
)
# --- Encoding

def yn(val): return 1 if val == "Yes" else 0
def sex_val(val): return 1 if val == "Male" else 0

features = np.array([[
    high_bp,
    high_chol,
    yn(chol_check),
    bmi,
    yn(smoker),
    yn(stroke),
    yn(heart_disease),
    yn(phys_activity),
    yn(fruits),
    yn(veggies),
    yn(alcohol),
    yn(any_healthcare),
    yn(no_doc_cost),
    genhlth,
    menthlth,
    physhlth,
    yn(diffwalk),
    sex_val(sex),
    age,
    education,
    income
]])

# --- Prediction

st.subheader("📊 Prediction")
if st.button("Predict Diabetes Risk"):
    prediction = int(model.predict(features)[0])
    prob = model.predict_proba(features)[0]

    if prediction == 1:
        st.error("🛑 High risk of diabetes.")
    else:
        st.success("✅ Low risk of diabetes.")

    st.markdown(f"**Confidence:** {prob[prediction]*100:.2f}%")
