import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("diabetes_model_rf.pkl")

# --- Page config ---
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Risk Prediction")

st.markdown("""
Welcome to the **Diabetes Risk Predictor**! 🧠🩺  
Enter your health details to estimate your risk of diabetes.
""")

with st.sidebar:
    st.header("⚙️ Doctor Panel")
    show_info = st.checkbox("Show medical guidance", value=True)

# --- Section Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🩸 Vitals", "🏃 Lifestyle", "🏥 Access", "🧠 Mental Health", "🧬 Demographics"])

with tab1:
    st.subheader("1. Blood Pressure & Cholesterol")

    systolic = st.number_input("Systolic BP (mmHg)", 70, 250, 120)
    diastolic = st.number_input("Diastolic BP (mmHg)", 40, 140, 80)
    pulse = st.number_input("Resting Pulse (bpm)", 40, 180, 75)

    if show_info:
        st.markdown("""
        - **Normal BP**: < 120/80 mmHg  
        - **Elevated**: 120–129/<80  
        - **High**: ≥130 or ≥80
        """)

    high_bp = 1 if systolic >= 130 or diastolic >= 80 else 0

    cholesterol = st.number_input("Total Cholesterol (mg/dL)", 100, 400, 180)

    if show_info:
        st.markdown("""
        - **Normal**: < 200  
        - **Borderline**: 200–239  
        - **High**: ≥ 240
        """)

    high_chol = 1 if cholesterol >= 200 else 0
    chol_check = st.radio("Had cholesterol checked in past 5 years?", ["No", "Yes"])

    st.markdown("### ⚖️ Calculate Your BMI")
    height_cm = st.number_input("Height (cm)", 100, 250, 170)
    weight_kg = st.number_input("Weight (kg)", 30, 200, 65)
    bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
    st.markdown(f"**Calculated BMI:** `{bmi}`")

    bmi_status = ""
    if bmi < 18.5:
        bmi_status = "Underweight"
        st.warning("⚠️ Underweight")
    elif 18.5 <= bmi < 25:
        bmi_status = "Normal weight"
        st.success("✅ Normal weight")
    elif 25 <= bmi < 30:
        bmi_status = "Overweight"
        st.info("ℹ️ Overweight")
    else:
        bmi_status = "Obese"
        st.error("🛑 Obese")

with tab2:
    st.subheader("2. Lifestyle")
    smoker = st.radio("Do you smoke?", ["No", "Yes"])
    stroke = st.radio("History of stroke?", ["No", "Yes"])
    heart_disease = st.radio("Heart disease history?", ["No", "Yes"])
    phys_activity = st.radio("Physical activity?", ["No", "Yes"])
    fruits = st.radio("Eat fruits daily?", ["No", "Yes"])
    veggies = st.radio("Eat vegetables daily?", ["No", "Yes"])
    alcohol = st.radio("Heavy alcohol consumption?", ["No", "Yes"])

with tab3:
    st.subheader("3. Healthcare Access")
    any_healthcare = st.radio("Healthcare coverage?", ["No", "Yes"])
    no_doc_cost = st.radio("Couldn't see doctor due to cost?", ["No", "Yes"])

with tab4:
    st.subheader("4. General & Mental Health")

    genhlth = st.slider("General health (1=Excellent, 5=Poor)", 1, 5, 3)

    mental_illness = st.radio("Are you suffering from any mental illness?", ["No", "Yes"])
    menthlth = 5 if mental_illness == "Yes" else 0
    if mental_illness == "Yes":
        st.text_input("Optional: Name or type of mental illness")

    physical_disability = st.radio("Do you have any physical disability?", ["No", "Yes"])
    physhlth = 5 if physical_disability == "Yes" else 0
    if physical_disability == "Yes":
        st.text_input("Optional: Name or type of physical disability")

    diffwalk = st.radio("Difficulty walking?", ["No", "Yes"])

with tab5:
    st.subheader("5. Demographics")
    sex = st.radio("Biological Sex", ["Female", "Male"])
    age_real = st.slider("Your age", 13, 100, 30)

    # Convert to BRFSS age group (1 = 18–24, ..., 13 = 80+)
    if age_real < 18:
        age = 1
    elif age_real < 25:
        age = 1
    elif age_real < 30:
        age = 2
    elif age_real < 35:
        age = 3
    elif age_real < 40:
        age = 4
    elif age_real < 45:
        age = 5
    elif age_real < 50:
        age = 6
    elif age_real < 55:
        age = 7
    elif age_real < 60:
        age = 8
    elif age_real < 65:
        age = 9
    elif age_real < 70:
        age = 10
    elif age_real < 75:
        age = 11
    elif age_real < 80:
        age = 12
    else:
        age = 13

    education = st.slider("Education level (1=None, 6=College)", 1, 6, 4)
    income = st.slider("Income level (1=<10k, 8=>75k)", 1, 8, 5)

# --- Encoding ---
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

# --- Prediction ---
st.subheader("📊 Prediction")
if st.button("Predict Diabetes Risk"):
    prediction = int(model.predict(features)[0])
    prob = model.predict_proba(features)[0]

    if prediction == 1:
        st.error("🛑 High risk of diabetes.")
    else:
        st.success("✅ Low risk of diabetes.")

    st.markdown(f"**Confidence:** `{prob[prediction]*100:.2f}%`")

    # Visualization
    st.markdown("### 🧠 Risk Breakdown")
    fig, ax = plt.subplots()
    ax.pie([prob[1], prob[0]], labels=["Diabetes Risk", "Low Risk"], autopct="%.1f%%", colors=["#FF4B4B", "#4CAF50"])
    st.pyplot(fig)

    st.markdown(f"""
    #### 📝 Summary
    - **Risk**: {'🛑 High' if prediction == 1 else '✅ Low'}
    - **BMI**: `{bmi}` ({bmi_status})
    - **BP**: {'High' if high_bp else 'Normal'}
    - **Cholesterol**: {'High' if high_chol else 'Normal'}
    """)
