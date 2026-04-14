import streamlit as st
import numpy as np
import pickle
import base64

with open('disease_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load heart image as base64
with open('heart.png', 'rb') as f:
    heart_base64 = base64.b64encode(f.read()).decode()

st.set_page_config(page_title="MediPredict", page_icon="🏥", layout="centered")

st.markdown("""
<style>
.stApp {
    background: 
        radial-gradient(ellipse at 20% 30%, rgba(0, 200, 100, 0.12) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 70%, rgba(66, 165, 245, 0.10) 0%, transparent 50%),
        linear-gradient(135deg, #0a1f0a 0%, #0d2b1a 50%, #0a1f0a 100%);
}
.badge {
    background: rgba(0, 200, 100, 0.15);
    border: 1px solid #00C864;
    border-radius: 50px;
    padding: 6px 20px;
    color: #00C864;
    font-size: 13px;
    font-weight: bold;
    letter-spacing: 2px;
    text-align: center;
    display: inline-block;
    margin-bottom: 20px;
}
.main-title {
    font-size: 62px;
    font-weight: 900;
    color: white;
    text-align: center;
    line-height: 1.2;
    margin-top: 40px;
    margin-bottom: 10px;
}
.highlight {
    background: linear-gradient(90deg, #00C864, #00E676);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    font-size: 17px;
    color: #cccccc;
    text-align: center;
    margin-bottom: 40px;
    line-height: 1.7;
}
.predict-title {
    font-size: 48px;
    font-weight: 900;
    text-align: center;
    margin-bottom: 5px;
    margin-top: 10px;
    background: linear-gradient(90deg, #00C864, #42A5F5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.predict-subtitle {
    font-size: 15px;
    color: #aaaaaa;
    text-align: center;
    margin-bottom: 30px;
}
div[data-testid="stNumberInput"] > div {
    background-color: black !important;
    border: 1.5px solid #00C864 !important;
    border-radius: 10px !important;
    box-shadow: 0 0 10px rgba(0,200,100,0.25) !important;
    overflow: hidden !important;
}
div[data-testid="stNumberInput"] input {
    background-color: black !important;
    color: white !important;
    border: none !important;
    box-shadow: none !important;
}
div[data-testid="stNumberInput"] button {
    background-color: #001a00 !important;
    color: #00C864 !important;
    border: none !important;
    border-left: 1px solid #00C864 !important;
    border-radius: 0 !important;
    margin: 0 !important;
}
div[data-baseweb="select"] {
    background-color: black !important;
    border: 1.5px solid #00C864 !important;
    border-radius: 10px !important;
    box-shadow: 0 0 10px rgba(0,200,100,0.25) !important;
    overflow: hidden !important;
}
div[data-baseweb="select"] > div {
    background-color: black !important;
    color: white !important;
    border: none !important;
    box-shadow: none !important;
}
div[data-baseweb="select"] span {
    color: white !important;
}
label, p {
    color: #00E676 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}
h1, h2, h3 {
    color: white !important;
}
.stButton > button {
    background: linear-gradient(90deg, #00C864, #00897B) !important;
    font-size: 18px !important;
    font-weight: bold !important;
    border-radius: 50px !important;
    padding: 12px 20px !important;
    border: none !important;
    box-shadow: 0px 4px 20px rgba(0, 200, 100, 0.4) !important;
    letter-spacing: 1px !important; 
    white-space: nowrap !important;   /* 🔥 VERY IMPORTANT */
}
.stButton > button * {
    color: black !important;
    fill: black !important;
}
.stButton > button:hover {
    transform: scale(1.05);
    transition: 0.2s;
}
.result-box-low {
    background: linear-gradient(135deg, #0a3d1f, #0d5c2a);
    border: 2px solid #00C864;
    border-radius: 20px;
    padding: 35px;
    text-align: center;
    box-shadow: 0px 0px 40px rgba(0, 200, 100, 0.4);
    margin-top: 20px;
}
.result-box-high {
    background: linear-gradient(135deg, #3d0a0a, #5c0d0d);
    border: 2px solid #FF5252;
    border-radius: 20px;
    padding: 35px;
    text-align: center;
    box-shadow: 0px 0px 40px rgba(255, 82, 82, 0.4);
    margin-top: 20px;
}
.result-title {
    font-size: 36px;
    font-weight: 900;
    margin-bottom: 10px;
    color: white;
}
.result-advice {
    font-size: 15px;
    color: #cccccc;
    margin-top: 15px;
    line-height: 1.8;
}
.thankyou-box {
    background: linear-gradient(135deg, #0a3d1f, #0d2b1a);
    border: 2px solid #00C864;
    border-radius: 20px;
    padding: 40px;
    text-align: center;
    box-shadow: 0px 0px 50px rgba(0, 200, 100, 0.3);
    margin-top: 20px;
}
/* 🔥 FORCE GREEN FOR PRIMARY BUTTONS */
button[data-testid="baseButton-primary"] {
    background: linear-gradient(90deg, #00C864, #00897B) !important;
    color: black !important;
    border: none !important;
    box-shadow: 0px 4px 20px rgba(0, 200, 100, 0.4) !important;
}
</style>
""", unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state.page = 'welcome'
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None

# WELCOME PAGE
if st.session_state.page == 'welcome':
    st.markdown('<div style="text-align:center"><span class="badge">🏥 MEDIPREDICT — HEALTH AI SYSTEM</span></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="main-title">
        Early Disease<br>Detection with<br>
        <span class="highlight">AI Precision</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="subtitle">
        A smart medical tool that analyzes your health parameters<br>
        and predicts diabetes risk using Neural Network technology.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🩺 Check My Health Now", use_container_width=True):
            st.session_state.page = 'predict'
            st.rerun()

# PREDICT PAGE
elif st.session_state.page == 'predict':
   

    st.markdown("""
    <style>

    div[data-testid="stButton"] button {
        background: linear-gradient(90deg, #00C864, #00897B) !important;
        color: black !important;
        border: none !important;
        border-radius: 50px !important;
        box-shadow: 0px 4px 20px rgba(0, 200, 100, 0.4) !important;
    }

    </style>
    """, unsafe_allow_html=True)

    # ✅ FIXED ALIGNMENT
    if st.button("← Back"):
        st.session_state.page = 'welcome'
        st.rerun()
    st.markdown('<div class="predict-title">Diabetes Risk Analyser</div>', unsafe_allow_html=True)
    st.markdown('<div class="predict-subtitle">Enter your health details below for an instant AI-powered diagnosis</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        glucose = st.number_input("Glucose (Blood Sugar)", min_value=0, max_value=300, value=100)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
        insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
    with col2:
        age = st.number_input("Age", min_value=1, max_value=120, value=25)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        if gender == "Female":
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
        else:
            pregnancies = 0

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyse = st.button("🩺 Analyse My Health",use_container_width=True)

    if analyse:
        input_data = np.array([[pregnancies, glucose, blood_pressure, 0, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        st.session_state.prediction_result = int(prediction)
        st.session_state.confidence = probability.tolist()
        st.session_state.page = 'result'
        st.rerun()

# RESULT PAGE
elif st.session_state.page == 'result':
    st.markdown("""
    <style>
    
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("← Back"):
        st.session_state.page = 'predict'
        st.rerun()

    prediction = st.session_state.prediction_result
    probability = st.session_state.confidence

    st.markdown("<br>", unsafe_allow_html=True)
    if prediction == 1:
        st.markdown(
            "<div class='result-box-high'>"
            "<div class='result-title'>⚠️WARNING - HIGH RISK DETECTED</div>"
            "<div style='font-size:44px;font-weight:900;color:#FF5252;margin:10px 0;'>"
            + str(round(probability[1]*100, 1)) +
            "%</div>"
            "<div style='color:#FF8A80;font-size:18px;margin-bottom:10px;'>Confidence of Diabetes</div>"
            "<div class='result-advice'>"
            "Please consult a doctor immediately🏥 <br>"
            "Early treatment can prevent serious complications ⚕️<br>"
            "Follow a healthy diet and medication plan💊"
            "</div></div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='result-box-low'>"
            "<div class='result-title'>LOW RISK</div>"
            "<div style='font-size:44px;font-weight:900;color:#00E676;margin:10px 0;'>"
            + str(round(probability[0]*100, 1)) +
            "%</div>"
            "<div style='color:#69F0AE;font-size:18px;margin-bottom:10px;'>Confidence of No Diabetes</div>"
            "<div class='result-advice'>"
            "You are healthy! Great job💚<br>"
            "Maintain a balanced diet and stay active🥗<br>"
            "Exercise regularly and stay hydrated 🏃 "
            "</div></div>",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Continue",use_container_width=True, type="primary"):
            st.session_state.page = 'thankyou'
            st.rerun()

# THANK YOU PAGE
elif st.session_state.page == 'thankyou':
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div class='thankyou-box'>"
        "<img src='data:image/png;base64," + heart_base64 + "' width='150' style='margin-bottom:20px;border-radius:10px;'/>"
        "<div style='font-size:42px;font-weight:900;color:white;margin-bottom:10px;'>Thank You!</div>"
        "<div style='font-size:20px;color:#00C864;font-weight:bold;margin-bottom:20px;'>"
        "MediPredict - Your Health AI Companion</div>"
        "<div style='font-size:16px;color:#cccccc;line-height:1.8;margin-bottom:20px;'>"
        "Your health is our priority.<br>"
        "Stay healthy, stay happy, stay strong!<br><br>"
        "Health Tip: Drink at least 8 glasses of water daily "
        "and exercise for 30 minutes every day!"
        "</div>"
        "<div style='font-size:14px;color:#aaaaaa;'>- MediPredict Team</div>"
        "</div>",
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Check Again", use_container_width=True,type="primary"):
            st.session_state.page = 'welcome'
            st.rerun()