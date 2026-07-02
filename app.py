# ── Imports ───────────────────────────────────────
import streamlit as st
import pickle
import numpy as np
import google.generativeai as genai

# ── Page configuration ────────────────────────────
st.set_page_config(
    page_title="AI Road Safety Advisor",
    page_icon="🚦",
    layout="centered"
)

# ── Load the trained ML model ─────────────────────
# model.pkl is the saved Random Forest classifier
try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error("Model file not found. Make sure model.pkl is in the project folder.")
    st.stop()

# ── Gemini AI setup ───────────────────────────────
# API key is stored securely in Streamlit secrets
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    available_models = [m.name for m in genai.list_models()
                        if 'generateContent' in m.supported_generation_methods]
    gemini = genai.GenerativeModel(available_models[0])
except Exception as e:
    st.error("Gemini API key error. Check your Streamlit secrets.")
    st.stop()

# ── Custom CSS styling ────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0C447C, #185FA5);
        padding: 24px 28px;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .main-header h1 {
        color: white;
        font-size: 24px;
        font-weight: 600;
        margin: 0;
    }
    .main-header p {
        color: rgba(255,255,255,0.75);
        font-size: 14px;
        margin: 4px 0 0;
    }
    .section-title {
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #888780;
        margin-bottom: 12px;
    }
    .meter-wrap {
        display: flex;
        gap: 6px;
        margin: 8px 0 4px;
    }
    .meter-seg {
        flex: 1;
        height: 8px;
        border-radius: 99px;
    }
    .meter-labels {
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        color: #888780;
        margin-bottom: 16px;
    }
    .result-low {
        background: #EAF3DE;
        border: 0.5px solid #C0DD97;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 12px 0;
    }
    .result-mid {
        background: #FAEEDA;
        border: 0.5px solid #FAC775;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 12px 0;
    }
    .result-high {
        background: #FCEBEB;
        border: 0.5px solid #F7C1C1;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 12px 0;
    }
    .result-low .title  { color: #27500A; font-weight: 600; font-size: 15px; margin: 0 0 4px; }
    .result-mid .title  { color: #633806; font-weight: 600; font-size: 15px; margin: 0 0 4px; }
    .result-high .title { color: #791F1F; font-weight: 600; font-size: 15px; margin: 0 0 4px; }
    .result-low .desc   { color: #3B6D11; font-size: 13px; margin: 0; }
    .result-mid .desc   { color: #854F0B; font-size: 13px; margin: 0; }
    .result-high .desc  { color: #A32D2D; font-size: 13px; margin: 0; }
    .stButton > button {
        background: #0C447C !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background: #185FA5 !important;
    }
    .stSelectbox label {
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #5F5E5A !important;
    }
    .chat-header {
        background: #0C447C;
        color: white;
        padding: 12px 16px;
        border-radius: 10px 10px 0 0;
        font-size: 14px;
        font-weight: 500;
    }
    .about-box {
        background: #F1EFE8;
        border: 0.5px solid #D3D1C7;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 20px;
    }
    .about-box h4 {
        font-size: 13px;
        font-weight: 600;
        color: #444441;
        margin: 0 0 8px;
    }
    .about-box p {
        font-size: 12px;
        color: #5F5E5A;
        margin: 0;
        line-height: 1.6;
    }
    hr { border-color: #D3D1C7 !important; }
</style>
""", unsafe_allow_html=True)

# ── App header ────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🚦 AI Road Safety Advisor</h1>
    <p>Predict accident severity and get AI-powered safety tips</p>
</div>
""", unsafe_allow_html=True)

# ── Model accuracy badge ──────────────────────────
col_a, col_b, col_c = st.columns(3)
col_a.metric("Model Accuracy", "90.58%")
col_b.metric("Dataset", "UK Road Safety")
col_c.metric("Algorithm", "Random Forest")

st.markdown("<br>", unsafe_allow_html=True)

# ── User input section ────────────────────────────
st.markdown('<p class="section-title">Travel conditions</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    # Weather dropdown
    weather = st.selectbox("🌦 Weather Condition", [
        "Fine no high winds", "Raining no high winds",
        "Snowing no high winds", "Fine + high winds",
        "Fog or mist", "Raining + high winds", "Other"
    ])
    # Light condition dropdown
    light = st.selectbox("💡 Light Condition", [
        "Daylight", "Darkness - lights lit",
        "Darkness - no lighting", "Darkness - lights unlit"
    ])
    # Area type dropdown
    area = st.selectbox("🏙 Area Type", ["Urban", "Rural"])

with col2:
    # Road type dropdown
    road = st.selectbox("🛣 Road Type", [
        "Single carriageway", "Dual carriageway",
        "Roundabout", "One way street", "Slip road"
    ])
    # Road surface dropdown
    surface = st.selectbox("🌧 Road Surface", [
        "Dry", "Wet or damp", "Snow",
        "Frost or ice", "Flood over 3cm deep"
    ])

# ── Encode user inputs to numeric values ──────────
# These mappings match the label encoding done during training
weather_map = {"Fine no high winds":1,"Raining no high winds":2,
               "Snowing no high winds":3,"Fine + high winds":4,
               "Fog or mist":5,"Raining + high winds":6,"Other":7}
road_map    = {"Single carriageway":6,"Dual carriageway":2,
               "Roundabout":1,"One way street":3,"Slip road":7}
light_map   = {"Daylight":1,"Darkness - lights lit":4,
               "Darkness - no lighting":6,"Darkness - lights unlit":5}
surface_map = {"Dry":1,"Wet or damp":2,"Snow":4,
               "Frost or ice":3,"Flood over 3cm deep":5}
area_map    = {"Urban":1,"Rural":2}

# Build input array for model
input_data = np.array([[
    weather_map[weather], road_map[road],
    light_map[light], surface_map[surface],
    area_map[area]
]])

# ── Prediction section ────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict = st.button("🔍 Predict Accident Severity")

if predict:
    # Run prediction with loading spinner
    with st.spinner("Analyzing road conditions..."):
        try:
            result = model.predict(input_data)[0]
            # Get prediction confidence percentage
            proba = model.predict_proba(input_data)[0]
            confidence = round(max(proba) * 100, 1)
        except Exception as e:
            st.error("Prediction failed. Please try again.")
            st.stop()

    # Show risk meter and result
    st.markdown('<p class="section-title">Risk assessment</p>', unsafe_allow_html=True)

    if result == 1:
        st.markdown(f"""
        <div class="meter-wrap">
            <div class="meter-seg" style="background:#97C459"></div>
            <div class="meter-seg" style="background:#EF9F27;opacity:.3"></div>
            <div class="meter-seg" style="background:#E24B4A;opacity:.3"></div>
        </div>
        <div class="meter-labels"><span>Low</span><span>Medium</span><span>High</span></div>
        <div class="result-low">
            <p class="title">✅ Low risk — Safe to travel &nbsp;<small style="font-weight:400;font-size:12px">({confidence}% confident)</small></p>
            <p class="desc">Follow speed limits. Stay alert. Wear your seatbelt at all times.</p>
        </div>
        """, unsafe_allow_html=True)

    elif result == 2:
        st.markdown(f"""
        <div class="meter-wrap">
            <div class="meter-seg" style="background:#97C459;opacity:.3"></div>
            <div class="meter-seg" style="background:#EF9F27"></div>
            <div class="meter-seg" style="background:#E24B4A;opacity:.3"></div>
        </div>
        <div class="meter-labels"><span>Low</span><span>Medium</span><span>High</span></div>
        <div class="result-mid">
            <p class="title">⚠️ Medium risk — Drive carefully &nbsp;<small style="font-weight:400;font-size:12px">({confidence}% confident)</small></p>
            <p class="desc">Reduce speed. Maintain safe distance. Avoid phone use while driving.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="meter-wrap">
            <div class="meter-seg" style="background:#97C459;opacity:.3"></div>
            <div class="meter-seg" style="background:#EF9F27;opacity:.3"></div>
            <div class="meter-seg" style="background:#E24B4A"></div>
        </div>
        <div class="meter-labels"><span>Low</span><span>Medium</span><span>High</span></div>
        <div class="result-high">
            <p class="title">🚨 High risk — Avoid travel if possible &nbsp;<small style="font-weight:400;font-size:12px">({confidence}% confident)</small></p>
            <p class="desc">Postpone your journey. If urgent, drive very slowly and stay on main roads only.</p>
        </div>
        """, unsafe_allow_html=True)

# ── Chatbot section ───────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="chat-header">💬 Road Safety AI Chatbot — Ask me anything</div>',
            unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Accept new user message
user_input = st.chat_input("Ask a road safety question...")

if user_input:
    # Display and store user message
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Build conversation history string for context
    history = ""
    for msg in st.session_state.messages[:-1]:
        role = "User" if msg["role"] == "user" else "Bot"
        history += f"{role}: {msg['content']}\n"

    # Get AI response from Gemini
    with st.spinner("Thinking..."):
        prompt = f"""You are a helpful road safety advisor chatbot.
Only answer questions related to road safety, driving tips,
accident prevention, traffic rules, and vehicle safety.
Keep answers short, clear and practical (max 4 lines).
If the question is not about road safety, politely say
you can only help with road safety topics.

Conversation so far:
{history}
User: {user_input}
Bot:"""
        try:
            response = gemini.generate_content(prompt)
            if response and response.text:
                bot_reply = response.text
            else:
                bot_reply = "Sorry, I could not generate a response. Please try asking differently."
        except Exception as e:
            bot_reply = "Sorry, I had trouble answering that. Please try rephrasing your question."

    # Display and store bot reply
    st.chat_message("assistant").write(bot_reply)
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply
    })

# ── About section ─────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="about-box">
    <h4>About this project</h4>
    <p>
        AI Road Safety Advisor is a machine learning powered web application that predicts
        road accident severity based on weather, road, and environmental conditions.
        Built using Random Forest classification trained on the UK Road Safety dataset
        (1.8M records, 90.58% accuracy). Powered by Google Gemini AI for intelligent
        safety recommendations.<br><br>
        <strong>Dataset:</strong> UK Road Safety — data.gov.uk &nbsp;|&nbsp;
        <strong>Model:</strong> Random Forest Classifier &nbsp;|&nbsp;
        <strong>Built with:</strong> Python, Scikit-learn, Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
