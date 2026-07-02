import streamlit as st
import pickle
import numpy as np
import google.generativeai as genai

# ── Page config ───────────────────────────────────
st.set_page_config(
    page_title="AI Road Safety Advisor",
    page_icon="🚦",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────
st.markdown("""
<style>
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0C447C, #185FA5);
        padding: 24px 28px;
        border-radius: 12px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 16px;
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
        margin: 0;
    }

    /* Section titles */
    .section-title {
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #888780;
        margin-bottom: 12px;
    }

    /* Risk meter */
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

    /* Result cards */
    .result-low {
        background: #EAF3DE;
        border: 0.5px solid #C0DD97;
        border-radius: 10px;
        padding: 14px 18px;
        display: flex;
        align-items: center;
        gap: 14px;
        margin: 12px 0;
    }
    .result-mid {
        background: #FAEEDA;
        border: 0.5px solid #FAC775;
        border-radius: 10px;
        padding: 14px 18px;
        display: flex;
        align-items: center;
        gap: 14px;
        margin: 12px 0;
    }
    .result-high {
        background: #FCEBEB;
        border: 0.5px solid #F7C1C1;
        border-radius: 10px;
        padding: 14px 18px;
        display: flex;
        align-items: center;
        gap: 14px;
        margin: 12px 0;
    }
    .result-low .title  { color: #27500A; font-weight: 600; font-size: 15px; }
    .result-mid .title  { color: #633806; font-weight: 600; font-size: 15px; }
    .result-high .title { color: #791F1F; font-weight: 600; font-size: 15px; }
    .result-low .desc   { color: #3B6D11; font-size: 13px; }
    .result-mid .desc   { color: #854F0B; font-size: 13px; }
    .result-high .desc  { color: #A32D2D; font-size: 13px; }

    /* Predict button */
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

    /* Selectbox label */
    .stSelectbox label {
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #5F5E5A !important;
    }

    /* Chat section */
    .chat-header {
        background: #0C447C;
        color: white;
        padding: 12px 16px;
        border-radius: 10px 10px 0 0;
        font-size: 14px;
        font-weight: 500;
    }

    /* Hide debug line */
    .debug { display: none; }

    /* Divider color */
    hr { border-color: #D3D1C7 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────
model = pickle.load(open('model.pkl', 'rb'))

# ── Gemini setup ──────────────────────────────────
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
available_models = [m.name for m in genai.list_models()
                    if 'generateContent' in m.supported_generation_methods]
gemini = genai.GenerativeModel(available_models[0])

# ── Header ────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div>
        <h1>🚦 AI Road Safety Advisor</h1>
        <p>Predict accident severity and get AI-powered safety tips</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────
st.markdown('<p class="section-title">Travel conditions</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    weather = st.selectbox("🌦 Weather Condition", [
        "Fine no high winds", "Raining no high winds",
        "Snowing no high winds", "Fine + high winds",
        "Fog or mist", "Raining + high winds", "Other"
    ])
    light = st.selectbox("💡 Light Condition", [
        "Daylight", "Darkness - lights lit",
        "Darkness - no lighting", "Darkness - lights unlit"
    ])
    area = st.selectbox("🏙 Area Type", ["Urban", "Rural"])

with col2:
    road = st.selectbox("🛣 Road Type", [
        "Single carriageway", "Dual carriageway",
        "Roundabout", "One way street", "Slip road"
    ])
    surface = st.selectbox("🌧 Road Surface", [
        "Dry", "Wet or damp", "Snow",
        "Frost or ice", "Flood over 3cm deep"
    ])

# ── Encode ────────────────────────────────────────
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

input_data = np.array([[
    weather_map[weather], road_map[road],
    light_map[light], surface_map[surface],
    area_map[area]
]])

# ── Predict ───────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict = st.button("🔍 Predict Accident Severity")

if predict:
    result = model.predict(input_data)[0]

    st.markdown('<p class="section-title">Risk meter</p>', unsafe_allow_html=True)

    if result == 1:
        st.markdown("""
        <div class="meter-wrap">
            <div class="meter-seg" style="background:#97C459"></div>
            <div class="meter-seg" style="background:#EF9F27;opacity:.3"></div>
            <div class="meter-seg" style="background:#E24B4A;opacity:.3"></div>
        </div>
        <div class="meter-labels"><span>Low</span><span>Medium</span><span>High</span></div>
        <div class="result-low">
            <div>
                <p class="title">✅ Low risk — Safe to travel</p>
                <p class="desc">Follow speed limits. Stay alert. Wear seatbelt.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif result == 2:
        st.markdown("""
        <div class="meter-wrap">
            <div class="meter-seg" style="background:#97C459;opacity:.3"></div>
            <div class="meter-seg" style="background:#EF9F27"></div>
            <div class="meter-seg" style="background:#E24B4A;opacity:.3"></div>
        </div>
        <div class="meter-labels"><span>Low</span><span>Medium</span><span>High</span></div>
        <div class="result-mid">
            <div>
                <p class="title">⚠️ Medium risk — Drive carefully</p>
                <p class="desc">Reduce speed. Keep safe distance. Avoid distractions.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="meter-wrap">
            <div class="meter-seg" style="background:#97C459;opacity:.3"></div>
            <div class="meter-seg" style="background:#EF9F27;opacity:.3"></div>
            <div class="meter-seg" style="background:#E24B4A"></div>
        </div>
        <div class="meter-labels"><span>Low</span><span>Medium</span><span>High</span></div>
        <div class="result-high">
            <div>
                <p class="title">🚨 High risk — Avoid travel if possible</p>
                <p class="desc">Postpone journey. If urgent, drive very slowly and stay on main roads.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Chatbot ───────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="chat-header">💬 Road Safety AI Chatbot — Ask me anything</div>',
            unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Ask a road safety question...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    history = ""
    for msg in st.session_state.messages[:-1]:
        role = "User" if msg["role"] == "user" else "Bot"
        history += f"{role}: {msg['content']}\n"

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

    st.chat_message("assistant").write(bot_reply)
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply
    })
