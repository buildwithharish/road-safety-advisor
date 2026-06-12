import streamlit as st
import pickle
import numpy as np
import google.generativeai as genai

# Load the saved model
model = pickle.load(open('model.pkl', 'rb'))

# Gemini setup
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini = genai.GenerativeModel("gemini-1.5-flash")

# App title
st.title("🚦 AI Road Safety Advisor")
st.write("Fill in the details below to predict accident severity.")
st.divider()

# ── User Inputs ──────────────────────────────────
weather = st.selectbox("🌦 Weather Condition", [
    "Fine no high winds", "Raining no high winds",
    "Snowing no high winds", "Fine + high winds",
    "Fog or mist", "Raining + high winds", "Other"
])
road = st.selectbox("🛣 Road Type", [
    "Single carriageway", "Dual carriageway",
    "Roundabout", "One way street", "Slip road"
])
light = st.selectbox("💡 Light Condition", [
    "Daylight", "Darkness - lights lit",
    "Darkness - no lighting", "Darkness - lights unlit"
])
surface = st.selectbox("🌧 Road Surface", [
    "Dry", "Wet or damp", "Snow",
    "Frost or ice", "Flood over 3cm deep"
])
area = st.selectbox("🏙 Area Type", ["Urban", "Rural"])

# ── Encode inputs ─────────────────────────────────
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

# ── Predict button ─────────────────────────────────
st.divider()
if st.button("🔍 Predict Accident Severity"):
    result = model.predict(input_data)[0]
    if result == 1:
        st.success("✅ LOW RISK — Safe to travel")
        st.info("💡 Follow speed limits. Stay alert.")
    elif result == 2:
        st.warning("⚠️ MEDIUM RISK — Drive carefully")
        st.info("💡 Reduce speed. Keep safe distance.")
    else:
        st.error("🚨 HIGH RISK — Avoid travel if possible")
        st.info("💡 Postpone journey. Drive very slowly if urgent.")

# ── Gemini AI Chatbot ──────────────────────────────
st.divider()
st.subheader("💬 Road Safety AI Chatbot")
st.write("Ask me anything about road safety!")

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
user_input = st.chat_input("Ask a road safety question...")

if user_input:
    # Show user message
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Build conversation history for Gemini
    history = ""
    for msg in st.session_state.messages[:-1]:
        role = "User" if msg["role"] == "user" else "Bot"
        history += f"{role}: {msg['content']}\n"

    # Get Gemini response
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

        response = gemini.generate_content(prompt)
        bot_reply = response.text

    # Show and store bot reply
    st.chat_message("assistant").write(bot_reply)
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply
    })
