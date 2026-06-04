import streamlit as st
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open('model.pkl', 'rb'))

# App title
st.title("🚦 AI Road Safety Advisor")
st.write("Fill in the details below to predict accident severity.")
st.divider()

# ── User Inputs ──────────────────────────────────
weather = st.selectbox("🌦 Weather Condition", [
    "Fine no high winds",
    "Raining no high winds",
    "Snowing no high winds",
    "Fine + high winds",
    "Fog or mist",
    "Raining + high winds",
    "Other"
])

road = st.selectbox("🛣 Road Type", [
    "Single carriageway",
    "Dual carriageway",
    "Roundabout",
    "One way street",
    "Slip road"
])

light = st.selectbox("💡 Light Condition", [
    "Daylight",
    "Darkness - lights lit",
    "Darkness - no lighting",
    "Darkness - lights unlit"
])

surface = st.selectbox("🌧 Road Surface", [
    "Dry",
    "Wet or damp",
    "Snow",
    "Frost or ice",
    "Flood over 3cm deep"
])

area = st.selectbox("🏙 Area Type", [
    "Urban",
    "Rural"
])

# ── Encode inputs to numbers ─────────────────────
weather_map = {"Fine no high winds":1,"Raining no high winds":2,
               "Snowing no high winds":3,"Fine + high winds":4,
               "Fog or mist":5,"Raining + high winds":6,"Other":7}

road_map = {"Single carriageway":6,"Dual carriageway":2,
            "Roundabout":1,"One way street":3,"Slip road":7}

light_map = {"Daylight":1,"Darkness - lights lit":4,
             "Darkness - no lighting":6,"Darkness - lights unlit":5}

surface_map = {"Dry":1,"Wet or damp":2,"Snow":4,
               "Frost or ice":3,"Flood over 3cm deep":5}

area_map = {"Urban":1, "Rural":2}

input_data = np.array([[
    weather_map[weather],
    road_map[road],
    light_map[light],
    surface_map[surface],
    area_map[area]
]])

# ── Predict button ────────────────────────────────
st.divider()
if st.button("🔍 Predict Accident Severity"):
    result = model.predict(input_data)[0]

    if result == 1:
        st.success("✅ LOW RISK — Safe to travel")
        st.info("💡 Tips: Follow speed limits. Stay alert.")

    elif result == 2:
        st.warning("⚠️ MEDIUM RISK — Drive carefully")
        st.info("💡 Tips: Reduce speed. Keep safe distance. Avoid distractions.")

    else:
        st.error("🚨 HIGH RISK — Avoid travel if possible")
        st.info("💡 Tips: Postpone journey. If urgent, drive very slowly and stay on main roads.")