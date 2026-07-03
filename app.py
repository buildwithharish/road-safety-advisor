# ── Imports ───────────────────────────────────────
import streamlit as st
import pickle
import numpy as np
import google.generativeai as genai
import requests
from datetime import datetime
from streamlit_js_eval import get_geolocation

# ── Page config ───────────────────────────────────
st.set_page_config(
    page_title="AI Road Safety Advisor",
    page_icon="🚦",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────
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
    .info-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin-bottom: 16px;
    }
    .info-card {
        background: #F1EFE8;
        border: 0.5px solid #D3D1C7;
        border-radius: 10px;
        padding: 12px 14px;
        text-align: center;
    }
    .info-card .label {
        font-size: 11px;
        color: #888780;
        margin: 0 0 4px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .info-card .value {
        font-size: 14px;
        font-weight: 500;
        color: #1A1917;
        margin: 0;
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

# ── Load ML model ─────────────────────────────────
try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception:
    st.error("Model file not found. Make sure model.pkl is in the project folder.")
    st.stop()

# ── Gemini AI setup ───────────────────────────────
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    available_models = [m.name for m in genai.list_models()
                        if 'generateContent' in m.supported_generation_methods]
    gemini = genai.GenerativeModel(available_models[0])
except Exception:
    st.error("Gemini API key error. Check your Streamlit secrets.")
    st.stop()

# ── Helper functions ──────────────────────────────

def get_weather(lat, lon):
    """Fetch weather conditions from OpenWeatherMap"""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={st.secrets['OPENWEATHER_API_KEY']}"
        r = requests.get(url).json()
        weather_main = r['weather'][0]['main']
        weather_desc = r['weather'][0]['description']
        temp = round(r['main']['temp'] - 273.15, 1)

        # Map to our model's weather categories
        if 'Rain' in weather_main:
            weather_code = 2
            weather_label = "Raining no high winds"
        elif 'Snow' in weather_main:
            weather_code = 3
            weather_label = "Snowing no high winds"
        elif 'Fog' in weather_main or 'Mist' in weather_main:
            weather_code = 5
            weather_label = "Fog or mist"
        elif 'Wind' in weather_desc or 'Squall' in weather_main:
            weather_code = 4
            weather_label = "Fine + high winds"
        else:
            weather_code = 1
            weather_label = "Fine no high winds"

        # Surface condition based on weather
        if 'Rain' in weather_main:
            surface_code = 2
            surface_label = "Wet or damp"
        elif 'Snow' in weather_main:
            surface_code = 4
            surface_label = "Snow"
        else:
            surface_code = 1
            surface_label = "Dry"

        return weather_code, weather_label, surface_code, surface_label, temp, weather_desc

    except Exception:
        return 1, "Fine no high winds", 1, "Dry", 0, "Unknown"


def get_location_info(lat, lon):
    """Get area type and road type from OpenStreetMap"""
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        headers = {"User-Agent": "RoadSafetyAdvisor/1.0"}
        r = requests.get(url, headers=headers).json()

        address = r.get('address', {})
        location_name = r.get('display_name', 'Unknown location')

        # Determine urban or rural
        if any(k in address for k in ['city', 'town', 'suburb', 'neighbourhood']):
            area_code = 1
            area_label = "Urban"
        else:
            area_code = 2
            area_label = "Rural"

        # Determine road type
        road_type = r.get('type', '')
        if road_type in ['motorway', 'trunk']:
            road_code = 2
            road_label = "Dual carriageway"
        elif road_type in ['primary', 'secondary']:
            road_code = 6
            road_label = "Single carriageway"
        elif road_type == 'roundabout':
            road_code = 1
            road_label = "Roundabout"
        else:
            road_code = 6
            road_label = "Single carriageway"

        return area_code, area_label, road_code, road_label, location_name

    except Exception:
        return 1, "Urban", 6, "Single carriageway", "Unknown location"


def get_light_condition():
    """Auto-detect light condition based on current hour"""
    hour = datetime.now().hour
    if 6 <= hour < 19:
        return 1, "Daylight"
    else:
        return 4, "Darkness - lights lit"


def get_destination_info(destination):
    """Geocode destination using Google Maps API"""
    try:
        url = f"https://maps.googleapis.com/maps/api/geocode/json?address={destination}&key={st.secrets['GOOGLE_MAPS_API_KEY']}"
        r = requests.get(url).json()
        if r['status'] == 'OK':
            result = r['results'][0]
            dest_name = result['formatted_address']
            dest_lat = result['geometry']['location']['lat']
            dest_lng = result['geometry']['location']['lng']
            return dest_name, dest_lat, dest_lng
        return destination, None, None
    except Exception:
        return destination, None, None


# ── App header ────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🚦 AI Road Safety Advisor</h1>
    <p>Auto-detects your location and conditions — just enter your destination</p>
</div>
""", unsafe_allow_html=True)

# ── Model metrics ─────────────────────────────────
col_a, col_b, col_c = st.columns(3)
col_a.metric("Model Accuracy", "90.58%")
col_b.metric("Dataset", "UK Road Safety")
col_c.metric("Algorithm", "Random Forest")

st.markdown("<br>", unsafe_allow_html=True)

# ── GPS Location fetch ────────────────────────────
st.markdown('<p class="section-title">📍 Your location</p>',
            unsafe_allow_html=True)
st.info("Click below to fetch your current GPS location automatically.")

location = get_geolocation()

if location:
    lat = location['coords']['latitude']
    lon = location['coords']['longitude']

    # Fetch all conditions automatically
    with st.spinner("Fetching road conditions automatically..."):
        weather_code, weather_label, surface_code, surface_label, temp, weather_desc = get_weather(lat, lon)
        area_code, area_label, road_code, road_label, location_name = get_location_info(lat, lon)
        light_code, light_label = get_light_condition()

    st.success(f"📍 Location detected: {location_name[:60]}...")

    # Show auto-fetched conditions
    st.markdown('<p class="section-title">Auto-detected conditions</p>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-grid">
        <div class="info-card">
            <p class="label">🌦 Weather</p>
            <p class="value">{weather_label}</p>
        </div>
        <div class="info-card">
            <p class="label">🌡 Temperature</p>
            <p class="value">{temp}°C</p>
        </div>
        <div class="info-card">
            <p class="label">🌧 Road Surface</p>
            <p class="value">{surface_label}</p>
        </div>
        <div class="info-card">
            <p class="label">💡 Light</p>
            <p class="value">{light_label}</p>
        </div>
        <div class="info-card">
            <p class="label">🛣 Road Type</p>
            <p class="value">{road_label}</p>
        </div>
        <div class="info-card">
            <p class="label">🏙 Area</p>
            <p class="value">{area_label}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Destination input ─────────────────────────
    st.markdown('<p class="section-title">🗺 Your destination</p>',
                unsafe_allow_html=True)
    destination = st.text_input("Enter destination",
                                 placeholder="e.g. Chennai Central Railway Station")

    # ── Predict button ────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Predict Accident Severity"):

        # Resolve destination
        dest_name = destination
        if destination:
            with st.spinner("Resolving destination..."):
                dest_name, dest_lat, dest_lng = get_destination_info(destination)

        # Build input and predict
        input_data = np.array([[
            weather_code, road_code,
            light_code, surface_code,
            area_code
        ]])

        with st.spinner("Analyzing road conditions..."):
            try:
                result = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0]
                confidence = round(max(proba) * 100, 1)
            except Exception:
                st.error("Prediction failed. Please try again.")
                st.stop()

        # Show route summary
        st.markdown('<p class="section-title">Route summary</p>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-grid">
            <div
