# ── Imports ───────────────────────────────────────
import streamlit as st
import pickle
import numpy as np
import google.generativeai as genai
import requests
import folium
from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation
from datetime import datetime
from timezonefinder import TimezoneFinder
import pytz
import gspread
from google.oauth2.service_account import Credentials
import json
import pandas as pd

# ── Page config ───────────────────────────────────
st.set_page_config(
    page_title="AI Road Safety Advisor",
    page_icon="🚦",
    layout="wide"
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
        font-size: 26px;
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
        margin-top: 16px;
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
    .feature-card {
        background: #F8F7F2;
        border: 0.5px solid #D3D1C7;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .feature-title {
        font-size: 13px;
        font-weight: 600;
        color: #1A1917;
        margin: 0 0 4px;
    }
    .feature-desc {
        font-size: 12px;
        color: #5F5E5A;
        margin: 0;
        line-height: 1.5;
    }
    .meter-wrap {
        display: flex;
        gap: 6px;
        margin: 8px 0 4px;
    }
    .meter-seg {
        flex: 1;
        height: 10px;
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
    .risk-score-box {
        background: linear-gradient(135deg, #0C447C, #185FA5);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
        margin-bottom: 16px;
    }
    .risk-score-num {
        font-size: 48px;
        font-weight: 700;
        margin: 0;
        line-height: 1;
    }
    .risk-score-label {
        font-size: 13px;
        opacity: 0.8;
        margin: 4px 0 0;
    }
    .profile-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 16px 20px;
        color: white;
        margin-bottom: 16px;
    }
    .profile-box h4 {
        font-size: 14px;
        font-weight: 600;
        margin: 0 0 8px;
        color: white;
    }
    .profile-box p {
        font-size: 12px;
        opacity: 0.8;
        margin: 0 0 4px;
        color: white;
    }
    .forecast-card {
        background: var(--color-background-primary);
        border: 0.5px solid var(--color-border-tertiary);
        border-radius: 8px;
        padding: 10px 12px;
        text-align: center;
        margin-bottom: 6px;
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

# ── Google Sheets setup ───────────────────────────
def get_sheet():
    """Connect to Google Sheets using service account credentials"""
    try:
        creds_dict = json.loads(st.secrets["GCP_CREDENTIALS"])
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(
            creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(st.secrets["SHEET_ID"]).sheet1
        return sheet
    except Exception as e:
        return None


def save_trip_to_sheet(sheet, data):
    """Save a trip record to Google Sheets"""
    try:
        # Add header if sheet is empty
        if sheet.row_count <= 1 and not sheet.get_all_values():
            headers = [
                "Timestamp", "Location", "Destination",
                "Weather", "Temperature", "Wind Speed",
                "Light", "Surface", "Area", "Road Type",
                "Risk Score", "Prediction", "Confidence",
                "Distance (km)", "Duration (mins)",
                "Hours Driving", "Feedback"
            ]
            sheet.append_row(headers)
        sheet.append_row(data)
        return True
    except Exception:
        return False


def get_trip_history(sheet):
    """Get all trips from Google Sheets"""
    try:
        records = sheet.get_all_records()
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()


def get_weather_forecast(lat, lon):
    """Get 24-hour weather forecast from OpenWeatherMap"""
    try:
        url = (f"https://api.openweathermap.org/data/2.5/forecast"
               f"?lat={lat}&lon={lon}"
               f"&appid={st.secrets['OPENWEATHER_API_KEY']}&cnt=8")
        r = requests.get(url, timeout=5).json()
        forecasts = []
        for item in r.get('list', [])[:8]:
            dt = datetime.fromtimestamp(item['dt'])
            weather_main = item['weather'][0]['main']
            temp = round(item['main']['temp'] - 273.15, 1)
            wind = round(item['wind']['speed'] * 3.6, 1)

            if 'Rain' in weather_main:
                w_code, s_code = 2, 2
                risk_add = 45
            elif 'Snow' in weather_main:
                w_code, s_code = 3, 4
                risk_add = 65
            elif 'Fog' in weather_main or 'Mist' in weather_main:
                w_code, s_code = 5, 1
                risk_add = 50
            elif wind > 50:
                w_code, s_code = 4, 1
                risk_add = 35
            else:
                w_code, s_code = 1, 1
                risk_add = 10

            hour = dt.hour
            l_code = 1 if 6 <= hour < 19 else 4
            light_add = 5 if l_code == 1 else 20

            forecast_risk = min(risk_add + light_add, 100)
            forecasts.append({
                'time': dt.strftime('%I %p'),
                'weather': item['weather'][0]['description'].title(),
                'temp': temp,
                'risk': forecast_risk,
                'risk_label': ('Low' if forecast_risk < 35
                               else 'Medium' if forecast_risk < 65
                               else 'High')
            })
        return forecasts
    except Exception:
        return []


def get_optimal_departure(forecasts):
    """Find the safest time to travel in next 24 hours"""
    if not forecasts:
        return None, None
    best = min(forecasts, key=lambda x: x['risk'])
    return best['time'], best['risk']


# ── Helper functions ──────────────────────────────

def get_local_time(lat, lon):
    try:
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lat=lat, lng=lon)
        if timezone_str:
            tz = pytz.timezone(timezone_str)
            local_time = datetime.now(tz)
            return local_time, timezone_str
        return datetime.now(), "UTC"
    except Exception:
        return datetime.now(), "UTC"


def get_weather(lat, lon):
    try:
        url = (f"https://api.openweathermap.org/data/2.5/weather"
               f"?lat={lat}&lon={lon}"
               f"&appid={st.secrets['OPENWEATHER_API_KEY']}")
        r = requests.get(url, timeout=5).json()
        weather_main = r['weather'][0]['main']
        weather_desc = r['weather'][0]['description'].title()
        temp = round(r['main']['temp'] - 273.15, 1)
        humidity = r['main']['humidity']
        wind_speed = round(r['wind']['speed'] * 3.6, 1)

        if 'Rain' in weather_main:
            weather_code, weather_label = 2, "Raining no high winds"
            surface_code, surface_label = 2, "Wet or damp"
        elif 'Snow' in weather_main:
            weather_code, weather_label = 3, "Snowing no high winds"
            surface_code, surface_label = 4, "Snow"
        elif 'Fog' in weather_main or 'Mist' in weather_main:
            weather_code, weather_label = 5, "Fog or mist"
            surface_code, surface_label = 1, "Dry"
        elif wind_speed > 50:
            weather_code, weather_label = 4, "Fine + high winds"
            surface_code, surface_label = 1, "Dry"
        else:
            weather_code, weather_label = 1, "Fine no high winds"
            surface_code, surface_label = 1, "Dry"

        return (weather_code, weather_label, surface_code,
                surface_label, temp, weather_desc,
                humidity, wind_speed)
    except Exception:
        return 1, "Fine no high winds", 1, "Dry", 0, "Unknown", 0, 0


def get_location_info(lat, lon):
    try:
        url = (f"https://nominatim.openstreetmap.org/reverse"
               f"?lat={lat}&lon={lon}&format=json")
        headers = {"User-Agent": "RoadSafetyAdvisor/1.0"}
        r = requests.get(url, headers=headers, timeout=5).json()
        address = r.get('address', {})
        location_name = r.get('display_name', 'Unknown location')

        if any(k in address for k in
               ['city', 'town', 'suburb', 'neighbourhood']):
            area_code, area_label = 1, "Urban"
        else:
            area_code, area_label = 2, "Rural"

        road_type = r.get('type', '')
        if road_type in ['motorway', 'trunk']:
            road_code, road_label = 2, "Dual carriageway"
        elif road_type == 'roundabout':
            road_code, road_label = 1, "Roundabout"
        else:
            road_code, road_label = 6, "Single carriageway"

        return area_code, area_label, road_code, road_label, location_name
    except Exception:
        return 1, "Urban", 6, "Single carriageway", "Unknown location"


def get_light_condition(local_time):
    hour = local_time.hour
    if 6 <= hour < 19:
        return 1, "Daylight"
    else:
        return 4, "Darkness - lights lit"


def get_destination_coords(destination):
    try:
        url = (f"https://nominatim.openstreetmap.org/search"
               f"?q={destination}&format=json&limit=1")
        headers = {"User-Agent": "RoadSafetyAdvisor/1.0"}
        r = requests.get(url, headers=headers, timeout=5).json()
        if r:
            dest_name = r[0]['display_name']
            dest_lat = float(r[0]['lat'])
            dest_lon = float(r[0]['lon'])
            return dest_name, dest_lat, dest_lon
        return destination, None, None
    except Exception:
        return destination, None, None


def get_nearest_hospitals(lat, lon):
    try:
        query = f"""
        [out:json];
        node["amenity"="hospital"](around:5000,{lat},{lon});
        out 3;
        """
        url = "https://overpass-api.de/api/interpreter"
        r = requests.post(url, data=query, timeout=8).json()
        hospitals = []
        for element in r.get('elements', [])[:3]:
            name = element.get('tags', {}).get('name', 'Hospital')
            hlat = element['lat']
            hlon = element['lon']
            hospitals.append((name, hlat, hlon))
        return hospitals
    except Exception:
        return []


def get_road_route(src_lat, src_lon, dest_lat, dest_lon):
    try:
        url = (f"http://router.project-osrm.org/route/v1/driving/"
               f"{src_lon},{src_lat};{dest_lon},{dest_lat}"
               f"?overview=full&geometries=geojson")
        r = requests.get(url, timeout=10).json()
        if r['code'] == 'Ok':
            coords = r['routes'][0]['geometry']['coordinates']
            route_coords = [[c[1], c[0]] for c in coords]
            distance = round(r['routes'][0]['distance'] / 1000, 1)
            duration = round(r['routes'][0]['duration'] / 60)
            return route_coords, distance, duration
        return None, 0, 0
    except Exception:
        return None, 0, 0


def calculate_risk_score(weather_code, light_code,
                          surface_code, road_code, area_code):
    score = 0
    score += {1: 5, 2: 25, 3: 35, 4: 20,
              5: 30, 6: 35, 7: 15}.get(weather_code, 10)
    score += {1: 5, 4: 20, 5: 25, 6: 30}.get(light_code, 15)
    score += {1: 5, 2: 20, 3: 30, 4: 35,
              5: 40}.get(surface_code, 10)
    score += {1: 10, 2: 5, 3: 15, 6: 10,
              7: 20}.get(road_code, 10)
    score += {1: 10, 2: 5}.get(area_code, 10)
    return min(score, 100)


def build_route_map(src_lat, src_lon, src_name,
                    dest_lat, dest_lon, dest_name,
                    hospitals, risk_score):
    center_lat = (src_lat + dest_lat) / 2
    center_lon = (src_lon + dest_lon) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='CartoDB positron'
    )

    route_coords, distance, duration = get_road_route(
        src_lat, src_lon, dest_lat, dest_lon)

    route_color = ('#27500A' if risk_score < 35
                   else '#854F0B' if risk_score < 65
                   else '#791F1F')

    if route_coords:
        folium.PolyLine(
            route_coords,
            color=route_color,
            weight=6,
            opacity=0.85,
            tooltip=(f"🛣 {distance}km | "
                     f"~{duration} mins | "
                     f"Risk: {risk_score}/100")
        ).add_to(m)

        if risk_score > 50 and len(route_coords) > 2:
            mid = route_coords[len(route_coords) // 2]
            folium.Marker(
                mid,
                popup=folium.Popup(
                    f"⚠️ Accident Blackspot Zone<br>"
                    f"Risk Score: {risk_score}/100",
                    max_width=200),
                tooltip="⚠️ High Risk Zone",
                icon=folium.Icon(color='orange',
                                  icon='warning-sign',
                                  prefix='glyphicon')
            ).add_to(m)
    else:
        folium.PolyLine(
            [[src_lat, src_lon], [dest_lat, dest_lon]],
            color=route_color,
            weight=5,
            opacity=0.8
        ).add_to(m)

    folium.Marker(
        [src_lat, src_lon],
        popup=folium.Popup(
            f"📍 You are here<br>{src_name[:40]}",
            max_width=200),
        tooltip="Your Location",
        icon=folium.Icon(color='blue', icon='user', prefix='fa')
    ).add_to(m)

    folium.Marker(
        [dest_lat, dest_lon],
        popup=folium.Popup(
            f"🏁 Destination<br>{dest_name[:40]}",
            max_width=200),
        tooltip="Destination",
        icon=folium.Icon(color='green', icon='flag', prefix='fa')
    ).add_to(m)

    for name, hlat, hlon in hospitals:
        folium.Marker(
            [hlat, hlon],
            popup=folium.Popup(f"🏥 {name}", max_width=200),
            tooltip=f"Hospital: {name}",
            icon=folium.Icon(color='red',
                              icon='plus-sign',
                              prefix='glyphicon')
        ).add_to(m)

    m.fit_bounds([[src_lat, src_lon], [dest_lat, dest_lon]])
    return m, distance, duration


# ── App header ────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🚦 AI Road Safety Advisor</h1>
    <p>Real-time GPS-based accident severity prediction with
    intelligent route safety analysis and self-learning AI</p>
</div>
""", unsafe_allow_html=True)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Model Accuracy", "90.58%")
col_b.metric("Dataset Records", "1.8M+")
col_c.metric("Algorithm", "Random Forest")
col_d.metric("AI Chatbot", "Gemini AI")

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🚗 Trip Advisor",
    "📊 Trip History & Profile",
    "🕐 Journey Planner"
])

# ════════════════════════════════════════════════════
# TAB 1 — TRIP ADVISOR
# ════════════════════════════════════════════════════
with tab1:

    st.markdown(
        '<p class="section-title">📍 Step 1 — Allow GPS location access</p>',
        unsafe_allow_html=True)
    st.info("Allow location access when your browser asks.")

    location = get_geolocation()

    if location:
        lat = location['coords']['latitude']
        lon = location['coords']['longitude']
        accuracy = round(location['coords'].get('accuracy', 0))

        local_time, timezone_str = get_local_time(lat, lon)

        with st.spinner("🔄 Fetching real-time road conditions..."):
            (weather_code, weather_label, surface_code,
             surface_label, temp, weather_desc,
             humidity, wind_speed) = get_weather(lat, lon)

            (area_code, area_label, road_code,
             road_label, location_name) = get_location_info(lat, lon)

            light_code, light_label = get_light_condition(local_time)

            risk_score = calculate_risk_score(
                weather_code, light_code,
                surface_code, road_code, area_code)

        st.success(
            f"📍 Location detected | "
            f"Accuracy: ±{accuracy}m | "
            f"Timezone: {timezone_str}")

        left_col, right_col = st.columns([1, 1.2])

        with left_col:

            st.markdown(
                '<p class="section-title">Auto-detected conditions</p>',
                unsafe_allow_html=True)
            st.markdown(f"""
            <div class="info-grid">
                <div class="info-card">
                    <p class="label">🌦 Weather</p>
                    <p class="value">{weather_desc}</p>
                </div>
                <div class="info-card">
                    <p class="label">🌡 Temp</p>
                    <p class="value">{temp}°C</p>
                </div>
                <div class="info-card">
                    <p class="label">💧 Humidity</p>
                    <p class="value">{humidity}%</p>
                </div>
                <div class="info-card">
                    <p class="label">💨 Wind</p>
                    <p class="value">{wind_speed} km/h</p>
                </div>
                <div class="info-card">
                    <p class="label">🌧 Surface</p>
                    <p class="value">{surface_label}</p>
                </div>
                <div class="info-card">
                    <p class="label">💡 Light</p>
                    <p class="value">{light_label}</p>
                </div>
                <div class="info-card">
                    <p class="label">🛣 Road</p>
                    <p class="value">{road_label}</p>
                </div>
                <div class="info-card">
                    <p class="label">🏙 Area</p>
                    <p class="value">{area_label}</p>
                </div>
                <div class="info-card">
                    <p class="label">⏰ Local Time</p>
                    <p class="value">{local_time.strftime('%I:%M %p')}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Risk score
            st.markdown(
                '<p class="section-title">📊 Real-time risk score</p>',
                unsafe_allow_html=True)
            risk_label_text = ('Low Risk' if risk_score < 35
                               else 'Medium Risk' if risk_score < 65
                               else 'High Risk')
            st.markdown(f"""
            <div class="risk-score-box">
                <p class="risk-score-num">{risk_score}</p>
                <p class="risk-score-label">/ 100 — {risk_label_text}</p>
            </div>
            """, unsafe_allow_html=True)

            # Fatigue check
            st.markdown(
                '<p class="section-title">😴 Driver fatigue check</p>',
                unsafe_allow_html=True)
            hours_driving = st.slider(
                "Hours since your last rest break?", 0, 12, 0)
            if hours_driving >= 4:
                st.error(
                    "🚨 Fatigue Warning — Take a break immediately.")
            elif hours_driving >= 2:
                st.warning("⚠️ Consider taking a short break soon.")
            else:
                st.success("✅ You are well rested. Stay alert.")

            # Best time
            st.markdown(
                '<p class="section-title">🕐 Best time to travel</p>',
                unsafe_allow_html=True)
            current_hour = local_time.hour
            if 6 <= current_hour < 9:
                st.warning(
                    "⚠️ Morning rush hour — consider leaving after 10 AM.")
            elif 9 <= current_hour < 17:
                st.success(
                    "✅ Good time to travel — light traffic expected.")
            elif 17 <= current_hour < 20:
                st.warning(
                    "⚠️ Evening rush hour — consider leaving after 8 PM.")
            else:
                st.info(
                    "🌙 Night driving — low traffic but reduced visibility.")

            # Destination
            st.markdown(
                '<p class="section-title">'
                '🗺 Step 2 — Enter destination</p>',
                unsafe_allow_html=True)
            destination = st.text_input(
                "Where are you going?",
                placeholder="e.g. Chennai Central Railway Station"
            )

            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("🔍 Analyze Route & Predict Risk")

        with right_col:
            st.markdown(
                '<p class="section-title">🗺 Live route map</p>',
                unsafe_allow_html=True)

            if predict_btn and destination:
                with st.spinner("🔄 Building your route..."):
                    dest_name, dest_lat, dest_lon = \
                        get_destination_coords(destination)
                    hospitals = get_nearest_hospitals(lat, lon)

                if dest_lat and dest_lon:
                    with st.spinner("🗺 Rendering map..."):
                        route_map, distance, duration = build_route_map(
                            lat, lon, location_name,
                            dest_lat, dest_lon, dest_name,
                            hospitals, risk_score
                        )
                        st.session_state['route_map'] = route_map
                        st.session_state['distance'] = distance
                        st.session_state['duration'] = duration
                        st.session_state['dest_name'] = dest_name
                        st.session_state['hospitals'] = hospitals
                        st.session_state['location_name'] = location_name

                    input_data = np.array([[
                        weather_code, road_code,
                        light_code, surface_code,
                        area_code
                    ]])
                    try:
                        result = model.predict(input_data)[0]
                        proba = model.predict_proba(input_data)[0]
                        confidence = round(max(proba) * 100, 1)
                        st.session_state['result'] = result
                        st.session_state['confidence'] = confidence
                        st.session_state['trip_data'] = {
                            'timestamp': local_time.strftime(
                                '%Y-%m-%d %H:%M'),
                            'location': location_name[:50],
                            'destination': dest_name[:50],
                            'weather': weather_desc,
                            'temp': temp,
                            'wind': wind_speed,
                            'light': light_label,
                            'surface': surface_label,
                            'area': area_label,
                            'road': road_label,
                            'risk_score': risk_score,
                            'prediction': (
                                'Low' if result == 1
                                else 'Medium' if result == 2
                                else 'High'),
                            'confidence': confidence,
                            'distance': distance,
                            'duration': duration,
                            'hours_driving': hours_driving
                        }
                    except Exception:
                        st.error("Prediction failed.")
                        st.stop()

                    # Save trip to Google Sheets
                    sheet = get_sheet()
                    if sheet and 'trip_data' in st.session_state:
                        td = st.session_state['trip_data']
                        save_trip_to_sheet(sheet, [
                            td['timestamp'], td['location'],
                            td['destination'], td['weather'],
                            td['temp'], td['wind'],
                            td['light'], td['surface'],
                            td['area'], td['road'],
                            td['risk_score'], td['prediction'],
                            td['confidence'], td['distance'],
                            td['duration'], td['hours_driving'],
                            'Pending'
                        ])

                    # AI advice
                    try:
                        route_prompt = f"""You are an expert road safety advisor.
A driver is traveling from {location_name[:40]} to {dest_name[:40]}.
Conditions: {weather_desc}, {temp}°C, {wind_speed}km/h wind,
{humidity}% humidity, {light_label}, {surface_label} road,
{area_label} area. Risk score: {risk_score}/100.
Driver hours since rest: {hours_driving} hours.
Give exactly 4 specific safety tips for this journey.
Each tip on a new line starting with an emoji.
Keep it short and practical."""
                        route_response = gemini.generate_content(
                            route_prompt)
                        if route_response and route_response.text:
                            st.session_state['ai_advice'] = \
                                route_response.text
                    except Exception:
                        st.session_state['ai_advice'] = None

                else:
                    st.error(
                        "Could not find destination. "
                        "Please try a more specific address.")

            # Show map from session state
            if 'route_map' in st.session_state:
                st_folium(
                    st.session_state['route_map'],
                    width=620, height=450,
                    key="persistent_map")

                dist = st.session_state.get('distance', 0)
                dur = st.session_state.get('duration', 0)
                dest_n = st.session_state.get('dest_name', '')
                loc_n = st.session_state.get('location_name', '')

                st.markdown(f"""
                <div class="info-grid">
                    <div class="info-card">
                        <p class="label">📍 From</p>
                        <p class="value">{loc_n[:22]}...</p>
                    </div>
                    <div class="info-card">
                        <p class="label">🏁 To</p>
                        <p class="value">{dest_n[:22]}...</p>
                    </div>
                    <div class="info-card">
                        <p class="label">⏰ Departure</p>
                        <p class="value">
                        {local_time.strftime('%I:%M %p')}</p>
                    </div>
                    <div class="info-card">
                        <p class="label">📏 Distance</p>
                        <p class="value">{dist} km</p>
                    </div>
                    <div class="info-card">
                        <p class="label">⏱ Est. Time</p>
                        <p class="value">{dur} mins</p>
                    </div>
                    <div class="info-card">
                        <p class="label">🚦 Risk Score</p>
                        <p class="value">{risk_score}/100</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if 'result' in st.session_state:
                    result = st.session_state['result']
                    confidence = st.session_state['confidence']

                    st.markdown(
                        '<p class="section-title">Risk assessment</p>',
                        unsafe_allow_html=True)

                    if result == 1:
                        st.markdown(f"""
                        <div class="meter-wrap">
                            <div class="meter-seg"
                            style="background:#97C459"></div>
                            <div class="meter-seg"
                            style="background:#EF9F27;opacity:.3"></div>
                            <div class="meter-seg"
                            style="background:#E24B4A;opacity:.3"></div>
                        </div>
                        <div class="meter-labels">
                            <span>Low</span>
                            <span>Medium</span>
                            <span>High</span>
                        </div>
                        <div class="result-low">
                            <p class="title">✅ Low risk — Safe to travel
                            <small style="font-weight:400;font-size:12px">
                            ({confidence}% confident)</small></p>
                            <p class="desc">Conditions are favorable.
                            Follow speed limits and stay alert.</p>
                        </div>
                        """, unsafe_allow_html=True)

                    elif result == 2:
                        st.markdown(f"""
                        <div class="meter-wrap">
                            <div class="meter-seg"
                            style="background:#97C459;opacity:.3"></div>
                            <div class="meter-seg"
                            style="background:#EF9F27"></div>
                            <div class="meter-seg"
                            style="background:#E24B4A;opacity:.3"></div>
                        </div>
                        <div class="meter-labels">
                            <span>Low</span>
                            <span>Medium</span>
                            <span>High</span>
                        </div>
                        <div class="result-mid">
                            <p class="title">⚠️ Medium risk — Drive carefully
                            <small style="font-weight:400;font-size:12px">
                            ({confidence}% confident)</small></p>
                            <p class="desc">Reduce speed.
                            Maintain safe distance. Avoid distractions.</p>
                        </div>
                        """, unsafe_allow_html=True)

                    else:
                        st.markdown(f"""
                        <div class="meter-wrap">
                            <div class="meter-seg"
                            style="background:#97C459;opacity:.3"></div>
                            <div class="meter-seg"
                            style="background:#EF9F27;opacity:.3"></div>
                            <div class="meter-seg"
                            style="background:#E24B4A"></div>
                        </div>
                        <div class="meter-labels">
                            <span>Low</span>
                            <span>Medium</span>
                            <span>High</span>
                        </div>
                        <div class="result-high">
                            <p class="title">
                            🚨 High risk — Avoid travel if possible
                            <small style="font-weight:400;font-size:12px">
                            ({confidence}% confident)</small></p>
                            <p class="desc">Postpone journey if possible.
                            If urgent, drive very slowly on main roads.</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Emergency auto-alert for high risk
                        st.error(
                            "🆘 EMERGENCY ALERT — High risk detected! "
                            "Nearest hospitals shown on map. "
                            "Consider calling emergency services if needed.")

                # Hospitals
                hospitals_saved = st.session_state.get('hospitals', [])
                if hospitals_saved:
                    st.markdown(
                        '<p class="section-title">🏥 Nearest hospitals</p>',
                        unsafe_allow_html=True)
                    for name, hlat, hlon in hospitals_saved:
                        st.markdown(f"""
                        <div class="feature-card">
                            <p class="feature-title">🏥 {name}</p>
                            <p class="feature-desc">
                            Lat: {round(hlat,4)},
                            Lon: {round(hlon,4)}</p>
                        </div>
                        """, unsafe_allow_html=True)

                # AI advice
                ai_advice = st.session_state.get('ai_advice')
                if ai_advice:
                    st.markdown(
                        '<p class="section-title">'
                        '🤖 AI route safety advice</p>',
                        unsafe_allow_html=True)
                    st.info(ai_advice)

                # Feedback loop
                st.markdown(
                    '<p class="section-title">'
                    '⭐ Was this prediction accurate?</p>',
                    unsafe_allow_html=True)
                col_y, col_n = st.columns(2)
                with col_y:
                    if st.button("👍 Yes, accurate"):
                        sheet = get_sheet()
                        if sheet:
                            try:
                                last_row = len(
                                    sheet.get_all_values())
                                sheet.update_cell(
                                    last_row, 17, 'Accurate')
                                st.success(
                                    "Thanks! Feedback saved ✅")
                            except Exception:
                                st.success("Thanks for feedback!")
                with col_n:
                    if st.button("👎 No, inaccurate"):
                        sheet = get_sheet()
                        if sheet:
                            try:
                                last_row = len(
                                    sheet.get_all_values())
                                sheet.update_cell(
                                    last_row, 17, 'Inaccurate')
                                st.info(
                                    "Thanks! We'll use this "
                                    "to improve the model.")
                            except Exception:
                                st.info("Thanks for feedback!")

            else:
                m = folium.Map(
                    location=[lat, lon],
                    zoom_start=14,
                    tiles='CartoDB positron')
                folium.Marker(
                    [lat, lon],
                    popup="📍 You are here",
                    tooltip="Your Location",
                    icon=folium.Icon(
                        color='blue', icon='user', prefix='fa')
                ).add_to(m)
                folium.Circle(
                    [lat, lon],
                    radius=500,
                    color='#0C447C',
                    fill=True,
                    fill_opacity=0.1
                ).add_to(m)
                st_folium(m, width=620, height=450,
                          key="default_map")
                st.info(
                    "👆 Enter your destination and "
                    "click Analyze Route.")

    else:
        st.warning("👆 Please allow location access.")
        st.markdown("""
        **How to allow location:**
        - A popup appears at the top of your browser
        - Click **Allow**
        - The app will automatically fetch your GPS coordinates
        """)

# ════════════════════════════════════════════════════
# TAB 2 — TRIP HISTORY & PROFILE
# ════════════════════════════════════════════════════
with tab2:
    st.markdown(
        '<p class="section-title">📋 Your trip history</p>',
        unsafe_allow_html=True)

    sheet = get_sheet()
    if sheet:
        df = get_trip_history(sheet)
        if not df.empty:
            st.dataframe(df, use_container_width=True)

            # Personal risk profile
            st.markdown(
                '<p class="section-title">'
                '🧠 Your personal risk profile</p>',
                unsafe_allow_html=True)

            total_trips = len(df)
            if 'Risk Score' in df.columns:
                avg_risk = round(
                    pd.to_numeric(
                        df['Risk Score'],
                        errors='coerce').mean(), 1)
            else:
                avg_risk = 0

            if 'Prediction' in df.columns:
                most_common = df['Prediction'].mode()
                common_risk = (most_common[0]
                               if len(most_common) > 0
                               else 'Unknown')
            else:
                common_risk = 'Unknown'

            if 'Feedback' in df.columns:
                accurate = len(
                    df[df['Feedback'] == 'Accurate'])
                accuracy_rate = (round(accurate / total_trips * 100)
                                 if total_trips > 0 else 0)
            else:
                accuracy_rate = 0

            st.markdown(f"""
            <div class="profile-box">
                <h4>🧠 Personal Driving Risk Profile</h4>
                <p>Total trips recorded: <strong>{total_trips}</strong></p>
                <p>Average risk score: <strong>{avg_risk}/100</strong></p>
                <p>Most common risk level: <strong>{common_risk}</strong></p>
                <p>Prediction accuracy (your feedback):
                <strong>{accuracy_rate}%</strong></p>
            </div>
            """, unsafe_allow_html=True)

            # Risk trend chart
            if 'Risk Score' in df.columns and total_trips > 1:
                st.markdown(
                    '<p class="section-title">📈 Risk score trend</p>',
                    unsafe_allow_html=True)
                df['Risk Score'] = pd.to_numeric(
                    df['Risk Score'], errors='coerce')
                st.line_chart(df['Risk Score'])

            # AI profile analysis
            if total_trips >= 3:
                st.markdown(
                    '<p class="section-title">'
                    '🤖 AI analysis of your driving pattern</p>',
                    unsafe_allow_html=True)
                try:
                    profile_prompt = f"""You are a road safety analyst.
A driver has completed {total_trips} trips.
Average risk score: {avg_risk}/100.
Most common risk level: {common_risk}.
Analyze their driving pattern and give 3 personalized
safety recommendations. Keep it short and practical.
Start each point with an emoji."""
                    profile_response = gemini.generate_content(
                        profile_prompt)
                    if profile_response and profile_response.text:
                        st.info(profile_response.text)
                except Exception:
                    pass
        else:
            st.info(
                "No trips recorded yet. "
                "Complete a trip in the Trip Advisor tab first!")
    else:
        st.error(
            "Could not connect to Google Sheets. "
            "Check your secrets configuration.")

# ════════════════════════════════════════════════════
# TAB 3 — JOURNEY PLANNER
# ════════════════════════════════════════════════════
with tab3:
    st.markdown(
        '<p class="section-title">'
        '🌦 24-hour weather risk forecast</p>',
        unsafe_allow_html=True)

    location2 = get_geolocation()

    if location2:
        lat2 = location2['coords']['latitude']
        lon2 = location2['coords']['longitude']

        with st.spinner("Fetching 24-hour forecast..."):
            forecasts = get_weather_forecast(lat2, lon2)
            best_time, best_risk = get_optimal_departure(forecasts)

        if forecasts:
            # Optimal departure
            st.markdown(
                '<p class="section-title">'
                '🕐 Optimal departure time</p>',
                unsafe_allow_html=True)

            risk_color = ('green' if best_risk < 35
                          else 'orange' if best_risk < 65
                          else 'red')
            st.markdown(f"""
            <div class="risk-score-box">
                <p class="risk-score-num">🕐 {best_time}</p>
                <p class="risk-score-label">
                Best time to depart — Risk: {best_risk}/100</p>
            </div>
            """, unsafe_allow_html=True)

            # Forecast table
            st.markdown(
                '<p class="section-title">'
                'Hourly risk forecast — next 24 hours</p>',
                unsafe_allow_html=True)

            cols = st.columns(4)
            for i, fc in enumerate(forecasts):
                with cols[i % 4]:
                    color = ('#EAF3DE' if fc['risk'] < 35
                             else '#FAEEDA' if fc['risk'] < 65
                             else '#FCEBEB')
                    border = ('#C0DD97' if fc['risk'] < 35
                              else '#FAC775' if fc['risk'] < 65
                              else '#F7C1C1')
                    text = ('#27500A' if fc['risk'] < 35
                            else '#633806' if fc['risk'] < 65
                            else '#791F1F')
                    st.markdown(f"""
                    <div style="background:{color};
                    border:0.5px solid {border};
                    border-radius:10px;
                    padding:12px;
                    text-align:center;
                    margin-bottom:8px">
                        <p style="font-size:13px;
                        font-weight:600;color:{text};margin:0">
                        {fc['time']}</p>
                        <p style="font-size:11px;
                        color:{text};margin:2px 0">
                        {fc['weather']}</p>
                        <p style="font-size:11px;
                        color:{text};margin:2px 0">
                        {fc['temp']}°C</p>
                        <p style="font-size:14px;
                        font-weight:700;color:{text};margin:0">
                        {fc['risk']}/100</p>
                        <p style="font-size:11px;
                        color:{text};margin:0">
                        {fc['risk_label']}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Risk timeline chart
            st.markdown(
                '<p class="section-title">📊 Risk timeline</p>',
                unsafe_allow_html=True)
            chart_data = pd.DataFrame({
                'Time': [f['time'] for f in forecasts],
                'Risk Score': [f['risk'] for f in forecasts]
            }).set_index('Time')
            st.line_chart(chart_data)

            # AI journey planning advice
            st.markdown(
                '<p class="section-title">'
                '🤖 AI journey planning advice</p>',
                unsafe_allow_html=True)
            try:
                forecast_summary = ", ".join([
                    f"{f['time']}: {f['risk_label']}"
                    for f in forecasts[:4]])
                plan_prompt = f"""You are a road safety journey planner.
The next 24 hours forecast: {forecast_summary}.
Best time to travel: {best_time} with risk {best_risk}/100.
Give 3 journey planning tips based on this forecast.
Keep it short, practical, start each with an emoji."""
                plan_response = gemini.generate_content(plan_prompt)
                if plan_response and plan_response.text:
                    st.info(plan_response.text)
            except Exception:
                pass

    else:
        st.warning("Please allow location access to see forecast.")

# ── Chatbot ───────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<div class="chat-header">'
    '💬 Road Safety AI Chatbot — Ask me anything'
    '</div>',
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
        "role": "user", "content": user_input
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
            bot_reply = (response.text if response and response.text
                         else "Sorry, please try asking differently.")
        except Exception:
            bot_reply = ("Sorry, I had trouble answering. "
                         "Please try rephrasing.")

    st.chat_message("assistant").write(bot_reply)
    st.session_state.messages.append({
        "role": "assistant", "content": bot_reply
    })

# ── About ─────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="about-box">
    <h4>About this project</h4>
    <p>
        AI Road Safety Advisor is a self-learning intelligent driving
        assistant that uses GPS to auto-detect location, fetches
        real-time weather via OpenWeatherMap, determines road conditions
        via OpenStreetMap, plots real road routes via OSRM, and predicts
        accident severity using a Random Forest ML model trained on 1.8M
        UK road accident records (90.58% accuracy). Features include
        live risk scoring, interactive route maps, accident blackspot
        warnings, nearest hospital detection, driver fatigue alerts,
        best travel time suggestions, 24-hour weather risk forecast,
        optimal departure time AI, trip history logging to Google Sheets,
        personal driving risk profile, user feedback self-learning loop,
        emergency auto-alerts, and AI-powered route safety advice via
        Google Gemini.<br><br>
        <strong>Dataset:</strong> UK Road Safety — data.gov.uk &nbsp;|&nbsp;
        <strong>Model:</strong> Random Forest Classifier &nbsp;|&nbsp;
        <strong>Built with:</strong> Python, Scikit-learn, Streamlit,
        Folium, OSRM, Gemini AI, Google Sheets
    </p>
</div>
""", unsafe_allow_html=True)
