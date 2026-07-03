# ── Imports ───────────────────────────────────────
import streamlit as st
import pickle
import numpy as np
import google.generativeai as genai
import requests
import folium
from folium import plugins
from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation
from datetime import datetime
from timezonefinder import TimezoneFinder
import pytz

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

def get_local_time(lat, lon):
    """Get accurate local time using GPS coordinates"""
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
    """Fetch real-time weather from OpenWeatherMap"""
    try:
        url = (f"https://api.openweathermap.org/data/2.5/weather"
               f"?lat={lat}&lon={lon}&appid={st.secrets['OPENWEATHER_API_KEY']}")
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
    """Get area and road type from OpenStreetMap"""
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
    """Detect light condition from local time"""
    hour = local_time.hour
    if 6 <= hour < 19:
        return 1, "Daylight"
    else:
        return 4, "Darkness - lights lit"


def get_destination_coords(destination):
    """Geocode destination using OpenStreetMap — completely free"""
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
    """Find nearest hospitals using OpenStreetMap Overpass API"""
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


def calculate_risk_score(weather_code, light_code,
                          surface_code, road_code, area_code):
    """Calculate a 0-100 risk score from conditions"""
    score = 0
    score += {1: 5, 2: 25, 3: 35, 4: 20, 5: 30, 6: 35, 7: 15
              }.get(weather_code, 10)
    score += {1: 5, 4: 20, 5: 25, 6: 30}.get(light_code, 15)
    score += {1: 5, 2: 20, 3: 30, 4: 35, 5: 40}.get(surface_code, 10)
    score += {1: 10, 2: 5, 3: 15, 6: 10, 7: 20}.get(road_code, 10)
    score += {1: 10, 2: 5}.get(area_code, 10)
    return min(score, 100)


def build_route_map(src_lat, src_lon, src_name,
                    dest_lat, dest_lon, dest_name,
                    hospitals, risk_score):
    """Build interactive Folium map with route and markers"""
    center_lat = (src_lat + dest_lat) / 2
    center_lon = (src_lon + dest_lon) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='CartoDB positron'
    )

    # Source marker
    folium.Marker(
        [src_lat, src_lon],
        popup=folium.Popup(f"📍 You are here<br>{src_name[:40]}", max_width=200),
        tooltip="Your Location",
        icon=folium.Icon(color='blue', icon='user', prefix='fa')
    ).add_to(m)

    # Destination marker
    folium.Marker(
        [dest_lat, dest_lon],
        popup=folium.Popup(f"🏁 Destination<br>{dest_name[:40]}", max_width=200),
        tooltip="Destination",
        icon=folium.Icon(color='green', icon='flag', prefix='fa')
    ).add_to(m)

    # Route line
    route_color = ('#27500A' if risk_score < 35
                   else '#854F0B' if risk_score < 65
                   else '#791F1F')
    folium.PolyLine(
        [[src_lat, src_lon], [dest_lat, dest_lon]],
        color=route_color,
        weight=5,
        opacity=0.8,
        tooltip=f"Risk score: {risk_score}/100"
    ).add_to(m)

    # Midpoint blackspot warning
    mid_lat = (src_lat + dest_lat) / 2
    mid_lon = (src_lon + dest_lon) / 2
    if risk_score > 50:
        folium.Marker(
            [mid_lat, mid_lon],
            popup=folium.Popup(
                f"⚠️ Accident Blackspot Zone<br>Risk Score: {risk_score}/100",
                max_width=200),
            tooltip="⚠️ High Risk Zone",
            icon=folium.Icon(color='red', icon='warning-sign',
                              prefix='glyphicon')
        ).add_to(m)

    # Hospital markers
    for name, hlat, hlon in hospitals:
        folium.Marker(
            [hlat, hlon],
            popup=folium.Popup(f"🏥 {name}", max_width=200),
            tooltip=f"Hospital: {name}",
            icon=folium.Icon(color='red', icon='plus-sign',
                              prefix='glyphicon')
        ).add_to(m)

    # Fit bounds
    m.fit_bounds([[src_lat, src_lon], [dest_lat, dest_lon]])

    return m


# ── App header ────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🚦 AI Road Safety Advisor</h1>
    <p>Real-time GPS-based accident severity prediction with
    intelligent route safety analysis</p>
</div>
""", unsafe_allow_html=True)

# ── Metrics row ───────────────────────────────────
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Model Accuracy", "90.58%")
col_b.metric("Dataset Records", "1.8M+")
col_c.metric("Algorithm", "Random Forest")
col_d.metric("AI Chatbot", "Gemini AI")

st.markdown("<br>", unsafe_allow_html=True)

# ── GPS fetch ─────────────────────────────────────
st.markdown('<p class="section-title">📍 Step 1 — Allow GPS location access</p>',
            unsafe_allow_html=True)
st.info("Click the button below — your browser will ask for location permission. Click Allow.")

location = get_geolocation()

if location:
    lat = location['coords']['latitude']
    lon = location['coords']['longitude']
    accuracy = round(location['coords'].get('accuracy', 0))

    # Get local time using GPS timezone
    local_time, timezone_str = get_local_time(lat, lon)

    # Fetch all conditions
    with st.spinner("🔄 Fetching real-time road conditions..."):
        (weather_code, weather_label, surface_code,
         surface_label, temp, weather_desc,
         humidity, wind_speed) = get_weather(lat, lon)

        (area_code, area_label, road_code,
         road_label, location_name) = get_location_info(lat, lon)

        light_code, light_label = get_light_condition(local_time)

        risk_score = calculate_risk_score(
            weather_code, light_code,
            surface_code, road_code, area_code
        )

    st.success(f"📍 Location detected — Accuracy: ±{accuracy}m | "
               f"Timezone: {timezone_str}")

    # ── Two column layout ─────────────────────────
    left_col, right_col = st.columns([1, 1.2])

    with left_col:

        # Auto-detected conditions
        st.markdown('<p class="section-title">Auto-detected conditions</p>',
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

        # Real-time risk score
        st.markdown('<p class="section-title">📊 Real-time risk score</p>',
                    unsafe_allow_html=True)
        risk_color = ('#27500A' if risk_score < 35
                      else '#854F0B' if risk_score < 65
                      else '#791F1F')
        risk_label_text = ('Low Risk' if risk_score < 35
                           else 'Medium Risk' if risk_score < 65
                           else 'High Risk')
        st.markdown(f"""
        <div class="risk-score-box">
            <p class="risk-score-num">{risk_score}</p>
            <p class="risk-score-label">/ 100 — {risk_label_text}</p>
        </div>
        """, unsafe_allow_html=True)

        # Fatigue warning
        st.markdown('<p class="section-title">😴 Driver fatigue check</p>',
                    unsafe_allow_html=True)
        hours_driving = st.slider(
            "Hours since your last rest break?", 0, 12, 0)
        if hours_driving >= 4:
            st.error("🚨 Fatigue Warning — You have been driving too long. "
                     "Take a break immediately before continuing.")
        elif hours_driving >= 2:
            st.warning("⚠️ You should take a short break soon. "
                       "Fatigue increases accident risk by 3x.")
        else:
            st.success("✅ You are well rested. Stay alert.")

        # Destination input
        st.markdown('<p class="section-title">🗺 Step 2 — Enter destination</p>',
                    unsafe_allow_html=True)
        destination = st.text_input(
            "Where are you going?",
            placeholder="e.g. Chennai Central Railway Station"
        )

        # Best time to travel suggestion
        st.markdown('<p class="section-title">🕐 Best time to travel today</p>',
                    unsafe_allow_html=True)
        current_hour = local_time.hour
        if 6 <= current_hour < 9:
            st.warning("⚠️ Morning rush hour — High traffic. "
                       "Consider leaving after 10 AM.")
        elif 9 <= current_hour < 17:
            st.success("✅ Good time to travel — Light traffic expected.")
        elif 17 <= current_hour < 20:
            st.warning("⚠️ Evening rush hour — High traffic. "
                       "Consider leaving after 8 PM.")
        else:
            st.info("🌙 Night driving — Low traffic but reduced visibility. "
                    "Drive carefully.")

        # Predict button
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 Analyze Route & Predict Risk")

    with right_col:
        st.markdown('<p class="section-title">🗺 Live route map</p>',
                    unsafe_allow_html=True)

        if predict_btn and destination:
            with st.spinner("🔄 Building your route map..."):
                dest_name, dest_lat, dest_lon = get_destination_coords(
                    destination)
                hospitals = get_nearest_hospitals(lat, lon)

            if dest_lat and dest_lon:
                # Build and show map
                route_map = build_route_map(
                    lat, lon, location_name,
                    dest_lat, dest_lon, dest_name,
                    hospitals, risk_score
                )
                st_folium(route_map, width=600, height=450)

                # Route summary
                st.markdown('<p class="section-title">Route summary</p>',
                            unsafe_allow_html=True)
                st.markdown(f"""
                <div class="info-grid">
                    <div class="info-card">
                        <p class="label">📍 From</p>
                        <p class="value">{location_name[:25]}...</p>
                    </div>
                    <div class="info-card">
                        <p class="label">🏁 To</p>
                        <p class="value">{dest_name[:25]}...</p>
                    </div>
                    <div class="info-card">
                        <p class="label">⏰ Departure</p>
                        <p class="value">{local_time.strftime('%I:%M %p')}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ML Prediction
                input_data = np.array([[
                    weather_code, road_code,
                    light_code, surface_code,
                    area_code
                ]])

                with st.spinner("🤖 Running AI prediction..."):
                    try:
                        result = model.predict(input_data)[0]
                        proba = model.predict_proba(input_data)[0]
                        confidence = round(max(proba) * 100, 1)
                    except Exception:
                        st.error("Prediction failed. Please try again.")
                        st.stop()

                st.markdown('<p class="section-title">Risk assessment</p>',
                            unsafe_allow_html=True)

                if result == 1:
                    st.markdown(f"""
                    <div class="meter-wrap">
                        <div class="meter-seg" style="background:#97C459"></div>
                        <div class="meter-seg" style="background:#EF9F27;opacity:.3"></div>
                        <div class="meter-seg" style="background:#E24B4A;opacity:.3"></div>
                    </div>
                    <div class="meter-labels">
                        <span>Low</span><span>Medium</span><span>High</span>
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
                        <div class="meter-seg" style="background:#97C459;opacity:.3"></div>
                        <div class="meter-seg" style="background:#EF9F27"></div>
                        <div class="meter-seg" style="background:#E24B4A;opacity:.3"></div>
                    </div>
                    <div class="meter-labels">
                        <span>Low</span><span>Medium</span><span>High</span>
                    </div>
                    <div class="result-mid">
                        <p class="title">⚠️ Medium risk — Drive carefully
                        <small style="font-weight:400;font-size:12px">
                        ({confidence}% confident)</small></p>
                        <p class="desc">Reduce speed. Maintain safe distance.
                        Avoid distractions.</p>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.markdown(f"""
                    <div class="meter-wrap">
                        <div class="meter-seg" style="background:#97C459;opacity:.3"></div>
                        <div class="meter-seg" style="background:#EF9F27;opacity:.3"></div>
                        <div class="meter-seg" style="background:#E24B4A"></div>
                    </div>
                    <div class="meter-labels">
                        <span>Low</span><span>Medium</span><span>High</span>
                    </div>
                    <div class="result-high">
                        <p class="title">🚨 High risk — Avoid travel if possible
                        <small style="font-weight:400;font-size:12px">
                        ({confidence}% confident)</small></p>
                        <p class="desc">Postpone your journey if possible.
                        If urgent, drive very slowly on main roads only.</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Hospitals info
                if hospitals:
                    st.markdown(
                        '<p class="section-title">🏥 Nearest hospitals</p>',
                        unsafe_allow_html=True)
                    for name, hlat, hlon in hospitals:
                        st.markdown(f"""
                        <div class="feature-card">
                            <p class="feature-title">🏥 {name}</p>
                            <p class="feature-desc">
                            Lat: {round(hlat,4)}, Lon: {round(hlon,4)}</p>
                        </div>
                        """, unsafe_allow_html=True)

                # AI route advice
                with st.spinner("🤖 Getting AI route safety advice..."):
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
                        route_response = gemini.generate_content(route_prompt)
                        if route_response and route_response.text:
                            st.markdown(
                                '<p class="section-title">'
                                '🤖 AI route safety advice</p>',
                                unsafe_allow_html=True)
                            st.info(route_response.text)
                    except Exception:
                        pass

            else:
                # Show basic map without route
                m = folium.Map(location=[lat, lon],
                               zoom_start=14,
                               tiles='CartoDB positron')
                folium.Marker(
                    [lat, lon],
                    tooltip="You are here",
                    icon=folium.Icon(color='blue', icon='user', prefix='fa')
                ).add_to(m)
                st_folium(m, width=600, height=450)
                st.error("Could not find destination. "
                         "Please try a more specific address.")

        else:
            # Default map showing current location
            m = folium.Map(location=[lat, lon],
                           zoom_start=14,
                           tiles='CartoDB positron')
            folium.Marker(
                [lat, lon],
                popup="📍 You are here",
                tooltip="Your Location",
                icon=folium.Icon(color='blue', icon='user', prefix='fa')
            ).add_to(m)
            folium.Circle(
                [lat, lon],
                radius=500,
                color='#0C447C',
                fill=True,
                fill_opacity=0.1
            ).add_to(m)
            st_folium(m, width=600, height=450)
            st.info("👆 Enter your destination and click "
                    "Analyze Route to see the full route map.")

else:
    st.warning("👆 Please allow location access when your browser asks.")
    st.markdown("""
    **How to allow location:**
    - A popup appears at the top of your browser
    - Click **Allow**
    - The app will automatically fetch your GPS coordinates
    """)

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
            bot_reply = ("Sorry, I had trouble answering that. "
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
        AI Road Safety Advisor uses GPS to auto-detect your location,
        fetches real-time weather via OpenWeatherMap, determines road
        conditions via OpenStreetMap, and predicts accident severity
        using a Random Forest ML model trained on 1.8M UK road accident
        records (90.58% accuracy). Features include live risk scoring,
        interactive route maps, accident blackspot warnings, nearest
        hospital detection, driver fatigue alerts, and AI-powered
        route safety advice via Google Gemini.<br><br>
        <strong>Dataset:</strong> UK Road Safety — data.gov.uk &nbsp;|&nbsp;
        <strong>Model:</strong> Random Forest Classifier &nbsp;|&nbsp;
        <strong>Built with:</strong> Python, Scikit-learn, Streamlit,
        Folium, Gemini AI
    </p>
</div>
""", unsafe_allow_html=True)
