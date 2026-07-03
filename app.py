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
from streamlit_autorefresh import st_autorefresh
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


@st.cache_data(ttl=300, show_spinner=False)
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


@st.cache_data(ttl=300, show_spinner=False)
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


@st.cache_data(ttl=3600, show_spinner=False)
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


@st.cache_data(ttl=300, show_spinner=False)
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


@st.cache_data(ttl=120, show_spinner=False)
def get_traffic_data(src_lat, src_lon, dest_lat, dest_lon):
    """Get live traffic congestion + delay using the TomTom Routing API
    (calculateRoute with traffic=true compares live travel time against
    free-flow travel time for the same route)."""
    try:
        url = (f"https://api.tomtom.com/routing/1/calculateRoute/"
               f"{src_lat},{src_lon}:{dest_lat},{dest_lon}/json")
        params = {"key": st.secrets["TOMTOM_API_KEY"], "traffic": "true"}
        r = requests.get(url, params=params, timeout=10).json()
        summary = r['routes'][0]['summary']
        time_with_traffic = summary['travelTimeInSeconds']
        time_no_traffic = summary.get(
            'noTrafficTravelTimeInSeconds', time_with_traffic)
        delay_sec = max(time_with_traffic - time_no_traffic, 0)
        delay_ratio = (delay_sec / time_no_traffic) if time_no_traffic else 0

        if delay_ratio < 0.10:
            level, color = "Light", "#27500A"
        elif delay_ratio < 0.30:
            level, color = "Moderate", "#854F0B"
        else:
            level, color = "Heavy", "#791F1F"

        return {
            "level": level,
            "color": color,
            "delay_min": round(delay_sec / 60, 1),
            "delay_ratio": round(delay_ratio, 2),
            "travel_time_min": round(time_with_traffic / 60),
            "travel_time_no_traffic_min": round(time_no_traffic / 60),
        }
    except Exception:
        return None


@st.cache_data(ttl=120, show_spinner=False)
def get_traffic_incidents(min_lat, min_lon, max_lat, max_lon):
    """Fetch nearby traffic incidents (jams, accidents, roadworks) from
    TomTom's Traffic Incident Details API within the route's bounding box."""
    try:
        # Pad the bounding box slightly so incidents right at the edges show up
        pad = 0.02
        bbox = (f"{min_lon - pad},{min_lat - pad},"
                f"{max_lon + pad},{max_lat + pad}")
        url = "https://api.tomtom.com/traffic/services/5/incidentDetails"
        params = {
            "bbox": bbox,
            "fields": "{incidents{type,geometry{type,coordinates},"
                      "properties{iconCategory,events{description}}}}",
            "key": st.secrets["TOMTOM_API_KEY"],
        }
        r = requests.get(url, params=params, timeout=8).json()
        incidents = []
        for inc in r.get('incidents', [])[:5]:
            props = inc.get('properties', {})
            events = props.get('events', [])
            desc = events[0]['description'] if events else "Traffic incident"
            coords = inc.get('geometry', {}).get('coordinates', [None, None])
            if coords and isinstance(coords[0], list):
                coords = coords[0]
            if coords and coords[0] is not None:
                incidents.append((desc, coords[1], coords[0]))
        return incidents
    except Exception:
        return []


def calculate_risk_score(weather_code, light_code,
                          surface_code, road_code, area_code,
                          traffic_delay_ratio=0):
    """Calculate a 0-100 risk score from conditions (+ live traffic delay)"""
    score = 0
    score += {1: 5, 2: 25, 3: 35, 4: 20, 5: 30, 6: 35, 7: 15
              }.get(weather_code, 10)
    score += {1: 5, 4: 20, 5: 25, 6: 30}.get(light_code, 15)
    score += {1: 5, 2: 20, 3: 30, 4: 35, 5: 40}.get(surface_code, 10)
    score += {1: 10, 2: 5, 3: 15, 6: 10, 7: 20}.get(road_code, 10)
    score += {1: 10, 2: 5}.get(area_code, 10)
    # Heavy live traffic raises effective risk (up to +15)
    score += min(round(traffic_delay_ratio * 50), 15)
    return min(score, 100)


def get_road_route(src_lat, src_lon, dest_lat, dest_lon):
    """Get actual road route coordinates using OSRM — completely free"""
    try:
        url = (f"http://router.project-osrm.org/route/v1/driving/"
               f"{src_lon},{src_lat};{dest_lon},{dest_lat}"
               f"?overview=full&geometries=geojson")
        r = requests.get(url, timeout=10).json()
        if r['code'] == 'Ok':
            coords = r['routes'][0]['geometry']['coordinates']
            # OSRM returns [lon, lat] — flip to [lat, lon] for folium
            route_coords = [[c[1], c[0]] for c in coords]
            distance = round(r['routes'][0]['distance'] / 1000, 1)
            duration = round(r['routes'][0]['duration'] / 60)
            return route_coords, distance, duration
        return None, 0, 0
    except Exception:
        return None, 0, 0


def build_route_map(src_lat, src_lon, src_name,
                    dest_lat, dest_lon, dest_name,
                    hospitals, risk_score):
    """Build interactive Folium map with real road route"""
    center_lat = (src_lat + dest_lat) / 2
    center_lon = (src_lon + dest_lon) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='CartoDB positron'
    )

    # Get real road route
    route_coords, distance, duration = get_road_route(
        src_lat, src_lon, dest_lat, dest_lon)

    # Route color based on risk
    route_color = ('#27500A' if risk_score < 35
                   else '#854F0B' if risk_score < 65
                   else '#791F1F')

    # Draw real road route or fallback to straight line
    if route_coords:
        folium.PolyLine(
            route_coords,
            color=route_color,
            weight=6,
            opacity=0.85,
            tooltip=f"🛣 Route | Risk: {risk_score}/100 | "
                    f"{distance}km | ~{duration} mins"
        ).add_to(m)

        # Add blackspot warning at midpoint if high risk
        if risk_score > 50 and len(route_coords) > 2:
            mid = route_coords[len(route_coords) // 2]
            folium.Marker(
                mid,
                popup=folium.Popup(
                    f"⚠️ Accident Blackspot Zone<br>"
                    f"Risk Score: {risk_score}/100",
                    max_width=200),
                tooltip="⚠️ High Risk Zone",
                icon=folium.Icon(
                    color='orange',
                    icon='warning-sign',
                    prefix='glyphicon')
            ).add_to(m)
    else:
        # Fallback straight line
        folium.PolyLine(
            [[src_lat, src_lon], [dest_lat, dest_lon]],
            color=route_color,
            weight=5,
            opacity=0.8
        ).add_to(m)

    # Source marker
    folium.Marker(
        [src_lat, src_lon],
        popup=folium.Popup(
            f"📍 You are here<br>{src_name[:40]}",
            max_width=200),
        tooltip="Your Location",
        icon=folium.Icon(color='blue', icon='user', prefix='fa')
    ).add_to(m)

    # Destination marker
    folium.Marker(
        [dest_lat, dest_lon],
        popup=folium.Popup(
            f"🏁 Destination<br>{dest_name[:40]}",
            max_width=200),
        tooltip="Destination",
        icon=folium.Icon(color='green', icon='flag', prefix='fa')
    ).add_to(m)

    # Hospital markers
    for name, hlat, hlon in hospitals:
        folium.Marker(
            [hlat, hlon],
            popup=folium.Popup(f"🏥 {name}", max_width=200),
            tooltip=f"Hospital: {name}",
            icon=folium.Icon(
                color='red',
                icon='plus-sign',
                prefix='glyphicon')
        ).add_to(m)

    # Fit map to show full route
    m.fit_bounds([[src_lat, src_lon], [dest_lat, dest_lon]])

    return m, distance, duration


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

# ── GPS fetch (silent, no visible step) ────────────
location = get_geolocation()

if location:
    # Round to ~11m precision so GPS jitter doesn't bust the caches above
    # or churn the map object on every tick — this is a big part of the
    # "everything keeps resetting" fix.
    lat = round(location['coords']['latitude'], 4)
    lon = round(location['coords']['longitude'], 4)
    accuracy = round(location['coords'].get('accuracy', 0))

    # Get local time using GPS timezone
    local_time, timezone_str = get_local_time(lat, lon)

    # Real-time background tick every 60s — this is what makes the
    # fatigue timer advance automatically without any user interaction.
    st_autorefresh(interval=60_000, key="fatigue_autorefresh")

    # ── Automatic fatigue tracking ─────────────────
    if 'drive_start_time' not in st.session_state:
        st.session_state['drive_start_time'] = local_time
        st.session_state['fatigue_alert_2h'] = False
        st.session_state['fatigue_alert_4h'] = False

    elapsed = local_time - st.session_state['drive_start_time']
    hours_driving = elapsed.total_seconds() / 3600

    if hours_driving >= 4 and not st.session_state['fatigue_alert_4h']:
        st.toast("🚨 4+ hours driving — take a mandatory break now!", icon="🚨")
        st.session_state['fatigue_alert_4h'] = True
    elif hours_driving >= 2 and not st.session_state['fatigue_alert_2h']:
        st.toast("⚠️ 2+ hours driving — a short break is due soon.", icon="⚠️")
        st.session_state['fatigue_alert_2h'] = True

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

    # Make the current conditions available to the chatbot below
    st.session_state['current_conditions'] = {
        'location_name': location_name,
        'weather_desc': weather_desc,
        'temp': temp,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'light_label': light_label,
        'surface_label': surface_label,
        'road_label': road_label,
        'area_label': area_label,
        'risk_score': risk_score,
        'hours_driving': round(hours_driving, 1),
        'timezone_str': timezone_str,
    }

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

        # Fatigue warning — fully automatic, real-time
        st.markdown('<p class="section-title">😴 Driver fatigue check (auto)</p>',
                    unsafe_allow_html=True)
        fh = int(hours_driving)
        fm = int((hours_driving - fh) * 60)
        st.caption(f"⏱ Time since last break: {fh}h {fm}m "
                   f"(updates automatically every minute)")
        if hours_driving >= 4:
            st.error("🚨 Fatigue Warning — You have been driving too long. "
                     "Take a break immediately before continuing.")
        elif hours_driving >= 2:
            st.warning("⚠️ You should take a short break soon. "
                       "Fatigue increases accident risk by 3x.")
        else:
            st.success("✅ You are well rested. Stay alert.")

        if st.button("☕ I took a break — reset timer"):
            st.session_state['drive_start_time'] = local_time
            st.session_state['fatigue_alert_2h'] = False
            st.session_state['fatigue_alert_4h'] = False
            st.rerun()

        # Destination input
        st.markdown('<p class="section-title">🗺 Step 1 — Enter destination</p>',
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
                # Live traffic (TomTom): travel time with vs without traffic
                with st.spinner("🚦 Checking live traffic..."):
                    traffic = get_traffic_data(lat, lon, dest_lat, dest_lon)
                    incidents = get_traffic_incidents(
                        min(lat, dest_lat), min(lon, dest_lon),
                        max(lat, dest_lat), max(lon, dest_lon)
                    )

                traffic_delay_ratio = traffic['delay_ratio'] if traffic else 0

                # Fold live traffic into the risk score used for this route
                route_risk_score = calculate_risk_score(
                    weather_code, light_code, surface_code,
                    road_code, area_code, traffic_delay_ratio
                )

                # Build map (unpack distance/duration from OSRM)
                route_map, distance, duration = build_route_map(
                    lat, lon, location_name,
                    dest_lat, dest_lon, dest_name,
                    hospitals, route_risk_score
                )

                # Persist everything needed to redraw this map on future
                # reruns (e.g. when the chatbot or autorefresh triggers one)
                st.session_state['route_map'] = route_map
                st.session_state['distance'] = distance
                st.session_state['duration'] = duration
                st.session_state['route_active'] = True
                st.session_state['location_name'] = location_name
                st.session_state['dest_name'] = dest_name
                st.session_state['departure_time'] = local_time.strftime('%I:%M %p')
                st.session_state['route_risk_score'] = route_risk_score
                st.session_state['traffic'] = traffic
                st.session_state['incidents'] = incidents

                st_folium(route_map, width=600, height=450,
                          key="persistent_map", returned_objects=[])

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
                    <div class="info-card">
                        <p class="label">📏 Distance</p>
                        <p class="value">{distance} km</p>
                    </div>
                    <div class="info-card">
                        <p class="label">⏱ Est. Time</p>
                        <p class="value">~{duration} mins</p>
                    </div>
                    <div class="info-card">
                        <p class="label">🚦 Risk Score</p>
                        <p class="value">{route_risk_score}/100</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Traffic conditions
                st.markdown('<p class="section-title">🚦 Live traffic</p>',
                            unsafe_allow_html=True)
                if traffic:
                    st.markdown(f"""
                    <div class="info-grid">
                        <div class="info-card">
                            <p class="label">Congestion</p>
                            <p class="value" style="color:{traffic['color']}">
                            {traffic['level']}</p>
                        </div>
                        <div class="info-card">
                            <p class="label">Delay vs free-flow</p>
                            <p class="value">+{traffic['delay_min']} min</p>
                        </div>
                        <div class="info-card">
                            <p class="label">Travel time now</p>
                            <p class="value">{traffic['travel_time_min']} min</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if incidents:
                        for desc, ilat, ilon in incidents:
                            st.markdown(f"""
                            <div class="feature-card">
                                <p class="feature-title">⚠️ Incident nearby</p>
                                <p class="feature-desc">{desc}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.caption("No reported incidents along this route right now.")
                else:
                    st.info("Live traffic data unavailable right now — "
                            "showing route without traffic adjustment.")

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
                        traffic_note = (
                            f"{traffic['level']} traffic, +{traffic['delay_min']} "
                            f"min delay vs free-flow" if traffic
                            else "traffic data unavailable")
                        route_prompt = f"""You are an expert road safety advisor.
A driver is traveling from {location_name[:40]} to {dest_name[:40]}.
Conditions: {weather_desc}, {temp}°C, {wind_speed}km/h wind,
{humidity}% humidity, {light_label}, {surface_label} road,
{area_label} area. Live traffic: {traffic_note}.
Risk score: {route_risk_score}/100.
Driver hours since rest: {round(hours_driving, 1)} hours.
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
                st_folium(m, width=600, height=450,
                          key="not_found_map", returned_objects=[])
                st.error("Could not find destination. "
                         "Please try a more specific address.")

        else:
            # No new prediction was triggered on this rerun (e.g. the
            # chatbot below or the 60s fatigue autorefresh caused this
            # rerun). Show the previously computed route map if we have
            # one, instead of resetting back to the plain default map.
            if st.session_state.get('route_active') and 'route_map' in st.session_state:
                st_folium(st.session_state['route_map'],
                          width=600, height=450,
                          key="persistent_map", returned_objects=[])

                saved_risk = st.session_state.get('route_risk_score', risk_score)
                st.markdown('<p class="section-title">Route summary</p>',
                            unsafe_allow_html=True)
                st.markdown(f"""
                <div class="info-grid">
                    <div class="info-card">
                        <p class="label">📍 From</p>
                        <p class="value">{st.session_state.get('location_name', '')[:25]}...</p>
                    </div>
                    <div class="info-card">
                        <p class="label">🏁 To</p>
                        <p class="value">{st.session_state.get('dest_name', '')[:25]}...</p>
                    </div>
                    <div class="info-card">
                        <p class="label">⏰ Departure</p>
                        <p class="value">{st.session_state.get('departure_time', '')}</p>
                    </div>
                    <div class="info-card">
                        <p class="label">📏 Distance</p>
                        <p class="value">{st.session_state.get('distance', 0)} km</p>
                    </div>
                    <div class="info-card">
                        <p class="label">⏱ Est. Time</p>
                        <p class="value">~{st.session_state.get('duration', 0)} mins</p>
                    </div>
                    <div class="info-card">
                        <p class="label">🚦 Risk Score</p>
                        <p class="value">{saved_risk}/100</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                saved_traffic = st.session_state.get('traffic')
                if saved_traffic:
                    st.markdown('<p class="section-title">🚦 Live traffic</p>',
                                unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="info-grid">
                        <div class="info-card">
                            <p class="label">Congestion</p>
                            <p class="value" style="color:{saved_traffic['color']}">
                            {saved_traffic['level']}</p>
                        </div>
                        <div class="info-card">
                            <p class="label">Delay vs free-flow</p>
                            <p class="value">+{saved_traffic['delay_min']} min</p>
                        </div>
                        <div class="info-card">
                            <p class="label">Travel time now</p>
                            <p class="value">{saved_traffic['travel_time_min']} min</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Default map showing current location — cached in
                # session_state so the 60s autorefresh tick doesn't rebuild
                # (and remount / reset the zoom of) a brand new map object.
                cache_key = (round(lat, 3), round(lon, 3))
                if st.session_state.get('default_map_key') != cache_key:
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
                    st.session_state['default_map'] = m
                    st.session_state['default_map_key'] = cache_key

                st_folium(st.session_state['default_map'], width=600, height=450,
                          key="default_map", returned_objects=[])
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
    '💬 Road Safety AI Chatbot — knows your live trip data'
    '</div>',
    unsafe_allow_html=True)

# Show a quick summary chip of what context the bot currently has
cc = st.session_state.get('current_conditions')
if cc:
    chips = [f"📍 {cc['location_name'][:28]}",
             f"🌦 {cc['weather_desc']}",
             f"🚦 Risk {cc['risk_score']}/100",
             f"😴 {cc['hours_driving']}h since break"]
    if st.session_state.get('route_active'):
        chips.append(f"🏁 To {st.session_state.get('dest_name', '')[:28]}")
        tr = st.session_state.get('traffic')
        if tr:
            chips.append(f"🚗 Traffic: {tr['level']}")
    st.caption(" · ".join(chips))
else:
    st.caption("Allow location access above so the chatbot can see your live conditions.")

st.markdown("<br>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Quick suggestion chips tied to whatever is currently on the map
if st.session_state.get('route_active'):
    suggestions = ["Is my route safe right now?",
                   "How's the traffic ahead?",
                   "Should I take a break?"]
else:
    suggestions = ["Is it safe to drive right now?",
                   "What should I watch out for?",
                   "Tips for tonight's drive?"]

clicked_suggestion = None
sugg_cols = st.columns(len(suggestions))
for i, (col, s) in enumerate(zip(sugg_cols, suggestions)):
    if col.button(s, use_container_width=True, key=f"sugg_{i}"):
        clicked_suggestion = s

user_input = st.chat_input("Ask a road safety question...") or clicked_suggestion

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({
        "role": "user", "content": user_input
    })

    history = ""
    for msg in st.session_state.messages[:-1]:
        role = "User" if msg["role"] == "user" else "Bot"
        history += f"{role}: {msg['content']}\n"

    # Build a live context block from whatever the map/inputs currently show
    if cc:
        context_str = f"""Live conditions for this driver right now:
- Location: {cc['location_name'][:60]}
- Weather: {cc['weather_desc']}, {cc['temp']}°C, wind {cc['wind_speed']}km/h, humidity {cc['humidity']}%
- Road: {cc['surface_label']} surface, {cc['road_label']}, {cc['area_label']} area, {cc['light_label']}
- Current risk score: {cc['risk_score']}/100
- Time driving since last break: {cc['hours_driving']} hours"""

        if st.session_state.get('route_active'):
            context_str += f"""
- Destination: {st.session_state.get('dest_name', '')[:60]}
- Distance: {st.session_state.get('distance', 0)} km, ETA: {st.session_state.get('duration', 0)} min
- Route risk score (weather+road+traffic combined): {st.session_state.get('route_risk_score', cc['risk_score'])}/100"""
            tr = st.session_state.get('traffic')
            if tr:
                context_str += (f"\n- Live traffic: {tr['level']} congestion, "
                                 f"+{tr['delay_min']} min delay vs free-flow")
            incs = st.session_state.get('incidents')
            if incs:
                context_str += ("\n- Nearby incidents: "
                                 + "; ".join(d for d, _, _ in incs[:3]))
    else:
        context_str = "No live location or trip data is available for this driver yet."

    with st.spinner("Thinking..."):
        prompt = f"""You are a helpful road safety advisor chatbot embedded in a
live trip-planning app. Only answer questions related to road safety, driving
tips, accident prevention, traffic rules, vehicle safety, or the driver's
current trip described below. When relevant, use the live data to give a
specific, personalized answer (cite the actual risk score, traffic level,
distance, or fatigue hours) instead of a generic one. Keep answers short,
clear and practical (max 5 lines). If the question is unrelated to road
safety or this trip, politely say you can only help with road safety topics.

{context_str}

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
        conditions via OpenStreetMap, checks live traffic congestion and
        incidents via TomTom, and predicts accident severity using a
        Random Forest ML model trained on 1.8M UK road accident records
        (90.58% accuracy). Features include live risk scoring, real
        road-following route maps, accident blackspot warnings, nearest
        hospital detection, fully automatic driver fatigue tracking with
        break alerts, and AI-powered route safety advice via Google
        Gemini.<br><br>
        <strong>Dataset:</strong> UK Road Safety — data.gov.uk &nbsp;|&nbsp;
        <strong>Model:</strong> Random Forest Classifier &nbsp;|&nbsp;
        <strong>Built with:</strong> Python, Scikit-learn, Streamlit,
        Folium, TomTom, Gemini AI
    </p>
</div>
""", unsafe_allow_html=True)
