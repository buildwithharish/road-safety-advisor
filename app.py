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

# ── Custom CSS — warm terracotta/amber theme ──────
# Premium pass: type system, elevation tokens, refined components,
# styled tabs/chat/alerts. Palette unchanged, execution upgraded.
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
    :root {
        --clr-bg: #FBF3E7;
        --clr-card: #FDF6EB;
        --clr-card-alt: #FEF9F0;
        --clr-border: #EAD3AA;
        --clr-terracotta: #A8431F;
        --clr-amber: #E8963C;
        --clr-ink: #3A2416;
        --clr-muted: #7A6047;
        --clr-label: #9C7B57;
        --shadow-sm: 0 1px 2px rgba(58,36,22,0.05), 0 1px 1px rgba(58,36,22,0.04);
        --shadow-md: 0 6px 20px -4px rgba(58,36,22,0.14), 0 2px 6px rgba(58,36,22,0.06);
        --shadow-lift: 0 12px 28px -6px rgba(58,36,22,0.20);
        --radius: 14px;
        --radius-sm: 10px;
        --ease: cubic-bezier(.22,.61,.36,1);
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { animation-duration: 0.001ms !important; transition-duration: 0.001ms !important; }
    }

    html, body, [class*="css"] {
        font-family: 'Manrope', -apple-system, sans-serif;
    }
    [data-testid="stAppViewContainer"] {
        background: var(--clr-bg);
    }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    section[data-testid="stSidebar"] {
        background: var(--clr-card);
        border-right: 1px solid var(--clr-border);
    }

    /* ── Hero header ─────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, var(--clr-terracotta), var(--clr-amber));
        padding: 30px 32px;
        border-radius: 20px;
        margin-bottom: 26px;
        box-shadow: var(--shadow-md);
        position: relative;
        overflow: hidden;
    }
    .main-header::after {
        content: "";
        position: absolute; inset: 0;
        background: radial-gradient(circle at 88% -10%, rgba(255,255,255,0.20), transparent 55%);
        pointer-events: none;
    }
    .main-header h1 {
        font-family: 'Fraunces', serif;
        color: white;
        font-size: 30px;
        font-weight: 600;
        letter-spacing: -0.01em;
        margin: 0;
    }
    .main-header p {
        color: rgba(255,255,255,0.88);
        font-size: 14px;
        margin: 6px 0 0;
        max-width: 640px;
    }

    /* ── Section labels ──────────────────────────── */
    .section-title {
        font-size: 11.5px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--clr-label);
        margin-bottom: 12px;
        margin-top: 20px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-title::before {
        content: "";
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--clr-amber);
        display: inline-block;
    }

    /* ── Info cards ──────────────────────────────── */
    .info-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-bottom: 18px;
    }
    .info-card {
        background: var(--clr-card);
        border: 1px solid var(--clr-border);
        border-radius: var(--radius-sm);
        padding: 14px 16px;
        text-align: center;
        box-shadow: var(--shadow-sm);
        transition: transform 0.2s var(--ease), box-shadow 0.2s var(--ease), border-color 0.2s var(--ease);
    }
    .info-card:hover {
        border-color: var(--clr-amber);
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }
    .info-card .label {
        font-size: 10.5px;
        color: var(--clr-label);
        margin: 0 0 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
    }
    .info-card .value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 14px;
        font-weight: 600;
        color: var(--clr-ink);
        margin: 0;
        letter-spacing: -0.01em;
    }

    /* ── Feature / incident cards ────────────────── */
    .feature-card {
        background: var(--clr-card-alt);
        border: 1px solid var(--clr-border);
        border-left: 3px solid var(--clr-amber);
        border-radius: var(--radius-sm);
        padding: 14px 16px;
        margin-bottom: 10px;
        box-shadow: var(--shadow-sm);
        transition: transform 0.2s var(--ease);
    }
    .feature-card:hover { transform: translateX(2px); }
    .feature-title {
        font-size: 13px;
        font-weight: 700;
        color: var(--clr-ink);
        margin: 0 0 4px;
    }
    .feature-desc {
        font-size: 12.5px;
        color: var(--clr-muted);
        margin: 0;
        line-height: 1.55;
    }

    /* ── Meters ───────────────────────────────────── */
    .meter-wrap { display: flex; gap: 6px; margin: 8px 0 4px; }
    .meter-seg { flex: 1; height: 8px; border-radius: 99px; }
    .meter-labels {
        display: flex; justify-content: space-between;
        font-size: 11px; color: var(--clr-label); margin-bottom: 16px;
    }

    /* ── Result banners (severity semantics kept) ─── */
    .result-low, .result-mid, .result-high {
        border-radius: var(--radius); padding: 16px 20px; margin: 14px 0;
        box-shadow: var(--shadow-sm);
    }
    .result-low  { background: #EDF3DC; border: 1px solid #B7D488; }
    .result-mid  { background: #FBEBD4; border: 1px solid #F0B458; }
    .result-high { background: #FBE5DD; border: 1px solid #E8916F; }
    .result-low .title  { color: #365E14; font-weight: 700; font-size: 15.5px; margin: 0 0 4px; }
    .result-mid .title  { color: #7A4B0A; font-weight: 700; font-size: 15.5px; margin: 0 0 4px; }
    .result-high .title { color: #96341B; font-weight: 700; font-size: 15.5px; margin: 0 0 4px; }
    .result-low .desc   { color: #4A7A1E; font-size: 13px; margin: 0; }
    .result-mid .desc   { color: #A0680F; font-size: 13px; margin: 0; }
    .result-high .desc  { color: #B94E2C; font-size: 13px; margin: 0; }

    /* ── Buttons ──────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, var(--clr-terracotta), var(--clr-amber)) !important;
        color: white !important;
        border: none !important;
        border-radius: 999px !important;
        padding: 11px 26px !important;
        font-size: 14.5px !important;
        font-weight: 600 !important;
        width: 100% !important;
        box-shadow: var(--shadow-sm) !important;
        transition: transform 0.15s var(--ease), box-shadow 0.15s var(--ease), filter 0.15s var(--ease) !important;
    }
    .stButton > button:hover {
        filter: brightness(1.06) !important;
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-md) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }
    .stButton > button:focus-visible {
        outline: 2px solid var(--clr-ink) !important;
        outline-offset: 2px !important;
    }

    /* ── Tabs — segmented pill control ───────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: var(--clr-card);
        border: 1px solid var(--clr-border);
        border-radius: 999px;
        padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        color: var(--clr-muted);
        font-weight: 600;
        font-size: 13.5px;
        padding: 8px 18px;
        transition: all 0.2s var(--ease);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--clr-terracotta), var(--clr-amber)) !important;
        color: white !important;
        box-shadow: var(--shadow-sm);
    }
    .stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }
    .stTabs [data-baseweb="tab-border"] { background: transparent !important; }

    /* ── Chat ─────────────────────────────────────── */
    .chat-header {
        background: linear-gradient(135deg, var(--clr-terracotta), var(--clr-amber));
        color: white;
        padding: 14px 18px;
        border-radius: var(--radius) var(--radius) 0 0;
        font-size: 14px;
        font-weight: 600;
        box-shadow: var(--shadow-sm);
    }
    [data-testid="stChatMessage"] {
        background: var(--clr-card) !important;
        border: 1px solid var(--clr-border) !important;
        border-radius: var(--radius-sm) !important;
        box-shadow: var(--shadow-sm);
        margin-bottom: 8px;
    }
    [data-testid="stChatInput"] {
        border-radius: 999px !important;
        border: 1px solid var(--clr-border) !important;
    }

    /* ── Alerts — recolor to warm palette ────────── */
    [data-testid="stAlert"] {
        border-radius: var(--radius-sm) !important;
        box-shadow: var(--shadow-sm);
    }

    /* ── About box ────────────────────────────────── */
    .about-box {
        background: var(--clr-card);
        border: 1px solid var(--clr-border);
        border-radius: var(--radius);
        padding: 18px 22px;
        margin-top: 22px;
        box-shadow: var(--shadow-sm);
    }
    .about-box h4 {
        font-family: 'Fraunces', serif;
        font-size: 15px;
        font-weight: 600;
        color: var(--clr-ink);
        margin: 0 0 8px;
    }
    .about-box p {
        font-size: 12.5px;
        color: var(--clr-muted);
        margin: 0;
        line-height: 1.65;
    }

    /* ── Hero stat box (departure time etc.) ─────── */
    .risk-score-box {
        background: linear-gradient(135deg, #8C3A16, var(--clr-amber));
        border-radius: var(--radius);
        padding: 22px;
        text-align: center;
        color: white;
        margin-bottom: 16px;
        box-shadow: var(--shadow-md);
    }
    .risk-score-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 46px;
        font-weight: 700;
        margin: 0;
        line-height: 1;
        letter-spacing: -0.02em;
    }
    .risk-score-label {
        font-size: 13px;
        opacity: 0.88;
        margin: 6px 0 0;
        font-weight: 500;
    }

    /* ── Risk gauge — signature element ──────────── */
    .gauge-wrap {
        display: flex; align-items: center; justify-content: center;
        gap: 22px;
        background: var(--clr-card);
        border: 1px solid var(--clr-border);
        border-radius: var(--radius);
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: var(--shadow-md);
    }
    .gauge-ring {
        width: 108px; height: 108px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
        transition: background 0.6s var(--ease);
    }
    .gauge-ring-inner {
        width: 84px; height: 84px;
        border-radius: 50%;
        background: var(--clr-card);
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        box-shadow: inset 0 1px 3px rgba(58,36,22,0.08);
    }
    .gauge-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 26px; font-weight: 700; color: var(--clr-ink);
        line-height: 1;
    }
    .gauge-max { font-size: 10px; color: var(--clr-label); margin-top: 2px; }
    .gauge-text { text-align: left; }
    .gauge-label {
        font-size: 11px; font-weight: 700; letter-spacing: 0.06em;
        text-transform: uppercase; color: var(--clr-label); margin: 0 0 4px;
    }
    .gauge-level {
        font-family: 'Fraunces', serif;
        font-size: 22px; font-weight: 600; margin: 0;
    }

    /* ── Profile box ──────────────────────────────── */
    .profile-box {
        background: linear-gradient(135deg, var(--clr-ink), #6B4226);
        border-radius: var(--radius);
        padding: 18px 22px;
        color: white;
        margin-bottom: 16px;
        box-shadow: var(--shadow-md);
    }
    .profile-box h4 {
        font-family: 'Fraunces', serif;
        font-size: 15px; font-weight: 600; margin: 0 0 8px; color: white;
    }
    .profile-box p { font-size: 12.5px; opacity: 0.88; margin: 0 0 4px; color: white; }

    .forecast-card {
        background: var(--clr-card);
        border: 1px solid var(--clr-border);
        border-radius: var(--radius-sm);
        padding: 10px 12px;
        text-align: center;
        margin-bottom: 6px;
        box-shadow: var(--shadow-sm);
        transition: transform 0.2s var(--ease);
    }
    .forecast-card:hover { transform: translateY(-2px); }

    hr { border-color: var(--clr-border) !important; }

    /* ── Safety net: force readable text on built-in Streamlit
       widgets. These normally inherit color from Streamlit's theme;
       without an explicit light theme they default to white text,
       which disappears on the light cards above. ─────────────── */
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricDelta"],
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] p,
    label, .stCheckbox label, .stRadio label,
    .stSlider label, .stNumberInput label,
    .stTextInput label, .stSelectbox label {
        color: var(--clr-ink) !important;
    }
    /* Sidebar text follows the same ink color */
    section[data-testid="stSidebar"] * { color: var(--clr-ink); }
    /* Alert boxes (info/success/warning/error) keep their tinted
       backgrounds but force readable dark text over them */
    [data-testid="stAlert"] p, [data-testid="stAlert"] div,
    [data-testid="stAlert"] span {
        color: var(--clr-ink) !important;
    }
    /* Dataframe / table text */
    [data-testid="stDataFrame"] * { color: var(--clr-ink) !important; }

    /* Dataframe toolbar (search/download/fullscreen icons) — this
       floats above the table and was rendering with a dark background
       + white icons regardless of theme. Target the stable testid
       (robust across Streamlit versions) plus the exact class from
       the screenshot as a redundant fallback — note that
       "st-emotion-cache-*" hashes are auto-generated and can change
       between Streamlit releases, so the testid rule is the one doing
       the real work long-term. */
    [data-testid="stElementToolbar"],
    div.stElementToolbar {
        background: var(--clr-card) !important;
        border: 1px solid var(--clr-border) !important;
        border-radius: var(--radius-sm) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    [data-testid="stElementToolbar"] button,
    div.stElementToolbar button {
        background: transparent !important;
        color: var(--clr-ink) !important;
    }
    [data-testid="stElementToolbar"] button:hover,
    div.stElementToolbar button:hover {
        background: var(--clr-card-alt) !important;
    }
    [data-testid="stElementToolbar"] svg,
    div.stElementToolbar svg {
        fill: var(--clr-ink) !important;
        color: var(--clr-ink) !important;
    }
    /* The search box that pops out of the toolbar */
    [data-testid="stElementToolbar"] input {
        background: var(--clr-card) !important;
        color: var(--clr-ink) !important;
        border-color: var(--clr-border) !important;
    }

    /* Text input / select box value text and their dropdown menus */
    .stTextInput input, .stNumberInput input {
        color: var(--clr-ink) !important;
        background: var(--clr-card) !important;
    }
    [data-baseweb="select"] * { color: var(--clr-ink) !important; }
    [data-baseweb="popover"] { background: var(--clr-card) !important; }
    [data-baseweb="menu"] li { color: var(--clr-ink) !important; }

    /* Elements that intentionally sit on a colored/gradient
       background must stay white — re-assert after the safety net
       above so specificity doesn't flip them dark. */
    .main-header h1, .main-header p,
    .stButton > button,
    .stTabs [aria-selected="true"],
    .chat-header,
    .risk-score-num, .risk-score-label,
    .profile-box, .profile-box h4, .profile-box p {
        color: white !important;
    }
</style>""", unsafe_allow_html=True)

# ── Load ML model ─────────────────────────────────
try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception:
    st.error("Model file not found. Make sure model.pkl is in the project folder.")
    st.stop()

# ── Gemini AI setup ───────────────────────────────
def init_gemini():
    """Set up Gemini — no cache so new API key is always picked up."""
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
   preferred = [
        "models/gemini-2.5-flash-lite",
        "models/gemini-2.0-flash-001",
        "models/gemini-2.0-flash-lite-001",
        "models/gemini-2.0-flash-lite",
        "models/gemini-2.5-flash-preview-tts",
    ]
    available = [m.name for m in genai.list_models()
                 if 'generateContent' in m.supported_generation_methods]
    for name in preferred:
        if name in available:
            return genai.GenerativeModel(name)
    if available:
        return genai.GenerativeModel(available[0])
    raise RuntimeError("No Gemini model supports generateContent")

try:
    gemini = init_gemini()
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
        st.session_state['sheets_error'] = None
        return sheet
    except Exception as e:
        st.session_state['sheets_error'] = f"{type(e).__name__}: {e}"
        return None


def save_trip_to_sheet(sheet, data):
    """Save a trip record to Google Sheets. Returns the 1-indexed row
    number the trip was written to, so feedback can target that exact
    row later instead of recomputing "last row" (which can collide if
    another user appends a trip in between)."""
    try:
        existing = sheet.get_all_values()
        if not existing:
            headers = [
                "Timestamp", "Location", "Destination",
                "Weather", "Temperature", "Wind Speed",
                "Light", "Surface", "Area", "Road Type",
                "Risk Score", "Prediction", "Confidence",
                "Distance (km)", "Duration (mins)",
                "Hours Driving", "Feedback"
            ]
            sheet.append_row(headers)
            existing = [headers]
        sheet.append_row(data)
        return len(existing) + 1  # row index of the row we just wrote
    except Exception:
        return None


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


@st.cache_data(ttl=300, show_spinner=False)
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


@st.cache_data(ttl=300, show_spinner=False)
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


@st.cache_data(ttl=3600, show_spinner=False)
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


@st.cache_data(ttl=300, show_spinner=False)
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
            gauge_color = ('#4A7A1E' if risk_score < 35
                           else '#C77A0C' if risk_score < 65
                           else '#B94E2C')
            gauge_deg = round(risk_score / 100 * 360)
            st.markdown(f"""
            <div class="gauge-wrap">
                <div class="gauge-ring" style="background: conic-gradient(
                    {gauge_color} {gauge_deg}deg, #EAD3AA {gauge_deg}deg)">
                    <div class="gauge-ring-inner">
                        <p class="gauge-num">{risk_score}</p>
                        <p class="gauge-max">/ 100</p>
                    </div>
                </div>
                <div class="gauge-text">
                    <p class="gauge-label">Current condition</p>
                    <p class="gauge-level" style="color:{gauge_color}">{risk_label_text}</p>
                </div>
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
                        st.session_state['trip_row'] = save_trip_to_sheet(sheet, [
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
                    key="persistent_map", returned_objects=[])

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
                        row = st.session_state.get('trip_row')
                        if sheet and row:
                            try:
                                sheet.update_cell(row, 17, 'Accurate')
                                st.success(
                                    "Thanks! Feedback saved ✅")
                            except Exception:
                                st.success("Thanks for feedback!")
                        else:
                            st.success("Thanks for feedback!")
                with col_n:
                    if st.button("👎 No, inaccurate"):
                        sheet = get_sheet()
                        row = st.session_state.get('trip_row')
                        if sheet and row:
                            try:
                                sheet.update_cell(row, 17, 'Inaccurate')
                                st.info(
                                    "Thanks! We'll use this "
                                    "to improve the model.")
                            except Exception:
                                st.info("Thanks for feedback!")
                        else:
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
                          key="default_map", returned_objects=[])
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
        err = st.session_state.get('sheets_error')
        if err:
            with st.expander("⚙️ Debug info (error detail)"):
                st.code(err)
        with st.expander("✅ Checklist to fix this"):
            st.markdown("""
1. **Secrets exist** — your Streamlit secrets must have both
   `GCP_CREDENTIALS` (the full service account JSON, as a string)
   and `SHEET_ID` (the spreadsheet ID from its URL).
2. **`GCP_CREDENTIALS` is valid JSON on one line** — paste the
   entire service account key file content, keep the `\\n` inside
   `private_key` escaped, and wrap the whole thing in triple quotes
   in `secrets.toml`, e.g.:
   ```
   GCP_CREDENTIALS = '''{"type": "service_account", ...}'''
   SHEET_ID = "1AbCдефID_from_your_sheet_url"
   ```
3. **The sheet is shared with the service account** — open the
   service account JSON, copy the `client_email` value
   (looks like `xyz@project.iam.gserviceaccount.com`), then share
   your Google Sheet with that email as **Editor**.
4. **APIs are enabled** — in Google Cloud Console, both the
   **Google Sheets API** and **Google Drive API** must be enabled
   for the project that owns the service account.
5. **`SHEET_ID` is correct** — it's the long string between
   `/d/` and `/edit` in the sheet's URL, not the sheet name.
            """)

# ════════════════════════════════════════════════════
# TAB 3 — JOURNEY PLANNER
# ════════════════════════════════════════════════════
with tab3:
    st.markdown(
        '<p class="section-title">'
        '🌦 24-hour weather risk forecast</p>',
        unsafe_allow_html=True)

    location2 = get_geolocation(component_key="tab3_location")

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
    '✨ AI Assistant — Powered by Gemini'
    '</div>',
    unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize Gemini chat session
if "gemini_chat" not in st.session_state:
    st.session_state.gemini_chat = gemini.start_chat(history=[])

# Clear chat button
col_chat1, col_chat2 = st.columns([5, 1])
with col_chat2:
    if st.button("🗑 Clear", key="clear_chat"):
        st.session_state.messages = []
        st.session_state.gemini_chat = gemini.start_chat(history=[])
        st.rerun()

# Show all previous messages with markdown rendering
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Message AI Assistant...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({
        "role": "user", "content": user_input
    })

    with st.chat_message("assistant"):
        with st.spinner(""):
            try:
                # Use Gemini chat session — keeps full conversation memory
                response = st.session_state.gemini_chat.send_message(
                    user_input,
                    generation_config={
                        "temperature": 0.9,
                        "top_p": 0.95,
                        "max_output_tokens": 2048,
                    }
                )
                bot_reply = response.text

            except Exception as e:
                err = str(e)
                if "429" in err or "quota" in err.lower():
                    bot_reply = (
                        "⏳ **AI quota limit reached.**\n\n"
                        "The free Gemini API resets every 24 hours. "
                        "Please try again later."
                    )
                else:
                    # Fallback with fresh chat session
                    try:
                        fallback_history = []
                        for m in st.session_state.messages[:-1]:
                            role = "user" if m["role"] == "user" else "model"
                            fallback_history.append({
                                "role": role,
                                "parts": [m["content"]]
                            })
                        fresh_chat = gemini.start_chat(
                            history=fallback_history)
                        fallback_resp = fresh_chat.send_message(
                            user_input,
                            generation_config={
                                "temperature": 0.9,
                                "max_output_tokens": 2048,
                            }
                        )
                        bot_reply = fallback_resp.text
                        st.session_state.gemini_chat = fresh_chat
                    except Exception as e2:
                        err2 = str(e2)
                        if "429" in err2 or "quota" in err2.lower():
                            bot_reply = (
                                "⏳ **Quota limit reached.** "
                                "Please try again in a few hours."
                            )
                        else:
                            bot_reply = f"⚠️ Error: {err2[:200]}"

        # Render as markdown — bold, lists, code blocks like Gemini
        st.markdown(bot_reply)

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
