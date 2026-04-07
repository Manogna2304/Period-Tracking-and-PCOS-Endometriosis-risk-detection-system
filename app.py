import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from cycle_predictor import CyclePredictor
from risk_model import HealthRiskModel
from symptom_cluster import SymptomClusterer, CLUSTER_PROFILES

# ── TIME ─────────────────────────────────────────
def get_local_date():
    return datetime.now().astimezone().date()

# ── PAGE CONFIG ─────────────────────────────────
st.set_page_config(
    page_title="Luna · Period Tracker",
    page_icon="🌙",
    layout="wide",
)

st.markdown("""
<style>

/* Fonts */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #3d1a47;
}

/* 🌸 Background Gradient */
.stApp {
    background: linear-gradient(135deg, #fdf6f0 0%, #fce4ec 50%, #f3e5f5 100%);
}

/* 🌙 Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #3d1a47 0%, #6b2d6b 100%);
}
[data-testid="stSidebar"] * {
    color: #e0c3fc !important;
}

/* Headings */
h1 {
    font-family: 'DM Serif Display', serif;
    color: #3d1a47;
}
h2, h3 {
    font-family: 'DM Serif Display', serif;
    color: #6b2d6b;
}

/* Metric cards look better */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #f3e5f5, #fce4ec);
    padding: 10px;
    border-radius: 12px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #f3e5f5, #fce4ec);
    color: #3d1a47;
    border-radius: 10px;
    border: 1px solid rgba(180,100,200,0.4);
}

/* Inputs */
label, .stSlider label, .stCheckbox label {
    color: #3d1a47 !important;
}

/* Small text */
small, .stCaption {
    color: #8e3a9d !important;
}

/* Remove header/footer */
#MainMenu, footer {visibility: hidden;}
header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# ── SAFE INIT ───────────────────────────────────
def init_models():
    if "predictor" not in st.session_state:
        st.session_state.predictor = CyclePredictor()

    if "clusterer" not in st.session_state:
        c = SymptomClusterer()
        c.fit()
        st.session_state.clusterer = c

    if "risk_model" not in st.session_state:
        try:
            st.session_state.risk_model = HealthRiskModel()
        except Exception as e:
            st.session_state.risk_model = None
            st.error(f"⚠️ Model failed to load: {e}")

init_models()

# ── SESSION DEFAULTS ────────────────────────────
if "period_dates" not in st.session_state:
    today = get_local_date()
    st.session_state.period_dates = [
        today - timedelta(days=160),
        today - timedelta(days=132),
        today - timedelta(days=102),
        today - timedelta(days=73),
        today - timedelta(days=42),
        today - timedelta(days=14)
    ]

if "symptom_logs" not in st.session_state:
    st.session_state.symptom_logs = []

# ── SIDEBAR ────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌙 Luna")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📅 Cycle Prediction", "🔬 Health Risk Check", "🌸 Symptom Patterns"]
    )

# ═══════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════
if page == "🏠 Home":

    st.markdown("# 🌙 Luna · Period Tracker")
    st.markdown("*Track your cycle, understand your body, powered by ML*")

    predictor = st.session_state.predictor
    dates = st.session_state.period_dates

    result = predictor.predict_next(dates)

    # ✅ FIXED LOGIC BLOCK
    next_period = result.get("next_predicted_date")

    if next_period is None:
        next_period = get_local_date()
        days_until = 0
    else:
        if isinstance(next_period, datetime):
            next_period = next_period.date()

        days_until = (next_period - get_local_date()).days

    # ✅ UI ALWAYS OUTSIDE IF/ELSE
    col1, col2, col3 = st.columns(3)

    col1.metric("Avg Cycle", result.get("avg_cycle", 28))
    col2.metric("Next Cycle Length", result.get("predicted_length", 28))

    label = (
        "Today" if days_until == 0 else
        f"in {days_until} days" if days_until > 0 else
        f"{abs(days_until)} days ago"
    )

    col3.metric("Next Period", label)

    st.markdown("---")

    # ── LOG SYMPTOMS ──
    st.subheader("Log Symptoms")

    col1, col2 = st.columns(2)

    with col1:
        cramps = st.checkbox("Cramps")
        fatigue = st.checkbox("Fatigue")
        nausea = st.checkbox("Nausea")

    with col2:
        mood_swings = st.checkbox("Mood Swings")
        acne = st.checkbox("Acne")
        back_pain = st.checkbox("Back Pain")

    flow = st.slider("Flow", 0, 5, 2)
    pain = st.slider("Pain", 0, 5, 2)

    if st.button("Save Log"):
        log = {
            "cramps": cramps,
            "fatigue": fatigue,
            "nausea": nausea,
            "mood_swings": mood_swings,
            "acne": acne,
            "back_pain": back_pain,
            "flow_intensity": flow,
            "pain_level": pain,
        }

        st.session_state.symptom_logs.append(log)

        try:
            cluster = st.session_state.clusterer.predict_day(log)
            st.success(f"{cluster['emoji']} {cluster['name']} day")
        except Exception as e:
            st.error(f"Clustering failed: {e}")

# ═══════════════════════════════════════════════
# CYCLE PREDICTION
# ═══════════════════════════════════════════════
elif page == "📅 Cycle Prediction":

    st.markdown("# 📅 Cycle Prediction")

    df = pd.DataFrame({"Date": st.session_state.period_dates})
    edited = st.data_editor(df)

    if st.button("Update"):
        try:
            new_dates = pd.to_datetime(edited["Date"]).dt.date.dropna().tolist()
            st.session_state.period_dates = sorted(new_dates)
            st.success("Updated!")
        except Exception as e:
            st.error(f"Invalid dates: {e}")

    predictor = st.session_state.predictor
    result = predictor.predict_next(st.session_state.period_dates)

    st.write(result)

# ═══════════════════════════════════════════════
# HEALTH RISK
# ═══════════════════════════════════════════════
elif page == "🔬 Health Risk Check":

    st.markdown("# 🔬 Health Risk")

    if st.session_state.risk_model is None:
        st.error("Model not loaded")
        st.stop()

    age = st.number_input("Age", 16, 50, 22)

    col1, col2 = st.columns(2)
    with col1:
        weight = st.number_input("Weight", 30.0, 120.0, 55.0)
    with col2:
        height = st.number_input("Height", 140.0, 200.0, 160.0)

    cycle_len = st.number_input("Cycle Length", 20, 60, 28)

    st.subheader("Symptoms")

    irregular = st.checkbox("Irregular cycles")
    weight_gain = st.checkbox("Weight gain")
    acne = st.checkbox("Acne")
    pelvic_pain = st.checkbox("Pelvic pain")
    heavy_bleed = st.checkbox("Heavy bleeding")

    if st.button("Analyze"):

        user_input = {
            "age": age,
            "weight_kg": weight,
            "height_cm": height,
            "cycle_length": cycle_len,
            "cycle_irregular": irregular,
            "weight_gain": weight_gain,
            "pimples": acne,
            "pelvic_pain": pelvic_pain,
            "heavy_bleeding": heavy_bleed,
        }

        try:
            result = st.session_state.risk_model.predict_risk(user_input)

            st.success("Done")

            st.write("PCOS:", result["PCOS"])
            st.write("Endometriosis:", result["Endometriosis"])

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ═══════════════════════════════════════════════
# SYMPTOM PATTERNS
# ═══════════════════════════════════════════════
elif page == "🌸 Symptom Patterns":

    st.markdown("# 🌸 Symptom Patterns")

    logs = st.session_state.symptom_logs

    if len(logs) < 3:
        st.warning("Need at least 3 logs")
    else:
        try:
            summary = st.session_state.clusterer.get_pattern_summary(logs)
            st.write(summary)
        except Exception as e:
            st.error(f"Pattern analysis failed: {e}")
