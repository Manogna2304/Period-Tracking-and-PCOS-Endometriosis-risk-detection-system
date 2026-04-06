import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from cycle_predictor import CyclePredictor
from risk_model import PCOSRiskModel
from symptom_cluster import SymptomClusterer, CLUSTER_PROFILES
# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Luna · Period Tracker",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #fdf6f0 0%, #fce4ec 50%, #f3e5f5 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #3d1a47 0%, #6b2d6b 100%);
    border-right: none;
}
[data-testid="stSidebar"] * { color: #e0c3fc !important; }
[data-testid="stSidebar"] .stRadio label { color: #e0c3fc !important; font-size: 1rem; }

/* Headers */
h1 { font-family: 'DM Serif Display', serif !important; color: #3d1a47 !important; font-size: 2.6rem !important; }
h2 { font-family: 'DM Serif Display', serif !important; color: #6b2d6b !important; }
h3 { font-family: 'DM Serif Display', serif !important; color: #8e3a9d !important; }

/* Cards */
.luna-card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 1.5rem 2rem;
    margin: 0.75rem 0;
    border: 1px solid rgba(180,100,200,0.15);
    box-shadow: 0 4px 24px rgba(109,45,107,0.08);
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #7b1fa2, #c2185b);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    color: white;
    text-align: center;
    box-shadow: 0 6px 20px rgba(123,31,162,0.3);
}
.metric-card .value { font-size: 2.4rem; font-weight: 700; font-family: 'DM Serif Display', serif; }
.metric-card .label { font-size: 0.85rem; opacity: 0.85; letter-spacing: 0.05em; text-transform: uppercase; }

/* Risk badge */
.risk-low    { background: #e8f5e9; color: #2e7d32; border: 2px solid #66bb6a; border-radius: 12px; padding: 0.5rem 1.2rem; font-weight: 600; display: inline-block; }
.risk-moderate { background: #fff8e1; color: #e65100; border: 2px solid #ffa726; border-radius: 12px; padding: 0.5rem 1.2rem; font-weight: 600; display: inline-block; }
.risk-high   { background: #ffebee; color: #b71c1c; border: 2px solid #ef5350; border-radius: 12px; padding: 0.5rem 1.2rem; font-weight: 600; display: inline-block; }

/* Cluster pill */
.cluster-pill {
    display: inline-block;
    border-radius: 50px;
    padding: 0.35rem 1rem;
    font-size: 0.9rem;
    font-weight: 600;
    color: white;
    margin: 0.2rem;
}

/* Table text visibility */
.luna-card table td, .luna-card table th {
    color: #3d1a47 !important;
    font-weight: 500;
}

/* Checkbox and widget labels */
.stCheckbox label, 
[data-testid="stCheckbox"] label p {
    color: #3d1a47 !important;
    font-weight: 500 !important;
}

/* Slider labels */
.stSlider label, 
[data-testid="stSlider"] label {
    color: #3d1a47 !important;
    font-weight: 500 !important;
}

/* General form label visibility */
[data-testid="stWidgetLabel"] p,
.stSelectSlider label,
label[data-testid] {
    color: #3d1a47 !important;
    font-weight: 500 !important;
}

/* Subtitle / italic markdown text */
p em, .stMarkdown p em {
    color: #6b2d6b !important;
    font-style: italic;
    font-weight: 400;
}



/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7b1fa2, #c2185b) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(123,31,162,0.4) !important;
}

/* Input fields */
.stNumberInput input, .stSelectbox select, .stSlider {
    border-radius: 10px !important;
}

/* Divider */
hr { border-color: rgba(180,100,200,0.2) !important; }

/* Warning / info boxes */
.stAlert { border-radius: 14px !important; }

/* Hide streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
if "cycle_lengths" not in st.session_state:
    st.session_state.cycle_lengths = [28, 30, 27, 29, 31, 28]
if "last_period_date" not in st.session_state:
    st.session_state.last_period_date = date.today() - timedelta(days=14)
if "symptom_logs" not in st.session_state:
    st.session_state.symptom_logs = []
if "predictor" not in st.session_state:
    st.session_state.predictor = CyclePredictor()
if "risk_model" not in st.session_state:
    st.session_state.risk_model = PCOSRiskModel()
if "clusterer" not in st.session_state:
    c = SymptomClusterer()
    c.fit()
    st.session_state.clusterer = c


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌙 Luna")
    st.markdown("*Your intelligent cycle companion*")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📅 Cycle Prediction", "🔬 Health Risk Check", "🌸 Symptom Patterns"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**ML Models Used**")
    st.markdown("📈 Linear Regression")
    st.markdown("🧬 Logistic Regression")
    st.markdown("🔵 K-Means Clustering")
    st.markdown("---")
    st.caption("⚠️ This app is for educational purposes only. Not a medical device.")


# ═══════════════════════════════════════════════════════════
# PAGE 1: HOME
# ═══════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("# 🌙 Luna · Period Tracker")
    st.markdown("*Track your cycle, understand your body, powered by ML*")
    st.markdown("---")

    # Quick stats
    predictor = st.session_state.predictor
    result = predictor.predict_next(st.session_state.cycle_lengths)
    next_period = st.session_state.last_period_date + timedelta(days=result["predicted_length"])
    days_until = (next_period - date.today()).days

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{result['avg_cycle']}</div>
            <div class="label">Avg Cycle (days)</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{result['predicted_length']}</div>
            <div class="label">Next Cycle Length</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        label = f"in {days_until}d" if days_until > 0 else "Today" if days_until == 0 else f"{abs(days_until)}d ago"
        st.markdown(f"""<div class="metric-card">
            <div class="value">{label}</div>
            <div class="label">Next Period</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{len(st.session_state.cycle_lengths)}</div>
            <div class="label">Cycles Logged</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two columns: log + calendar
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.markdown("### 📝 Log Today's Symptoms")
        with st.container():
            st.markdown('<div class="luna-card">', unsafe_allow_html=True)
            log_date = st.date_input("Date", value=date.today())
            col_a, col_b = st.columns(2)
            with col_a:
                cramps     = st.checkbox("🤕 Cramps")
                bloating   = st.checkbox("😮 Bloating")
                headache   = st.checkbox("🤯 Headache")
                fatigue    = st.checkbox("😴 Fatigue")
                nausea     = st.checkbox("🤢 Nausea")
            with col_b:
                mood_swings = st.checkbox("😤 Mood Swings")
                acne        = st.checkbox("😓 Acne")
                back_pain   = st.checkbox("🔙 Back Pain")

            flow = st.select_slider("Flow Intensity", options=[0, 1, 2, 3, 4, 5],
                                    format_func=lambda x: ["None","Spotting","Light","Moderate","Heavy","Very Heavy"][x])
            pain = st.slider("Pain Level", 0, 5, 0)
            energy = st.slider("Energy Level", 0, 5, 3)
            mood = st.slider("Mood Score", 0, 5, 3)

            if st.button("✅ Save Log"):
                log = {
                    "date": str(log_date),
                    "cramps": cramps, "bloating": bloating, "headache": headache,
                    "fatigue": fatigue, "nausea": nausea, "mood_swings": mood_swings,
                    "acne": acne, "back_pain": back_pain,
                    "flow_intensity": flow, "pain_level": pain,
                    "energy_level": energy, "mood_score": mood,
                }
                st.session_state.symptom_logs.append(log)

                # Predict today's cluster
                cluster_result = st.session_state.clusterer.predict_day(log)
                st.success(f"Logged! Today looks like a **{cluster_result['name']}** day {cluster_result['emoji']}")
                st.caption(cluster_result["tip"])
            st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown("### 📆 Cycle Calendar")
        with st.container():
            st.markdown('<div class="luna-card">', unsafe_allow_html=True)

            # Show next 3 months as a visual timeline
            today = date.today()
            last = st.session_state.last_period_date
            avg_len = int(result["avg_cycle"])
            pred_len = result["predicted_length"]

            # Generate upcoming period windows
            periods = []
            current = last
            for _ in range(4):
                periods.append((current, current + timedelta(days=5)))
                current = current + timedelta(days=avg_len)

            fig = go.Figure()

            # Timeline bar
            for i, (start, end) in enumerate(periods):
                color = "#c2185b" if i == 0 else "#e91e8c" if i == 1 else "#f48fb1"
                label = "Last Period" if i == 0 else f"Predicted #{i}"
                fig.add_trace(go.Bar(
                    x=[5], y=[start.strftime("%b %d")],
                    orientation="h",
                    marker_color=color,
                    name=label,
                    hovertemplate=f"{label}: {start.strftime('%b %d')} – {end.strftime('%b %d')}<extra></extra>",
                ))

            # Today marker
            fig.add_vline(x=0, line_color="#7b1fa2", line_dash="dot", annotation_text="Today")

            fig.update_layout(
    height=260,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    showlegend=True,
    legend=dict(
        orientation="h",
        y=-0.2,
        font=dict(color="#3d1a47", family="DM Sans")
    ),
    margin=dict(l=10, r=10, t=10, b=10),
    barmode="overlay",
    xaxis=dict(visible=False),
    yaxis=dict(
        tickfont=dict(color="#3d1a47", family="DM Sans"),
        gridcolor="rgba(180,100,200,0.1)",
    ),
    font=dict(family="DM Sans", color="#3d1a47"),
)
            
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
<table style="width:100%; color:#3d1a47; font-size:0.95rem;">
  <tr><td>🔴 <b>Last period started</b></td><td style="color:#6b2d6b; font-weight:600;">{st.session_state.last_period_date.strftime('%d %b %Y')}</td></tr>
  <tr><td>🔮 <b>Next period predicted</b></td><td style="color:#6b2d6b; font-weight:600;">{next_period.strftime('%d %b %Y')}</td></tr>
  <tr><td>📏 <b>Predicted cycle length</b></td><td style="color:#6b2d6b; font-weight:600;">{pred_len} days</td></tr>
  <tr><td>📊 <b>Confidence</b></td><td style="color:#6b2d6b; font-weight:600;">{result['confidence'].capitalize()}</td></tr>
</table>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE 2: CYCLE PREDICTION
# ═══════════════════════════════════════════════════════════
elif page == "📅 Cycle Prediction":
    st.markdown("# 📅 Cycle Prediction")
    st.markdown("*Linear Regression trained on your logged cycle history*")
    st.markdown("---")

    col_form, col_result = st.columns([1, 1.4])

    with col_form:
        st.markdown("### Your Cycle History")
        st.markdown('<div class="luna-card">', unsafe_allow_html=True)

        st.markdown("**Last period start date**")
        last_date = st.date_input("Last Period", value=st.session_state.last_period_date,
                                   label_visibility="collapsed")
        st.session_state.last_period_date = last_date

        st.markdown("**Past cycle lengths (days)**")
        st.caption("Enter your recent cycles separated by commas (oldest → newest)")
        raw = st.text_input("Cycles", value=", ".join(map(str, st.session_state.cycle_lengths)),
                             label_visibility="collapsed")
        try:
            parsed = [int(x.strip()) for x in raw.split(",") if x.strip()]
            parsed = [c for c in parsed if 15 <= c <= 60]
        except:
            parsed = st.session_state.cycle_lengths

        if st.button("🔮 Predict Next Period"):
            st.session_state.cycle_lengths = parsed
            st.session_state.predictor = CyclePredictor()
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # Model explanation
        st.markdown("### How it works")
        st.markdown('<div class="luna-card">', unsafe_allow_html=True)
        st.markdown("""
**Features used:**
- Previous 3 cycle lengths
- Rolling 3 & 6-cycle averages
- Standard deviation (regularity)
- Trend (increasing/decreasing)

**Model:** `sklearn.LinearRegression`

The model self-trains on your own cycle history, improving predictions as you log more cycles. Confidence is higher with 6+ logged cycles.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        cycles = st.session_state.cycle_lengths
        predictor = st.session_state.predictor
        result = predictor.predict_next(cycles)
        next_date = st.session_state.last_period_date + timedelta(days=result["predicted_length"])
        days_until = (next_date - date.today()).days

        st.markdown("### Prediction Results")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{result['predicted_length']} days</div>
                <div class="label">Predicted Cycle Length</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{next_date.strftime('%d %b')}</div>
                <div class="label">Expected Next Period</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Cycle length chart
        if len(cycles) >= 2:
            fig = go.Figure()
            x = list(range(1, len(cycles) + 1))
            fig.add_trace(go.Scatter(
                x=x, y=cycles,
                mode="lines+markers",
                name="Actual",
                line=dict(color="#c2185b", width=2.5),
                marker=dict(size=8, color="#c2185b"),
            ))
            fig.add_hline(y=result["avg_cycle"], line_dash="dash", line_color="#7b1fa2",
                          annotation_text=f"Avg: {result['avg_cycle']}d")
            fig.add_trace(go.Scatter(
                x=[len(cycles) + 1], y=[result["predicted_length"]],
                mode="markers",
                name="Predicted",
                marker=dict(size=14, color="#ff6090", symbol="star"),
            ))
            fig.update_layout(
                title="Cycle Length History + Prediction",
                xaxis_title="Cycle #",
                yaxis_title="Length (days)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.5)",
                font=dict(family="DM Sans"),
                height=300,
                legend=dict(orientation="h", y=-0.3),
                yaxis=dict(range=[15, 50]),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Fertile window estimate
        ov_day = result["predicted_length"] - 14
        fertile_start = st.session_state.last_period_date + timedelta(days=ov_day - 3)
        fertile_end = st.session_state.last_period_date + timedelta(days=ov_day + 1)

        st.markdown('<div class="luna-card">', unsafe_allow_html=True)
        st.markdown(f"""
**📊 Cycle Stats**

| Metric | Value |
|---|---|
| Average cycle | {result['avg_cycle']} days |
| Std deviation | {result['std_dev']} days |
| Regularity | {'Regular ✅' if result['std_dev'] < 4 else 'Slightly Irregular ⚠️' if result['std_dev'] < 7 else 'Irregular ❗'} |
| Predicted ovulation | ~{next_date - timedelta(days=14) if days_until >= 0 else 'Past cycle'} |
| Fertile window (est.) | {fertile_start.strftime('%d %b')} – {fertile_end.strftime('%d %b')} |
| Model confidence | **{result['confidence'].capitalize()}** ({len(cycles)} cycles logged) |
        """)
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE 3: HEALTH RISK CHECK
# ═══════════════════════════════════════════════════════════
elif page == "🔬 Health Risk Check":
    st.markdown("# 🔬 PCOS Risk Assessment")
    st.markdown("*Logistic Regression model trained on clinical + lifestyle features*")
    st.info("⚠️ This is a screening tool for educational purposes only. Please consult a doctor for actual diagnosis.")
    st.markdown("---")

    col_inputs, col_output = st.columns([1, 1.2])

    with col_inputs:
        st.markdown("### Enter Your Details")
        st.markdown('<div class="luna-card">', unsafe_allow_html=True)

        st.markdown("**Basic Info**")
        age = st.number_input("Age", 16, 50, 24)
        bmi = st.number_input("BMI", 14.0, 50.0, 22.5, step=0.1)
        cycle_len = st.number_input("Average Cycle Length (days)", 15, 60, 28)

        st.markdown("**Symptoms** *(check all that apply)*")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            cycle_irreg = st.checkbox("Irregular cycles")
            weight_gain = st.checkbox("Unexplained weight gain")
            hair_growth = st.checkbox("Excess hair growth")
            hair_loss   = st.checkbox("Hair loss / thinning")
        with col_s2:
            skin_dark = st.checkbox("Skin darkening")
            pimples   = st.checkbox("Persistent acne")
            fast_food = st.checkbox("High fast food intake")

        st.markdown("**Lifestyle**")
        exercise = st.slider("Exercise days per week", 0, 7, 3)
        sleep = st.slider("Average sleep (hours/night)", 3, 12, 7)
        stress = st.select_slider("Stress level", options=[1, 2, 3, 4, 5],
                                   format_func=lambda x: ["Very Low","Low","Moderate","High","Very High"][x-1])

        analyze = st.button("🔬 Analyze Risk")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_output:
        if analyze:
            user_input = {
                "age": age, "bmi": bmi, "cycle_length": cycle_len,
                "cycle_irregular": cycle_irreg, "weight_gain": weight_gain,
                "hair_growth": hair_growth, "skin_darkening": skin_dark,
                "hair_loss": hair_loss, "pimples": pimples,
                "fast_food": fast_food, "exercise": exercise,
                "sleep_hours": sleep, "stress_level": stress,
            }
            result = st.session_state.risk_model.predict_risk(user_input)

            st.markdown("### Assessment Result")
            risk_class = result["risk_level"].lower()
            st.markdown(f'<div class="risk-{risk_class}">⬤ {result["risk_level"]} Risk &nbsp;|&nbsp; {result["probability"]}% probability</div>',
                        unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["probability"],
                number={"suffix": "%", "font": {"size": 40, "family": "DM Serif Display"}},
                gauge={
                    "axis": {"range": [0, 100], "tickfont": {"size": 12}},
                    "bar": {"color": result["color"]},
                    "steps": [
                        {"range": [0, 35], "color": "#e8f5e9"},
                        {"range": [35, 65], "color": "#fff8e1"},
                        {"range": [65, 100], "color": "#ffebee"},
                    ],
                    "threshold": {"line": {"color": "#3d1a47", "width": 3}, "value": result["probability"]},
                },
                title={"text": "PCOS Risk Score", "font": {"family": "DM Serif Display", "size": 18}},
            ))
            fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(family="DM Sans"))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="luna-card">', unsafe_allow_html=True)
            st.markdown(f"**💬 {result['advice']}**")
            if result["top_factors"]:
                st.markdown("**Key risk factors identified:**")
                for f in result["top_factors"]:
                    st.markdown(f"- {f}")
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown('<div class="luna-card">', unsafe_allow_html=True)
            st.markdown("""
### What this model does

This **Logistic Regression** model assesses PCOS risk based on:

- **Clinical markers**: BMI, cycle regularity, cycle length
- **Symptoms**: hair growth, acne, skin darkening, etc.
- **Lifestyle factors**: exercise, sleep, stress, diet

**Dataset**: Trained on synthetic data mirroring the Kaggle PCOS Dataset by Prasoon Kottarathil (41 features, 541 samples).

> In your final project, replace synthetic training with the real CSV using `risk_model.train_from_csv('pcos_data.csv')`.
            """)
            st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE 4: SYMPTOM PATTERNS
# ═══════════════════════════════════════════════════════════
elif page == "🌸 Symptom Patterns":
    st.markdown("# 🌸 Symptom Pattern Analysis")
    st.markdown("*K-Means Clustering to discover patterns in your symptom logs*")
    st.markdown("---")

    logs = st.session_state.symptom_logs

    if len(logs) < 3:
        st.markdown('<div class="luna-card">', unsafe_allow_html=True)
        st.markdown("### Not enough data yet")
        st.markdown(f"You have **{len(logs)} day(s)** logged. Log at least **3 days** of symptoms on the Home page to see patterns here.")
        st.markdown("#### What you'll see here:")
        st.markdown("""
- 🔵 **Scatter plot** of all your days clustered by symptom similarity
- 🏷️ **Cluster labels** (High Pain, Hormonal, Energetic, Bloating)
- 📊 **Breakdown** of which day-types occur most in your cycle
- 💡 **Personalized tips** based on your dominant symptom pattern
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        # Demo with synthetic data
        st.markdown("### Demo with sample data")
        demo_logs = [
            {"cramps": 1, "bloating": 0, "headache": 0, "fatigue": 1, "mood_swings": 0,
             "acne": 0, "back_pain": 1, "nausea": 0, "flow_intensity": 4, "pain_level": 4, "energy_level": 1, "mood_score": 2},
            {"cramps": 0, "bloating": 1, "headache": 0, "fatigue": 1, "mood_swings": 1,
             "acne": 1, "back_pain": 0, "nausea": 1, "flow_intensity": 1, "pain_level": 2, "energy_level": 2, "mood_score": 2},
            {"cramps": 0, "bloating": 0, "headache": 0, "fatigue": 0, "mood_swings": 0,
             "acne": 0, "back_pain": 0, "nausea": 0, "flow_intensity": 0, "pain_level": 0, "energy_level": 5, "mood_score": 5},
            {"cramps": 1, "bloating": 1, "headache": 1, "fatigue": 1, "mood_swings": 0,
             "acne": 0, "back_pain": 1, "nausea": 0, "flow_intensity": 3, "pain_level": 3, "energy_level": 2, "mood_score": 3},
            {"cramps": 0, "bloating": 1, "headache": 0, "fatigue": 0, "mood_swings": 1,
             "acne": 1, "back_pain": 0, "nausea": 1, "flow_intensity": 0, "pain_level": 1, "energy_level": 3, "mood_score": 2},
            {"cramps": 0, "bloating": 0, "headache": 0, "fatigue": 0, "mood_swings": 0,
             "acne": 0, "back_pain": 0, "nausea": 0, "flow_intensity": 0, "pain_level": 0, "energy_level": 4, "mood_score": 4},
        ]
        logs = demo_logs
        st.caption("*(Showing demo data — log your own symptoms on the Home page)*")

    clusterer = st.session_state.clusterer
    summary = clusterer.get_pattern_summary(logs)

    if summary:
        col_chart, col_stats = st.columns([1.4, 1])

        with col_chart:
            st.markdown("### Symptom Clusters (PCA Projection)")
            days_df = pd.DataFrame(summary["days"])

            fig = px.scatter(
                days_df, x="x", y="y",
                color="cluster_name",
                color_discrete_map={v["name"]: v["color"] for v in CLUSTER_PROFILES.values()},
                text="day",
                hover_data={"x": False, "y": False, "day": True, "cluster_name": True},
                labels={"x": "Component 1", "y": "Component 2", "cluster_name": "Day Type"},
            )
            fig.update_traces(marker=dict(size=14), textposition="top center",
                              textfont=dict(size=10))
            fig.update_layout(
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.4)",
                font=dict(family="DM Sans"),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_stats:
            st.markdown("### Cluster Breakdown")
            counts = summary["cluster_counts"]
            total = sum(counts.values())

            for name, count in counts.items():
                profile = next((p for p in CLUSTER_PROFILES.values() if p["name"] == name), None)
                if profile:
                    pct = round(count / total * 100)
                    st.markdown(f"""
<div style="margin: 0.5rem 0;">
  <span class="cluster-pill" style="background:{profile['color']}">{profile['emoji']} {name}</span>
  <span style="font-size:0.9rem; color:#555;"> {count} day(s) · {pct}%</span>
  <div style="background:#eee; border-radius:50px; height:6px; margin-top:4px;">
    <div style="width:{pct}%; background:{profile['color']}; height:6px; border-radius:50px;"></div>
  </div>
</div>
""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            dominant = summary["dominant_profile"]
            st.markdown('<div class="luna-card">', unsafe_allow_html=True)
            st.markdown(f"**Your dominant pattern: {dominant['emoji']} {dominant['name']}**")
            st.markdown(dominant["description"])
            st.markdown(f"💡 *{dominant['tip']}*")
            st.markdown('</div>', unsafe_allow_html=True)

        # Day-by-day timeline
        st.markdown("### Day-by-Day Timeline")
        st.markdown('<div class="luna-card">', unsafe_allow_html=True)
        cols = st.columns(min(len(summary["days"]), 7))
        for i, day_info in enumerate(summary["days"][:7]):
            cid = day_info["cluster_id"]
            profile = CLUSTER_PROFILES[cid]
            with cols[i % 7]:
                st.markdown(f"""
<div style="text-align:center; padding:0.5rem; background:{profile['color']}22; border-radius:12px; border: 2px solid {profile['color']};">
  <div style="font-size:1.5rem">{profile['emoji']}</div>
  <div style="font-size:0.75rem; font-weight:600; color:{profile['color']}">Day {day_info['day']}</div>
  <div style="font-size:0.65rem; color:#555">{profile['name'].split()[0]}</div>
</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Model info
        st.markdown("### About the Model")
        st.markdown('<div class="luna-card">', unsafe_allow_html=True)
        st.markdown("""
**K-Means Clustering** groups your daily symptom logs into 4 natural clusters:

| Cluster | Characteristics |
|---|---|
| 🔴 High Pain | Cramps, back pain, low energy, heavy flow |
| 🟡 Hormonal | Mood swings, acne, fatigue |
| 🟢 Energetic | Minimal symptoms, high energy & mood |
| 🟠 Bloating | Bloating, nausea, mild discomfort |

PCA (Principal Component Analysis) is used to project the high-dimensional symptom space into 2D for visualization.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
