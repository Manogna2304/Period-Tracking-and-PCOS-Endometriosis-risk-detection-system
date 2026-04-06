import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from cycle_predictor import CyclePredictor
from risk_model import HealthRiskModel
from symptom_cluster import SymptomClusterer, CLUSTER_PROFILES

# ── Timezone Helper ───────────────────────────────────────────────────────────
def get_local_date():
    """Returns current date based on the system's local timezone."""
    return datetime.now().astimezone().date()

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
    color: #3d1a47;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #fdf6f0 0%, #fce4ec 50%, #f3e5f5 100%);
}

/* Sidebar - Lightened to support dark text */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fdf6f0 0%, #f3e5f5 100%);
    border-right: 1px solid rgba(180,100,200,0.2);
}
[data-testid="stSidebar"] * { color: #3d1a47 !important; }
[data-testid="stSidebar"] .stRadio label { color: #3d1a47 !important; font-size: 1rem; }

/* Headers */
h1 { font-family: 'DM Serif Display', serif !important; color: #3d1a47 !important; font-size: 2.6rem !important; margin-top: -20px; }
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
.luna-card p, .luna-card span, .luna-card div,
.luna-card td, .luna-card th, .luna-card li {
    color: #3d1a47 !important;
}

/* Metric cards - Lightened backgrounds with dark text */
.metric-card {
    background: rgba(255,255,255,0.6);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    color: #3d1a47;
    text-align: center;
    border: 1px solid rgba(180,100,200,0.3);
    box-shadow: 0 4px 15px rgba(109,45,107,0.05);
}
.metric-card .value { font-size: 2.4rem; font-weight: 700; font-family: 'DM Serif Display', serif; color: #3d1a47 !important; }
.metric-card .label { font-size: 0.85rem; opacity: 0.85; letter-spacing: 0.05em; text-transform: uppercase; color: #6b2d6b !important; }

/* Risk badge */
.risk-low    { background: #e8f5e9; color: #2e7d32 !important; border: 2px solid #66bb6a; border-radius: 12px; padding: 0.5rem 1.2rem; font-weight: 600; display: inline-block; margin-bottom: 10px; }
.risk-moderate { background: #fff8e1; color: #e65100 !important; border: 2px solid #ffa726; border-radius: 12px; padding: 0.5rem 1.2rem; font-weight: 600; display: inline-block; margin-bottom: 10px; }
.risk-high   { background: #ffebee; color: #b71c1c !important; border: 2px solid #ef5350; border-radius: 12px; padding: 0.5rem 1.2rem; font-weight: 600; display: inline-block; margin-bottom: 10px; }

/* Cluster pill - Dark text */
.cluster-pill { display: inline-block; border-radius: 50px; padding: 0.35rem 1rem; font-size: 0.9rem; font-weight: 600; color: #3d1a47 !important; margin: 0.2rem; border: 1px solid rgba(0,0,0,0.1); }

/* Buttons - Lightened background to support dark text */
.stButton > button {
    background: rgba(255,255,255,0.8) !important;
    color: #3d1a47 !important; 
    border: 1px solid #7b1fa2 !important; 
    border-radius: 12px !important; 
    font-weight: 600 !important;
    padding: 0.6rem 2rem !important; 
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: rgba(123,31,162,0.1) !important;
    border-color: #c2185b !important;
}

/* Hide streamlit branding and white header bars */
#MainMenu, footer { visibility: hidden; }
header { visibility: hidden !important; display: none !important; }
.stApp > header { background-color: transparent; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
if "period_dates" not in st.session_state:
    # Generate default past period dates based on today's local date
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
if "predictor" not in st.session_state:
    st.session_state.predictor = CyclePredictor()
if "risk_model" not in st.session_state:
    st.session_state.risk_model = HealthRiskModel()
if "clusterer" not in st.session_state:
    c = SymptomClusterer()
    c.fit()
    st.session_state.clusterer = c

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌙 Luna")
    st.markdown("*Your intelligent cycle companion*")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📅 Cycle Prediction", "🔬 Health Risk Check", "🌸 Symptom Patterns"],
        label_visibility="collapsed",
    )
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("**ML Models Used**")
    st.markdown("📈 Linear Regression")
    st.markdown("🧬 Logistic Regression")
    st.markdown("🔵 K-Means Clustering")
    st.caption("⚠️ This app is for educational purposes only. Not a medical device.")


# ═══════════════════════════════════════════════════════════
# PAGE 1: HOME
# ═══════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("# 🌙 Luna · Period Tracker")
    st.markdown("*Track your cycle, understand your body, powered by ML*")

    # Quick stats
    predictor = st.session_state.predictor
    dates = st.session_state.period_dates
    result = predictor.predict_next(dates)
    next_period = result.get("next_predicted_date", get_local_date())
    days_until = (next_period - get_local_date()).days

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{result.get('avg_cycle', 28)}</div>
            <div class="label">Avg Cycle (days)</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{result.get('predicted_length', 28)}</div>
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
            <div class="value">{max(0, len(dates) - 1)}</div>
            <div class="label">Cycles Logged</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two columns: log + calendar
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.markdown("### 📝 Log Today's Symptoms")
        with st.container():
            st.markdown('<div class="luna-card">', unsafe_allow_html=True)
            log_date = get_local_date()
            st.markdown(f"**Date:** {log_date.strftime('%B %d, %Y')} *(Timezone Auto-detected)*")
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
                cluster_result = st.session_state.clusterer.predict_day(log)
                st.success(f"Logged! Today looks like a **{cluster_result['name']}** day {cluster_result['emoji']}")
                st.caption(cluster_result["tip"])
            st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown("### 📆 Cycle Calendar")
        with st.container():
            st.markdown('<div class="luna-card">', unsafe_allow_html=True)
            last_date = max(dates) if dates else get_local_date()
            avg_len = int(result.get("avg_cycle", 28))
            
            periods = []
            current = last_date
            for _ in range(4):
                periods.append((current, current + timedelta(days=5)))
                current = current + timedelta(days=avg_len)

            fig = go.Figure()
            for i, (start, end) in enumerate(periods):
                color = "#c2185b" if i == 0 else "#e91e8c" if i == 1 else "#f48fb1"
                label = "Last Period" if i == 0 else f"Predicted #{i}"
                fig.add_trace(go.Bar(
                    x=[5], y=[start.strftime("%b %d")],
                    orientation="h", marker_color=color, name=label,
                    hovertemplate=f"{label}: {start.strftime('%b %d')} – {end.strftime('%b %d')}<extra></extra>",
                ))

            fig.add_vline(x=0, line_color="#7b1fa2", line_dash="dot", annotation_text="Today", annotation_font_color="#3d1a47")
            fig.update_layout(
                height=260, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=True, legend=dict(orientation="h", y=-0.2, font=dict(color="#3d1a47", family="DM Sans")),
                margin=dict(l=10, r=10, t=10, b=10), barmode="overlay",
                xaxis=dict(visible=False), yaxis=dict(tickfont=dict(color="#3d1a47", family="DM Sans"), gridcolor="rgba(180,100,200,0.1)"),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
<table style="width:100%; color:#3d1a47; font-size:0.95rem; border-collapse:collapse;">
  <tr style="border-bottom:1px solid rgba(180,100,200,0.2);">
    <td style="padding:0.4rem 0;">🔴 <b>Last period started</b></td>
    <td style="color:#6b2d6b; font-weight:600;">{last_date.strftime('%d %b %Y')}</td>
  </tr>
  <tr style="border-bottom:1px solid rgba(180,100,200,0.2);">
    <td style="padding:0.4rem 0;">🔮 <b>Next period predicted</b></td>
    <td style="color:#6b2d6b; font-weight:600;">{next_period.strftime('%d %b %Y')}</td>
  </tr>
</table>
""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE 2: CYCLE PREDICTION
# ═══════════════════════════════════════════════════════════
elif page == "📅 Cycle Prediction":
    st.markdown("# 📅 Cycle Prediction")
    st.markdown("*Linear Regression trained on your logged period dates*")

    col_form, col_result = st.columns([1, 1.4])

    with col_form:
        st.markdown("### Your Period History")
        st.markdown('<div class="luna-card">', unsafe_allow_html=True)
        st.caption("Add or modify your previous period start dates.")
        
        # DataFrame editor for dates
        df_dates = pd.DataFrame({"Period Start Date": st.session_state.period_dates})
        df_dates["Period Start Date"] = pd.to_datetime(df_dates["Period Start Date"]).dt.date
        
        edited_df = st.data_editor(df_dates, num_rows="dynamic", use_container_width=True)
        
        if st.button("🔮 Predict / Update Model"):
            # Update session state with valid dates, sorted
            new_dates = pd.to_datetime(edited_df["Period Start Date"]).dt.date.dropna().tolist()
            st.session_state.period_dates = sorted(new_dates)
            st.session_state.predictor = CyclePredictor()
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        dates = st.session_state.period_dates
        predictor = st.session_state.predictor
        result = predictor.predict_next(dates)
        
        if len(dates) >= 2:
            next_date = result["next_predicted_date"]
            cycles = predictor._get_cycle_lengths(dates)
            
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

            # Fertile window estimate
            ov_day = result["predicted_length"] - 14
            last_p = max(dates)
            fertile_start = last_p + timedelta(days=ov_day - 3)
            fertile_end = last_p + timedelta(days=ov_day + 1)

            st.markdown('<div class="luna-card">', unsafe_allow_html=True)
            st.markdown(f"""
<table style="width:100%; color:#3d1a47; font-size:0.95rem; border-collapse:collapse;">
  <tr style="border-bottom:1px solid rgba(180,100,200,0.15);">
    <td style="padding:0.35rem 0;">Average cycle</td>
    <td style="color:#6b2d6b; font-weight:600;">{result.get('avg_cycle')} days</td>
  </tr>
  <tr style="border-bottom:1px solid rgba(180,100,200,0.15);">
    <td style="padding:0.35rem 0;">Regularity</td>
    <td style="color:#6b2d6b; font-weight:600;">{'Regular ✅' if result.get('std_dev', 0) < 4 else 'Irregular ❗'}</td>
  </tr>
  <tr style="border-bottom:1px solid rgba(180,100,200,0.15);">
    <td style="padding:0.35rem 0;">Predicted ovulation</td>
    <td style="color:#6b2d6b; font-weight:600;">~{(next_date - timedelta(days=14)).strftime('%d %b')}</td>
  </tr>
  <tr style="border-bottom:1px solid rgba(180,100,200,0.15);">
    <td style="padding:0.35rem 0;">Fertile window (est.)</td>
    <td style="color:#6b2d6b; font-weight:600;">{fertile_start.strftime('%d %b')} – {fertile_end.strftime('%d %b')}</td>
  </tr>
</table>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Log at least two period start dates to generate predictions.")


# ═══════════════════════════════════════════════════════════
# PAGE 3: HEALTH RISK CHECK
# ═══════════════════════════════════════════════════════════
elif page == "🔬 Health Risk Check":
    st.markdown("# 🔬 Dual Health Risk Assessment")
    st.markdown("*Logistic Regression models evaluating PCOS and Endometriosis profiles using Kaggle Data*")
    st.info("⚠️ This is a screening tool for educational purposes only. Please consult a doctor for actual diagnosis.")

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
            pimples     = st.checkbox("Persistent acne")
            pelvic_pain = st.checkbox("Chronic pelvic pain")
        with col_s2:
            heavy_bleed = st.checkbox("Heavy menstrual bleeding")
            pain_intercourse = st.checkbox("Pain during intercourse")
            skin_dark = st.checkbox("Skin darkening")
            hair_loss   = st.checkbox("Hair loss / thinning")
            fast_food = st.checkbox("High fast food intake")

        st.markdown("**Lifestyle & History**")
        family_history = st.checkbox("Family history of Endometriosis/PCOS")
        exercise = st.slider("Exercise days per week", 0, 7, 3)
        sleep = st.slider("Average sleep (hours/night)", 3, 12, 7)

        analyze = st.button("🔬 Analyze Risk")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_output:
        if analyze:
            user_input = {
                "age": age, "bmi": bmi, "cycle_length": cycle_len,
                "cycle_irregular": cycle_irreg, "weight_gain": weight_gain,
                "hair_growth": hair_growth, "skin_darkening": skin_dark,
                "hair_loss": hair_loss, "pimples": pimples, "fast_food": fast_food,
                "pelvic_pain": pelvic_pain, "heavy_bleeding": heavy_bleed, 
                "pain_intercourse": pain_intercourse, "family_history": family_history,
                "exercise": exercise, "sleep_hours": sleep
            }
            results = st.session_state.risk_model.predict_risk(user_input)

            st.markdown("### Assessment Result")
            
            # Display PCOS
            pcos = results["PCOS"]
            st.markdown(f'<div class="risk-{pcos["risk_level"].lower()}">⬤ PCOS: {pcos["risk_level"]} Risk ({pcos["probability"]}%)</div>', unsafe_allow_html=True)
            
            # Display Endo
            endo = results["Endometriosis"]
            st.markdown(f'<div class="risk-{endo["risk_level"].lower()}">⬤ Endometriosis: {endo["risk_level"]} Risk ({endo["probability"]}%)</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="luna-card">', unsafe_allow_html=True)
            st.markdown("#### Recommendations")
            st.markdown(f"**PCOS:** {pcos['advice']}")
            st.markdown(f"**Endometriosis:** {endo['advice']}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="luna-card">', unsafe_allow_html=True)
            st.markdown("""
### What this model does

It runs two parallel **Logistic Regression** models connected to Kaggle data structures to assess:
1. **PCOS Risk:** Based on clinical markers (BMI, cycle length) and symptoms (hair growth, acne, skin darkening).
2. **Endometriosis Risk:** Based on specific pain profiles (pelvic pain, dysmenorrhea), heavy bleeding, and family history.

*The models dynamically load `pcos_dataset.csv` and `endo_dataset.csv` if placed in the directory, falling back seamlessly if unavailable.*
            """)
            st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE 4: SYMPTOM PATTERNS
# ═══════════════════════════════════════════════════════════
elif page == "🌸 Symptom Patterns":
    st.markdown("# 🌸 Symptom Pattern Analysis")
    st.markdown("*K-Means Clustering to discover patterns in your symptom logs*")

    logs = st.session_state.symptom_logs

    if len(logs) < 3:
        st.markdown('<div class="luna-card">', unsafe_allow_html=True)
        st.markdown("### Not enough data yet")
        st.markdown(f"You have **{len(logs)} day(s)** logged. Log at least **3 days** of symptoms on the Home page to see patterns here.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Load demo logs if < 3
        logs = [
            {"cramps": 1, "bloating": 0, "headache": 0, "fatigue": 1, "mood_swings": 0, "acne": 0, "back_pain": 1, "nausea": 0, "flow_intensity": 4, "pain_level": 4, "energy_level": 1, "mood_score": 2},
            {"cramps": 0, "bloating": 1, "headache": 0, "fatigue": 1, "mood_swings": 1, "acne": 1, "back_pain": 0, "nausea": 1, "flow_intensity": 1, "pain_level": 2, "energy_level": 2, "mood_score": 2},
            {"cramps": 0, "bloating": 0, "headache": 0, "fatigue": 0, "mood_swings": 0, "acne": 0, "back_pain": 0, "nausea": 0, "flow_intensity": 0, "pain_level": 0, "energy_level": 5, "mood_score": 5},
            {"cramps": 1, "bloating": 1, "headache": 1, "fatigue": 1, "mood_swings": 0, "acne": 0, "back_pain": 1, "nausea": 0, "flow_intensity": 3, "pain_level": 3, "energy_level": 2, "mood_score": 3},
        ]
        st.caption("*(Showing demo data — log your own symptoms on the Home page)*")

    clusterer = st.session_state.clusterer
    summary = clusterer.get_pattern_summary(logs)

    if summary:
        col_chart, col_stats = st.columns([1.4, 1])

        with col_chart:
            st.markdown("### Symptom Clusters (PCA Projection)")
            days_df = pd.DataFrame(summary["days"])
            fig = px.scatter(
                days_df, x="x", y="y", color="cluster_name",
                color_discrete_map={v["name"]: v["color"] for v in CLUSTER_PROFILES.values()},
                text="day", hover_data={"x": False, "y": False, "day": True, "cluster_name": True},
            )
            fig.update_traces(marker=dict(size=14), textposition="top center", textfont=dict(size=10, color="#3d1a47"))
            fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.4)")
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
  <span class="cluster-pill" style="background:{profile['color']}40">{profile['emoji']} {name}</span>
  <span style="font-size:0.9rem; color:#3d1a47;"> {count} day(s) · {pct}%</span>
</div>
""", unsafe_allow_html=True)
            
            dominant = summary["dominant_profile"]
            st.markdown('<div class="luna-card">', unsafe_allow_html=True)
            st.markdown(f"**Dominant pattern: {dominant['emoji']} {dominant['name']}**")
            st.markdown(f"💡 *{dominant['tip']}*")
            st.markdown('</div>', unsafe_allow_html=True)
