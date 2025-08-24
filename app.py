import os
import re
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from rules_engine import compute_bmi, evaluate_risks, rules_risk_assessment
from data_store import init_db, upsert_entry, get_user_history, to_csv_bytes

# ----------------------
# App Config & Theming
# ----------------------
st.set_page_config(
    page_title="Symptom Tracker & Risk Dashboard",
    page_icon="🧑‍⚕️",
    layout="wide",
)

PRIMARY_CONDITIONS = [
    "Type 2 Diabetes Risk",
    "Hypertension Risk",
    "Depression/Mood Concern",
    "Migraine Risk",
    "Sleep Apnea Risk",
    "Anemia Risk",
]

# Map human label → DB column (underscored, preserve case to match data_store.py)
def to_db_key(label: str) -> str:
    # e.g. "Type 2 Diabetes Risk" -> "risk_Type_2_Diabetes_Risk"
    return "risk_" + re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_")


@st.cache_resource(show_spinner=False)
def get_hf_pipeline():
    try:
        return load_zero_shot_classifier()
    except Exception:
        return None

# ----------------------
# Sidebar – Navigation
# ----------------------
with st.sidebar:
    st.title("🧑‍⚕️Symptom Tracker")
    st.caption("Detect • Connect • Personalize")
    page = st.radio("Go to", ["Home", "Start", "Input", "Dashboard", "History"], index=0)
    st.divider()
    demo_mode = st.toggle("Use Demo Data", value=False, help="Prefill a demo input")
    st.divider()
    st.info(
        "Educational use only. Not a medical device. Consult a clinician for advice.",
        icon="⚠️",
    )

def explain_rules(rules_explain: dict) -> str:
    """
    Converts the explanation dict from rules_risk_assessment into
    a Streamlit-friendly markdown string with bold labels.
    """
    if not rules_explain:
        return "No rule-based explanation available."

    lines = []
    for key, value in rules_explain.items():
        lines.append(f"- **{key}**: {value}")
    return "\n".join(lines)


# ----------------------
# Global init
# ----------------------
init_db()

# --- Landing page CSS (clean, centered, Scholar_GPT-like) ---
st.markdown(
    """
<style>
:root{
  --bg: #f7fbff;
  --card: #ffffff;
  --muted: #6b7280;
  --accent: #3b82f6;
  --accent-dark: #2563eb;
  --glass: rgba(255,255,255,0.92);
  --shadow: 0 10px 30px rgba(12,20,60,0.06);
}
/* page background */
body { background: linear-gradient(180deg, #f7fbff, #ffffff); }
/* centered container */
.landing {
  display:flex;
  align-items:center;
  justify-content:center;
  padding:36px 16px;
}
.container {
  width:100%;
  max-width:1100px;
  display:flex;
  gap:28px;
  align-items:flex-start;
}
/* left hero */
.left {
  flex: 1.1;
  display:block;
}
.brand {
  display:flex;
  gap:14px;
  align-items:center;
  margin-bottom:8px;
}
.logo {
  width:56px;
  height:56px;
  border-radius:12px;
  background:linear-gradient(135deg,#eef6ff,#f6ecff);
  display:flex;
  align-items:center;
  justify-content:center;
  font-size:24px;
}
.title {
  font-size:32px;
  font-weight:800;
  color:#071033;
  margin:0;
}
.subtitle {
  font-size:15px;
  color:var(--muted);
  margin-top:6px;
  line-height:1.55;
}
/* CTA row */
.cta-row { margin-top:18px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
.cta-primary {
  background: linear-gradient(90deg,var(--accent),var(--accent-dark));
  color:white;
  padding:10px 16px;
  border-radius:10px;
  font-weight:700;
  cursor:default;
  box-shadow: 0 6px 18px rgba(37,99,235,0.12);
}
.cta-secondary {
  background:transparent;
  padding:10px 14px;
  border-radius:10px;
  border:1px solid rgba(15,23,42,0.06);
  color:var(--muted);
}
/* right card */
.right { flex:0.9; }
.card {
  background: var(--card);
  border-radius:12px;
  padding:18px;
  box-shadow: var(--shadow);
  border: 1px solid rgba(12,20,40,0.03);
}
.card .heading { font-weight:800; font-size:14px; color:#071033; margin-bottom:8px; }
.card .text { color:var(--muted); font-size:13px; line-height:1.55; }
/* feature chips inside card */
.features { display:flex; gap:10px; margin-top:12px; flex-wrap:wrap; }
.chip {
  background: linear-gradient(180deg,#ffffff,#fbfdff);
  padding:10px 12px;
  border-radius:10px;
  border: 1px solid rgba(10,10,10,0.02);
  min-width:140px;
}
.chip b { display:block; font-weight:800; color:#071033; font-size:14px; }
.chip p { margin:6px 0 0 0; font-size:13px; color:var(--muted); }
/* small note below */
.small-note { margin-top:12px; font-size:13px; color:var(--muted); }
/* responsive */
@media (max-width:980px){
  .container { flex-direction:column; }
  .right { order:2; }
  .left { order:1; }
  .chip { min-width:120px; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------
# Home / Landing Page
# ----------------------
if page == "Home":
    st.markdown("<div class='landing'>", unsafe_allow_html=True)
    st.markdown("<div class='container'>", unsafe_allow_html=True)

    # LEFT: hero content
    st.markdown("<div class='left'>", unsafe_allow_html=True)
    st.markdown("<div class='brand'><div class='logo'>🩺</div><div><h1 class='title'>Symptom Tracker</h1></div></div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Symptom Tracker & Risk Dashboard — capture vitals and symptoms, generate explainable risk flags, and view longitudinal trends for practical monitoring and research.</div>", unsafe_allow_html=True)

    st.markdown("<div class='cta-row'>", unsafe_allow_html=True)
    # Scholar_GPT shows a prominent button; we will show a "Get Started" CTA for clarity (does not change app flow)
    st.markdown("<div class='cta-primary'>Get Started — use the sidebar to navigate</div>", unsafe_allow_html=True)
    st.markdown("<div class='cta-secondary'>Tip: Toggle 'Use Demo Data' in the sidebar to prefill demo values</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='small-note'>Designed for demonstrations and prototyping. Not a clinical diagnostic tool. Always consult a clinician for medical decisions.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT: card with clean features
    st.markdown("<div class='right'>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='heading'>Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='text'>NeuraVia provides a straightforward, auditable pipeline: collect demographics and vitals, run explainable rule-based risk scoring (optionally augmented with AI), persist entries locally, visualize trends, and export CSV for clinical review or research.</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='heading'>Key features</div>", unsafe_allow_html=True)
    st.markdown("<div class='features'>", unsafe_allow_html=True)
    st.markdown("<div class='chip'><b>Explainable Risk</b><p>Human-readable rules that show why an assessment was made.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='chip'><b>Trend Visuals</b><p>Time-series charts for vitals and risk metrics to follow changes.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='chip'><b>Exportable Data</b><p>Download CSV to share with clinicians or for analysis.</p></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close card
    st.markdown("</div>", unsafe_allow_html=True)  # close right
    st.markdown("</div>", unsafe_allow_html=True)  # close container
    st.markdown("</div>", unsafe_allow_html=True)  # close landing

    st.info("Use the sidebar to open **Input** and start a demo. Toggle **Use Demo Data** for prefilled example values.", icon="💡")

# ----------------------
# Start Page
# ----------------------
if page == "Start":
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("### What this does")
        st.write(
            "- Collect demographics, vitals, and symptoms.\n"
            "- Score conditions via explainable rules + optional HF zero-shot.\n"
            "- Store locally in SQLite and visualize trends.\n"
            "- Export CSV for sharing."
        )
        st.markdown("### How to demo")
        st.write(
            "1) Open **Input** and submit (or toggle Demo).\n"
            "2) Open **Dashboard** for cards & charts.\n"
            "3) Use **History** to review and export CSV."
        )
    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Beginner-friendly • Interpretable • Real-world relevance**")
        st.caption("Detect · Connect · Personalize")
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# Input Page
# ----------------------
if page == "Input":
    st.markdown("<div class='section-title'>Patient Data Input</div>", unsafe_allow_html=True)

    if demo_mode:
        default = dict(
            user_id="demo_user",
            age=42,
            sex="Male",
            height_cm=175,
            weight_kg=92,
            sbp=142,
            dbp=92,
            hr=88,
            spo2=96,
            glucose=118,
            symptoms=["Frequent urination", "Headache", "Daytime sleepiness", "Loud snoring"],
            free_text="Often thirsty and tired; loud snoring; morning headaches.",
        )
    else:
        default = dict(
            user_id="",
            age=30,
            sex="Female",
            height_cm=165,
            weight_kg=65,
            sbp=120,
            dbp=80,
            hr=72,
            spo2=98,
            glucose=95,
            symptoms=[],
            free_text="",
        )

    with st.form("input_form"):
        st.markdown("**Profile**")
        c1, c2, c3 = st.columns(3)
        user_id = c1.text_input("User ID (unique)", value=default["user_id"], help="Used to group your timeline")
        age = c2.number_input("Age", 1, 120, int(default["age"]))
        sex = c3.selectbox(
            "Sex",
            ["Female", "Male", "Other", "Prefer not to say"],
            index=1 if default["sex"] == "Male" else 0,
        )

        st.markdown("**Vitals**")
        c1, c2, c3, c4 = st.columns(4)
        height_cm = c1.number_input("Height (cm)", 80, 230, int(default["height_cm"]))
        weight_kg = c2.number_input("Weight (kg)", 20, 250, int(default["weight_kg"]))
        sbp = c3.number_input("Systolic BP (SBP)", 70, 240, int(default["sbp"]))
        dbp = c4.number_input("Diastolic BP (DBP)", 40, 140, int(default["dbp"]))

        c1, c2, c3 = st.columns(3)
        hr = c1.number_input("Heart Rate (bpm)", 30, 200, int(default["hr"]))
        spo2 = c2.number_input("SpO₂ (%)", 70, 100, int(default["spo2"]))
        glucose = c3.number_input("Glucose (mg/dL)", 50, 400, int(default["glucose"]))

        st.markdown("**Symptoms**")
        SYMPTOMS_LIST = [
            "Frequent urination", "Excessive thirst", "Unintended weight loss", "Blurred vision",
            "Headache", "Nausea", "Sensitivity to light", "Sleep problems", "Daytime sleepiness",
            "Loud snoring", "Pauses in breathing during sleep", "Shortness of breath",
            "Fatigue", "Dizziness", "Pale skin", "Cold hands/feet",
            "Low mood", "Loss of interest", "Anxiety", "Irritability",
        ]
        symptoms = st.multiselect("Select common symptoms (optional)", SYMPTOMS_LIST, default=default["symptoms"])
        free_text = st.text_area("Describe your symptoms (optional)", value=default["free_text"], height=110)

        submitted = st.form_submit_button("Compute Risk", use_container_width=True)

    if submitted:
        if not user_id:
            st.error("Please enter a User ID.")
            st.stop()

        bmi = compute_bmi(height_cm, weight_kg)
        rules_scores, rules_explain = rules_risk_assessment(
            age=age,
            sex=sex,
            sbp=sbp,
            dbp=dbp,
            hr=hr,
            spo2=spo2,
            glucose=glucose,
            bmi=bmi,
            symptoms=symptoms,
        )

        # Optional HF zero-shot scoring
        hf_scores = {}
        pipe = get_hf_pipeline()
        if pipe is not None:
            text = "; ".join(symptoms) + ("; " + free_text if free_text else "")
            hf_scores = zero_shot_score(pipe, text=text, labels=PRIMARY_CONDITIONS)

        # Combine scores (prefer explainable rules slightly)
        combined = {}
        for cond in PRIMARY_CONDITIONS:
            r = float(rules_scores.get(cond, 0.0))
            h = hf_scores.get(cond, np.nan)
            combined[cond] = r if (isinstance(h, float) and np.isnan(h)) else (0.65 * r + 0.35 * float(h))

        # Persist entry (note: DB uses underscored columns; upsert_entry expects dict keys with spaces)
        entry = {
            "user_id": user_id,
            "ts": datetime.utcnow().isoformat(),
            "age": age,
            "sex": sex,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "bmi": round(bmi, 2),
            "sbp": sbp,
            "dbp": dbp,
            "hr": hr,
            "spo2": spo2,
            "glucose": glucose,
            "symptoms": ", ".join(symptoms),
            "free_text": free_text,
            **{ to_db_key(cond): round(score, 3) for cond, score in combined.items() },
        }

        upsert_entry(entry)

        st.success("Risk computed and saved. Open the **Dashboard** tab.")
        with st.expander("See rule explanations"):
            st.markdown(explain_rules(rules_explain))


# ----------------------
# Dashboard Page
# ----------------------
if page == "Dashboard":
    st.markdown("<div class='section-title'>Risk Dashboard</div>", unsafe_allow_html=True)
    uid = st.text_input("Enter User ID to view", value="demo_user" if demo_mode else "")
    if not uid:
        st.info("Enter your User ID to see assessments.")
        st.stop()

    df = get_user_history(uid)
    if df.empty:
        st.warning("No records found. Submit an entry in **Input**.")
        st.stop()

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.sort_values("ts")
    latest = df.iloc[-1]

    # Top risk cards (use DB column names)
    st.markdown("#### Latest Risk Flags")
    cols = st.columns(3)

    def badge(v: float) -> str:
        if v >= 0.75:
            return "🔴 High"
        if v >= 0.5:
            return "🟠 Moderate"
        if v >= 0.25:
            return "🟡 Mild"
        return "🟢 Low"

    metrics = []
    for label in PRIMARY_CONDITIONS:
        colname = to_db_key(label)
        if colname in df.columns:
            try:
                val = float(latest[colname])
            except Exception:
                val = 0.0
            pretty = label.replace(" Risk", "")
            metrics.append((pretty, val))

    for i, (name, val) in enumerate(metrics):
        with cols[i % 3]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"**{name}**")
            st.markdown(f"<div class='metric'>{val:.2f}</div>", unsafe_allow_html=True)
            st.caption(badge(val))
            st.markdown("</div>", unsafe_allow_html=True)

    # Trend charts
    st.markdown("#### Trends Over Time")
    vitals = ["bmi", "sbp", "dbp", "hr", "spo2", "glucose"]
    display_to_col = {**{v: v for v in vitals}, **{f"risk: {lbl}": to_db_key(lbl) for lbl in PRIMARY_CONDITIONS}}

    pick = st.multiselect(
        "Choose metrics to plot",
        options=vitals + [f"risk: {lbl}" for lbl in PRIMARY_CONDITIONS],
        default=["bmi", "sbp", "dbp"],
    )

    for disp in pick:
        colname = display_to_col.get(disp, disp)
        if colname not in df.columns:
            continue
        chart = (
            alt.Chart(df).mark_line(point=True).encode(
                x=alt.X("ts:T", title="Time"),
                y=alt.Y(f"{colname}:Q", title=disp),
                tooltip=["ts:T", colname],
            ).properties(height=220)
        )
        st.altair_chart(chart, use_container_width=True)

# ----------------------
# History Page
# ----------------------
if page == "History":
    st.markdown("<div class='section-title'>History & Export</div>", unsafe_allow_html=True)
    uid = st.text_input("User ID", value="demo_user" if demo_mode else "")
    if not uid:
        st.info("Enter a User ID to load history.")
        st.stop()

    df = get_user_history(uid)
    if df.empty:
        st.warning("No records yet. Submit data in **Input**.")
    else:
        st.dataframe(df, use_container_width=True)
        csv_bytes = to_csv_bytes(df)
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name=f"{uid}_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.markdown("---")
st.caption("NeuraViaHacks submission by Qaisar A • Built with Streamlit • Not medical advice.")
