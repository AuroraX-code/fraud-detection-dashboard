"""
AI-Based Credit Card Fraud Detection Dashboard
Machine Learning Model: XGBoost Classifier

Features:
- Real-time single transaction prediction with a needle-based risk gauge
- Bulk CSV batch processing for many transactions at once
- Explainable AI: shows which features pushed a transaction toward fraud
- Hybrid system: machine learning combined with business rule overrides
- Prediction history and downloadable reports
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Business rule threshold: transactions above this amount always need review
HIGH_VALUE_THRESHOLD_USD = 10000.0
USD_TO_INR = 83.0

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%) !important; }
    [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%) !important;
    }
    [data-testid="stHeader"] { background: rgba(0,0,0,0) !important; }
    .main .block-container { padding-top: 1.5rem !important; }

    section[data-testid="stSidebar"] { background-color: #161b27 !important; }
    section[data-testid="stSidebar"] * { color: #e8ecf3 !important; }
    section[data-testid="stSidebar"] input { color: #fff !important; background-color: #0f1419 !important; }
    section[data-testid="stSidebar"] button {
        background-color: #21618c !important; color: #fff !important;
        border: 1px solid #3498db !important; border-radius: 8px !important; font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] button:hover {
        background-color: #2e86c1 !important; box-shadow: 0 0 12px #3498db !important;
    }

    .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label { color: #e8ecf3 !important; }

    [data-testid="stExpander"] { background-color: #161b27 !important; border: 1px solid #2a3142 !important; border-radius: 8px !important; }
    [data-testid="stExpander"] * { color: #e8ecf3 !important; }

    [data-testid="stMain"] button {
        background-color: #21618c !important; color: #fff !important;
        border: 1px solid #3498db !important; border-radius: 8px !important; font-weight: 600 !important;
    }
    [data-testid="stDownloadButton"] button { background-color: #21618c !important; color: #fff !important; border: 1px solid #3498db !important; }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #16222a 0%, #1e3c5a 50%, #21618c 100%);
        border-radius: 16px; padding: 20px 24px; text-align: center;
        margin-bottom: 22px; border: 1px solid #2a4a6a;
        box-shadow: 0 6px 24px rgba(33, 97, 140, 0.3);
    }
    .hero-title { font-size: 40px; font-weight: 900; color: #fff; margin: 0; letter-spacing: -1px; line-height: 1.1; }
    .hero-subtitle { font-size: 15px; color: #c5d0de; margin-top: 8px; font-weight: 500; }
    .hero-author { font-size: 13px; color: #9aa7b8; margin-top: 4px; }
    .hero-badge {
        display: inline-block; background: rgba(52,152,219,0.18); border: 1px solid #3498db;
        color: #5dade2; padding: 4px 12px; border-radius: 18px; font-size: 12px; font-weight: 600; margin: 6px 4px 0 4px;
    }

    .result-fraud {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
        padding: 26px; border-radius: 14px; text-align: center; color: #fff;
        font-size: 24px; font-weight: 700; box-shadow: 0 8px 24px rgba(231,76,60,0.4);
        animation: fadeIn 0.5s ease-out;
    }
    .result-safe {
        background: linear-gradient(135deg, #1e8449 0%, #27ae60 100%);
        padding: 26px; border-radius: 14px; text-align: center; color: #fff;
        font-size: 24px; font-weight: 700; box-shadow: 0 8px 24px rgba(39,174,96,0.4);
        animation: fadeIn 0.5s ease-out;
    }
    .result-review {
        background: linear-gradient(135deg, #b9770e 0%, #f39c12 100%);
        padding: 26px; border-radius: 14px; text-align: center; color: #fff;
        font-size: 24px; font-weight: 700; box-shadow: 0 8px 24px rgba(243,156,18,0.4);
        animation: fadeIn 0.5s ease-out;
    }
    @keyframes fadeIn { 0% { opacity: 0; transform: scale(0.95); } 100% { opacity: 1; transform: scale(1); } }

    .result-fraud:hover, .result-safe:hover, .result-review:hover {
        transform: translateY(-4px); transition: transform 0.3s ease;
    }

    .footer-card {
        background: linear-gradient(135deg, #16222a 0%, #1e3c5a 100%);
        border: 1px solid #2a4a6a; border-radius: 14px;
        padding: 18px 24px; text-align: center; margin-top: 10px;
    }
    .footer-card h4 { color: #fff; margin: 0 0 4px 0; font-size: 18px; }
    .footer-card p { color: #9aa7b8; margin: 0; font-size: 13px; }

    .info-box { background: #1a1f2e; padding: 18px; border-radius: 12px; border-left: 4px solid #3498db; color: #cdd5e0; }
    div[data-testid="stMetric"] { background: #1a1f2e; border: 1px solid #2a3142; border-radius: 12px; padding: 14px; }
    div[data-testid="stMetric"] label { color: #8b95a7 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load Model and Scalers
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("fraud_detection_model.pkl")
    amount_scaler = joblib.load("amount_scaler.pkl")
    time_scaler = joblib.load("time_scaler.pkl")
    return model, amount_scaler, time_scaler

try:
    model, amount_scaler, time_scaler = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">🛡️ Credit Card Fraud Detection System</h1>
    <p class="hero-subtitle">Real-Time Transaction Analysis Powered by XGBoost Machine Learning</p>
    <p class="hero-author">A BCA Project by Vikash Kumar &nbsp;|&nbsp; Amity University Online</p>
    <div>
        <span class="hero-badge">99.95% Accuracy</span>
        <span class="hero-badge">Real-Time Detection</span>
        <span class="hero-badge">Explainable AI</span>
        <span class="hero-badge">Hybrid Rule Engine</span>
    </div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("Model files could not be loaded. Please ensure 'fraud_detection_model.pkl', 'amount_scaler.pkl', and 'time_scaler.pkl' are in the same folder as this application.")
    st.code(load_error)
    st.stop()

# ---------------------------------------------------------------------------
# Sample transactions
# ---------------------------------------------------------------------------
sample_normal = {
    'Time': 40000, 'Amount': 88.50,
    'V1': -0.5, 'V2': 0.3, 'V3': 1.2, 'V4': 0.4, 'V5': -0.2, 'V6': 0.1,
    'V7': 0.05, 'V8': 0.1, 'V9': 0.3, 'V10': 0.1, 'V11': -0.4, 'V12': 0.2,
    'V13': -0.1, 'V14': 0.3, 'V15': 0.5, 'V16': -0.2, 'V17': 0.1, 'V18': 0.0,
    'V19': 0.1, 'V20': -0.1, 'V21': -0.02, 'V22': 0.28, 'V23': -0.11, 'V24': 0.07,
    'V25': 0.13, 'V26': -0.19, 'V27': 0.13, 'V28': -0.02
}
sample_fraud = {
    'Time': 47000, 'Amount': 0.0,
    'V1': -3.0, 'V2': 3.5, 'V3': -5.5, 'V4': 4.5, 'V5': -3.2, 'V6': -1.5,
    'V7': -5.0, 'V8': 1.2, 'V9': -2.5, 'V10': -5.5, 'V11': 4.0, 'V12': -6.0,
    'V13': -0.5, 'V14': -7.5, 'V15': 0.2, 'V16': -4.5, 'V17': -8.0, 'V18': -2.5,
    'V19': 1.0, 'V20': 0.5, 'V21': 0.8, 'V22': 0.1, 'V23': -0.3, 'V24': 0.1,
    'V25': 0.2, 'V26': 0.1, 'V27': 0.5, 'V28': 0.2
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def scale_row(time_value, amount_value, v_dict):
    """Build a single scaled feature row in the correct order."""
    scaled_time = time_scaler.transform([[time_value]])[0][0]
    scaled_amount = amount_scaler.transform([[amount_value]])[0][0]
    row = [scaled_time] + [v_dict[f"V{i}"] for i in range(1, 29)] + [scaled_amount]
    return np.array(row).reshape(1, -1)

def make_needle_gauge(value):
    """Speedometer gauge with a real needle drawn as a line."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': "%", 'font': {'size': 44, 'color': '#ffffff'}},
        title={'text': "Fraud Risk Level", 'font': {'size': 20, 'color': '#ffffff'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#fff', 'tickfont': {'color': '#fff', 'size': 13}},
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': "#27ae60"},
                {'range': [30, 50], 'color': "#f1c40f"},
                {'range': [50, 75], 'color': "#e67e22"},
                {'range': [75, 100], 'color': "#e74c3c"},
            ],
        }
    ))
    angle = math.pi * (1 - value / 100.0)
    r = 0.40
    cx, cy = 0.5, 0.0
    fig.add_shape(type="line", xref="paper", yref="paper",
                  x0=cx, y0=cy, x1=cx + r * math.cos(angle), y1=cy + r * math.sin(angle),
                  line=dict(color="#ffffff", width=5))
    fig.add_shape(type="circle", xref="paper", yref="paper",
                  x0=cx - 0.025, y0=cy - 0.025, x1=cx + 0.025, y1=cy + 0.025,
                  fillcolor="#ffffff", line_color="#ffffff")
    fig.update_layout(height=260, paper_bgcolor="#0f1419", font={'color': "white"},
                      margin=dict(l=40, r=40, t=50, b=10))
    return fig

def get_feature_contributions(scaled_row):
    """Explainable AI: how much each feature pushed toward fraud, using XGBoost contributions."""
    import xgboost as xgb
    booster = model.get_booster()
    try:
        dmatrix = xgb.DMatrix(scaled_row)
        contribs = booster.predict(dmatrix, pred_contribs=True)[0]  # last value is bias
    except Exception:
        # Fallback: use feature importances if contributions are unavailable
        importances = model.feature_importances_
        contribs = list(importances) + [0.0]
    feat_contribs = np.array(contribs[:-1])
    df = pd.DataFrame({"Feature": FEATURE_NAMES, "Contribution": feat_contribs})
    df["AbsContribution"] = df["Contribution"].abs()
    return df.sort_values("AbsContribution", ascending=False).head(8)

def make_contribution_chart(contrib_df):
    colors = ["#e74c3c" if c > 0 else "#27ae60" for c in contrib_df["Contribution"]]
    fig = go.Figure(go.Bar(
        x=contrib_df["Contribution"], y=contrib_df["Feature"], orientation='h',
        marker_color=colors, text=[f"{c:+.2f}" for c in contrib_df["Contribution"]], textposition='auto'
    ))
    fig.update_layout(
        title="Why this decision? Top feature contributions",
        height=320, paper_bgcolor="#0f1419", plot_bgcolor="#0f1419",
        font={'color': "white"}, margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(title="Push toward Fraud (red) / Safe (green)", gridcolor="#2a3142", color="#fff"),
        yaxis=dict(autorange="reversed", color="#fff")
    )
    return fig

def classify(prediction, fraud_prob, amount_usd):
    """Hybrid decision: ML result combined with business rule override."""
    if int(prediction) == 1 or fraud_prob >= 50.0:
        return "FRAUD"
    if amount_usd > HIGH_VALUE_THRESHOLD_USD:
        return "REVIEW"  # business rule override
    return "SAFE"

# ---------------------------------------------------------------------------
# Sidebar - input controls
# ---------------------------------------------------------------------------
st.sidebar.header("Transaction Input")

if 'txn_data' not in st.session_state:
    st.session_state.txn_data = sample_normal.copy()
if 'active_sample' not in st.session_state:
    st.session_state.active_sample = "Normal"
if 'history' not in st.session_state:
    st.session_state.history = []

col_a, col_b = st.sidebar.columns(2)

def load_sample(sample_dict, label):
    st.session_state.txn_data = sample_dict.copy()
    st.session_state.active_sample = label
    # Force the V1-V28 number_input widgets to take the new values
    for i in range(1, 29):
        st.session_state[f"in_V{i}"] = float(sample_dict[f"V{i}"])
    st.rerun()

if col_a.button("Sample Normal", use_container_width=True):
    load_sample(sample_normal, "Normal")
if col_b.button("Sample Fraud", use_container_width=True):
    load_sample(sample_fraud, "Fraud")

if st.session_state.active_sample == "Fraud":
    st.sidebar.markdown("<div style='background:#7d241c;padding:8px;border-radius:8px;text-align:center;font-weight:700;'>● Loaded: Sample Fraud</div>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<div style='background:#145a32;padding:8px;border-radius:8px;text-align:center;font-weight:700;'>● Loaded: Sample Normal</div>", unsafe_allow_html=True)

st.sidebar.markdown("---")
currency = st.sidebar.radio("Currency", ["USD ($)", "INR (₹)"], horizontal=True)
symbol = "$" if currency == "USD ($)" else "₹"

default_amount = float(st.session_state.txn_data['Amount'])
if currency == "INR (₹)":
    default_amount *= USD_TO_INR
amount_input = st.sidebar.number_input(f"Transaction Amount ({symbol})", value=round(default_amount, 2), step=10.0)
amount = amount_input / USD_TO_INR if currency == "INR (₹)" else amount_input
time_val = st.sidebar.number_input("Time (seconds since first transaction)", value=float(st.session_state.txn_data['Time']), step=1000.0)

with st.sidebar.expander("Advanced Features (V1 - V28)"):
    st.caption("Enter exact values. Default is 0.")
    # Initialize widget state from txn_data on first run
    for i in range(1, 29):
        wkey = f"in_V{i}"
        if wkey not in st.session_state:
            st.session_state[wkey] = float(st.session_state.txn_data[f"V{i}"])
    v_values = {}
    vcol1, vcol2 = st.columns(2)
    for i in range(1, 29):
        key = f"V{i}"
        target_col = vcol1 if i % 2 == 1 else vcol2
        v_values[key] = target_col.number_input(
            key, min_value=-30.0, max_value=30.0,
            step=0.1, format="%.2f", key=f"in_{key}")

st.sidebar.markdown("---")
st.sidebar.info(
    "**Model:** XGBoost Classifier\n\n"
    "**Dataset:** 284,807 Transactions\n\n"
    "**Fraud Cases:** 492\n\n"
    f"**Business Rule:** Amount > ${HIGH_VALUE_THRESHOLD_USD:,.0f} needs manual review\n\n"
    "**Imbalance Handling:** Class Weighting"
)

# ---------------------------------------------------------------------------
# Tabs: Single prediction | Bulk CSV
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🔍 Single Transaction", "📂 Bulk CSV Upload"])

# ===================== TAB 1: SINGLE TRANSACTION =====================
with tab1:
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("Transaction Summary")
        display_amount = amount * USD_TO_INR if currency == "INR (₹)" else amount
        rule_note = " (exceeds review limit)" if amount > HIGH_VALUE_THRESHOLD_USD else ""
        st.markdown(f"""
        <div class="info-box">
            <b>Transaction Type:</b> {st.session_state.active_sample} Sample<br>
            <b>Amount:</b> {symbol}{display_amount:,.2f}{rule_note}<br>
            <b>Time:</b> {time_val:,.0f} seconds<br>
            <b>Features analyzed:</b> 30 (Time, V1-V28, Amount)
        </div>
        """, unsafe_allow_html=True)
        predict_btn = st.button("Analyze Transaction", type="primary", use_container_width=True)

    with right_col:
        st.subheader("Prediction Result")
        result_ph = st.empty()
        result_ph.info("Click 'Analyze Transaction' to check this transaction.")

    if predict_btn:
        scaled_row = scale_row(time_val, amount, v_values)
        prediction = int(model.predict(scaled_row)[0])
        fraud_prob = float(model.predict_proba(scaled_row)[0][1] * 100)
        decision = classify(prediction, fraud_prob, amount)

        with right_col:
            if decision == "FRAUD":
                result_ph.markdown(f"<div class='result-fraud'>⚠️ FRAUD DETECTED<br><span style='font-size:18px;font-weight:500;'>Fraud Probability: {fraud_prob:.2f}%</span></div>", unsafe_allow_html=True)
            elif decision == "REVIEW":
                result_ph.markdown(f"<div class='result-review'>🔎 MANUAL REVIEW REQUIRED<br><span style='font-size:16px;font-weight:500;'>ML says low risk ({fraud_prob:.2f}%), but amount exceeds {symbol}{HIGH_VALUE_THRESHOLD_USD*(USD_TO_INR if currency=='INR (₹)' else 1):,.0f} business limit</span></div>", unsafe_allow_html=True)
            else:
                result_ph.markdown(f"<div class='result-safe'>✅ LEGITIMATE TRANSACTION<br><span style='font-size:18px;font-weight:500;'>Fraud Probability: {fraud_prob:.2f}%</span></div>", unsafe_allow_html=True)

        # Gauge in a centered narrower column so it does not over-size
        gcol1, gcol2, gcol3 = st.columns([1, 2, 1])
        with gcol2:
            st.plotly_chart(make_needle_gauge(fraud_prob), use_container_width=True)

        # Risk interpretation
        if fraud_prob < 30:
            st.success(f"**Low Risk** — appears legitimate ({fraud_prob:.1f}% fraud probability).")
        elif fraud_prob < 50:
            st.info(f"**Moderate Risk** — some suspicious patterns ({fraud_prob:.1f}%). Review recommended.")
        elif fraud_prob < 75:
            st.warning(f"**High Risk** — strong fraud indicators ({fraud_prob:.1f}%).")
        else:
            st.error(f"**Critical Risk** — very likely fraudulent ({fraud_prob:.1f}%).")

        # Explainable AI chart
        st.markdown("### 🧠 Explainable AI — Why this decision?")
        st.caption("Red bars pushed the transaction toward fraud; green bars toward legitimate. This opens the 'black box' of the model.")
        contrib_df = get_feature_contributions(scaled_row)
        st.plotly_chart(make_contribution_chart(contrib_df), use_container_width=True)

        # History + report
        hist_amount = display_amount
        st.session_state.history.append({
            "Amount": f"{symbol}{hist_amount:,.2f}",
            "Fraud Probability (%)": round(fraud_prob, 2),
            "Decision": decision
        })
        st.session_state.history = st.session_state.history[-20:]

        report_text = (
            "CREDIT CARD FRAUD DETECTION - ANALYSIS REPORT\n"
            "---------------------------------------------\n"
            f"Transaction Amount : {symbol}{hist_amount:,.2f}\n"
            f"Time               : {time_val:,.0f} seconds\n"
            f"Fraud Probability  : {fraud_prob:.2f}%\n"
            f"Final Decision     : {decision}\n"
            "---------------------------------------------\n"
            "Model: XGBoost Classifier | Dataset: 284,807 transactions\n"
        )
        st.download_button("Download Analysis Report", data=report_text, file_name="fraud_analysis_report.txt")

    # History table
    if st.session_state.history:
        st.markdown("---")
        st.subheader("Prediction History")
        h = pd.DataFrame(st.session_state.history)
        table_html = "<table style='width:100%;border-collapse:collapse;color:#e8ecf3;font-size:15px;'>"
        table_html += "<tr style='background:#21618c;'>" + "".join(f"<th style='padding:10px;text-align:left;border:1px solid #2a3142;'>{c}</th>" for c in h.columns) + "</tr>"
        for idx, row in h.iterrows():
            bg = "#1a1f2e" if idx % 2 == 0 else "#161b27"
            table_html += f"<tr style='background:{bg};'>"
            for c in h.columns:
                val = row[c]; color = "#e8ecf3"
                if c == "Decision":
                    color = "#e74c3c" if val == "FRAUD" else ("#f39c12" if val == "REVIEW" else "#2ecc71")
                table_html += f"<td style='padding:10px;border:1px solid #2a3142;color:{color};font-weight:600;'>{val}</td>"
            table_html += "</tr>"
        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)

# ===================== TAB 2: BULK CSV =====================
with tab2:
    st.subheader("📂 Bulk Transaction Processing")
    st.markdown("Upload a CSV with columns **Time, V1...V28, Amount** (the standard dataset format). The model will score every row at once, just like a real bank processing thousands of transactions.")

    sample_csv = pd.DataFrame([sample_normal, sample_fraud])[FEATURE_NAMES].to_csv(index=False)
    st.download_button("Download a sample CSV template", data=sample_csv, file_name="sample_transactions.csv")

    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
            missing = [c for c in FEATURE_NAMES if c not in df_in.columns]
            if missing:
                st.error(f"CSV is missing required columns: {missing}")
            else:
                work = df_in[FEATURE_NAMES].copy()
                # Scale Time and Amount, keep V columns as-is
                scaled = work.copy()
                scaled["Time"] = time_scaler.transform(work[["Time"]])
                scaled["Amount"] = amount_scaler.transform(work[["Amount"]])
                X = scaled[FEATURE_NAMES].values
                preds = model.predict(X)
                probs = model.predict_proba(X)[:, 1] * 100

                result = df_in.copy()
                result["Fraud_Probability_%"] = probs.round(2)
                decisions = []
                for p, prob, amt in zip(preds, probs, work["Amount"].values):
                    if int(p) == 1 or float(prob) >= 50.0:
                        decisions.append("FRAUD")
                    elif amt > HIGH_VALUE_THRESHOLD_USD:
                        decisions.append("REVIEW")
                    else:
                        decisions.append("SAFE")
                result["Decision"] = decisions

                total = len(result)
                fraud_count = decisions.count("FRAUD")
                review_count = decisions.count("REVIEW")
                safe_count = decisions.count("SAFE")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Transactions", f"{total}")
                c2.metric("Fraud Detected", f"{fraud_count}")
                c3.metric("Needs Review", f"{review_count}")
                c4.metric("Safe", f"{safe_count}")

                # Pie of outcomes
                pie = px.pie(names=["Safe", "Fraud", "Review"], values=[safe_count, fraud_count, review_count],
                             color=["Safe", "Fraud", "Review"],
                             color_discrete_map={"Safe": "#27ae60", "Fraud": "#e74c3c", "Review": "#f39c12"}, hole=0.45)
                pie.update_layout(height=320, paper_bgcolor="#0f1419", plot_bgcolor="#0f1419", font={'color': "white"})
                st.plotly_chart(pie, use_container_width=True)

                # Top suspicious transactions by fraud probability
                top_fraud = result.sort_values("Fraud_Probability_%", ascending=False).head(10).copy()
                if len(top_fraud) > 0:
                    st.markdown("#### Top Suspicious Transactions")
                    top_fraud["Label"] = ["Row " + str(idx) for idx in top_fraud.index]
                    bar = go.Figure(go.Bar(
                        x=top_fraud["Fraud_Probability_%"], y=top_fraud["Label"], orientation='h',
                        marker_color=top_fraud["Fraud_Probability_%"], marker_colorscale="Reds",
                        text=[f"{p:.1f}%" for p in top_fraud["Fraud_Probability_%"]], textposition='auto'
                    ))
                    bar.update_layout(
                        height=340, paper_bgcolor="#0f1419", plot_bgcolor="#0f1419", font={'color': "white"},
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis=dict(title="Fraud Probability (%)", gridcolor="#2a3142", color="#fff", range=[0, 100]),
                        yaxis=dict(autorange="reversed", color="#fff")
                    )
                    st.plotly_chart(bar, use_container_width=True)

                st.markdown("#### Results (fraud and review rows highlighted)")
                try:
                    def highlight(row):
                        if row["Decision"] == "FRAUD":
                            return ['background-color: rgba(231,76,60,0.25)'] * len(row)
                        if row["Decision"] == "REVIEW":
                            return ['background-color: rgba(243,156,18,0.25)'] * len(row)
                        return [''] * len(row)
                    st.dataframe(result.style.apply(highlight, axis=1), use_container_width=True, height=360)
                except Exception:
                    st.dataframe(result, use_container_width=True, height=360)

                st.download_button("Download Results CSV", data=result.to_csv(index=False),
                                   file_name="fraud_results.csv")
        except Exception as e:
            st.error(f"Could not process the file: {e}")

# ---------------------------------------------------------------------------
# Footer - performance metrics and model comparison
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Model Performance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", "99.95%")
m2.metric("Precision", "88.17%")
m3.metric("Recall", "83.67%")
m4.metric("F1-Score", "85.86%")

with st.expander("Model Comparison — how three models performed"):
    st.caption("Three algorithms were trained and compared. XGBoost was selected as the final model based on the best F1-score.")
    comp = go.Figure()
    metrics = ["Precision", "Recall", "F1-Score"]
    comp.add_trace(go.Bar(name="Logistic Regression", x=metrics, y=[6.09, 91.84, 11.41], marker_color="#7f8c8d"))
    comp.add_trace(go.Bar(name="Random Forest", x=metrics, y=[96.05, 74.49, 83.91], marker_color="#2980b9"))
    comp.add_trace(go.Bar(name="XGBoost (Selected)", x=metrics, y=[88.17, 83.67, 85.86], marker_color="#e74c3c"))
    comp.update_layout(
        barmode="group", height=360, paper_bgcolor="#0f1419", plot_bgcolor="#0f1419",
        font={'color': "white"}, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(font=dict(color="#fff")),
        xaxis=dict(color="#fff", gridcolor="#2a3142"),
        yaxis=dict(title="Score (%)", color="#fff", gridcolor="#2a3142", range=[0, 100])
    )
    st.plotly_chart(comp, use_container_width=True)

st.markdown("""
<div class="footer-card">
    <h4>Vikash Kumar</h4>
    <p>BCA, Semester VI &nbsp;|&nbsp; Amity University Online</p>
</div>
""", unsafe_allow_html=True)
