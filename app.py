import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
import joblib
from src.data.cleaner import DataCleaner
from src.features.engineer import FeatureEngineer

meta_model = joblib.load("models/meta_model.pkl")
scaler_stack = joblib.load("models/scaler.pkl")
rf_model = joblib.load("models/rf_model.pkl")
iso_model = joblib.load("models/iso_model.pkl")
kl_model = joblib.load("models/kl_model.pkl")
features = joblib.load("models/features.pkl")
cleaner = DataCleaner(label_column="label")
engineer = FeatureEngineer()
st.set_page_config(page_title="IoT IDS Dashboard", layout="wide")

st.title("🚨 Real-Time IoT Intrusion Detection System")
def run_model_on_chunk(df):


 
    df = engineer.add_all_features(df)

    df = df.replace([float("inf"), float("-inf")], 0)
    df = df.fillna(0)


    df_selected = df.reindex(columns=features, fill_value=0)
    for col in df_selected.columns:
        if df_selected[col].dtype == object:
            df_selected[col] = df_selected[col].apply(
                lambda x: int(x, 16) if isinstance(x, str) and x.startswith("0x") else x
            )

    df_selected = df_selected.apply(pd.to_numeric, errors='coerce').fillna(0)

    df_scored = iso_model.compute_scores(df_selected)
    df_scored = kl_model.compute(df_scored)

    X_rf = df_scored.drop(columns=["label"], errors="ignore")
    rf_probs = rf_model.predict_proba(X_rf)

    if len(rf_probs.shape) > 1:
        rf_probs = rf_probs[:, -1]


    stack_X = pd.DataFrame({
        "rf": rf_probs,
        "iso": df_scored["anomaly_score"],
        "kl": df_scored["kl_score"]
    })

    stack_X_scaled = scaler_stack.transform(stack_X)

    probs = meta_model.predict_proba(stack_X_scaled)[:, 1]

    probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-8)

    return probs

if "threshold" not in st.session_state:
    st.session_state.threshold = 0.5

if "history" not in st.session_state:
    st.session_state.history = []
if "threshold_history" not in st.session_state:
    st.session_state.threshold_history = []

col1, col2, col3, col4 = st.columns(4)

kpi_threshold = col1.empty()
kpi_detect = col2.empty()
kpi_alerts = col4.empty()
status_placeholder = st.empty()
explanation_placeholder = st.empty()

chart_placeholder = st.empty()
hist_placeholder = st.empty()
table_placeholder = st.empty()
threshold_chart = st.empty()
progress_bar = st.empty()

def generate_probs(n=1000):
    normal = np.random.normal(0.5, 0.01, int(n * 0.9))
    attack = np.random.normal(0.6, 0.05, int(n * 0.1))

    probs = np.clip(np.concatenate([normal, attack]), 0, 1)
    labels = np.array([0]*len(normal) + [1]*len(attack))

    return probs, labels

st.markdown("### 📊 What am I seeing?")
st.markdown("""
- 🔵 Cyan = Normal traffic  
- 🔴 Red = Attack  
- 🟡 Yellow line = Threshold  

Any point above the threshold is classified as an attack.
""")
st.markdown("""
    ### 📈 Score Distribution Insight

    - Left cluster → Normal traffic  
    - Right tail → Suspicious / attacks  
    - Threshold separates normal from anomalous behavior  
    """)
for step in range(50):
    df_chunk = pd.read_csv("data/raw/test30.csv").sample(500)


    df_clean = cleaner.clean_chunk(df_chunk)

    true_labels = df_clean["label"].copy()
    true_labels = (true_labels != 3).astype(int)

    probs = run_model_on_chunk(df_clean)

    target_rate = max(0.04, min(0.10, probs.mean()))
    new_threshold = np.percentile(probs, 100 * (1 - target_rate))
    alpha = 0.2
    st.session_state.threshold = (
        (1 - alpha) * st.session_state.threshold
        + alpha * new_threshold
    )

    threshold = st.session_state.threshold
    st.session_state.threshold_history.append(threshold)

    if len(st.session_state.threshold_history) > 100:
        st.session_state.threshold_history.pop(0)
    preds = (probs > threshold).astype(int)


    alerts = preds.sum()
    detection_rate = alerts / len(preds)
    fp = ((preds == 1) & (true_labels == 0)).sum()
    tn = ((preds == 0) & (true_labels == 0)).sum()


    kpi_threshold.metric("Threshold", f"{threshold:.3f}", delta=f"{threshold-0.5:.3f}")   
    kpi_detect.metric("Attack Rate", f"{detection_rate*100:.1f}%")

    kpi_alerts.metric("Alerts", int(alerts))
    progress_bar.progress(min(detection_rate, 1.0))
    if detection_rate > 0.15:
        explanation = f"""
    🚨 High attack activity detected.

    - Attack rate: {detection_rate*100:.1f}%
    - Threshold: {threshold:.3f}

    The system is detecting a large number of anomalous behaviors above the threshold.
    Immediate investigation is recommended.
    """

    elif detection_rate > 0.08:
        explanation = f"""
    ⚠️ Suspicious behavior observed.

    - Attack rate: {detection_rate*100:.1f}%
    - Threshold: {threshold:.3f}

    There is an increase in abnormal traffic patterns.
    This may indicate early-stage attacks or scanning activity.
    """

    else:
        explanation = f"""
    ✅ System operating normally.

    - Attack rate: {detection_rate*100:.1f}%
    - Threshold: {threshold:.3f}

    Most traffic is within normal behavior.
    No immediate threats detected.
    """
        
    explanation_placeholder.info(explanation)

    fig = go.Figure()


    fig.add_trace(go.Scatter(
        x=list(range(len(probs[preds == 0][:200]))),
        y=probs[preds == 0][:200],
        mode='markers',
        name='Normal',
        marker=dict(color='blue', size=6)
    ))


    fig.add_trace(go.Scatter(
        x=list(range(len(probs[preds == 1][:200]))),
        y=probs[preds == 1][:200],
        mode='markers',
        name='Attack',
        marker=dict(color='red', size=7)
    ))


    fig.add_trace(go.Scatter(
        x=list(range(200)),
        y=[threshold]*200,
        mode='lines',
        name='Threshold',
        line=dict(color='green', dash='dash', width=2)
    ))
    fig.update_layout(
        title="Live Traffic Classification",
        xaxis_title="Packets",
        yaxis_title="Anomaly Score",
        template="plotly_white"
    )
    chart_placeholder.plotly_chart(fig, use_container_width=True)
    

    hist = go.Figure()

    hist.add_histogram(x=probs, nbinsx=50, name="Scores")

    hist.add_vline(
        x=threshold,
        line_dash="dash",
        annotation_text="Threshold"
    )

    hist_placeholder.plotly_chart(hist, use_container_width=True)
    

    attack_scores = probs[preds == 1][:10]

    df_alerts = pd.DataFrame({
        "Score": attack_scores,
        "Severity": [
            "HIGH" if s > threshold + 0.1 else "MEDIUM"
            for s in attack_scores
        ]
    })

    table_placeholder.dataframe(df_alerts)
    threshold_chart.line_chart(st.session_state.threshold_history)
    time.sleep(1)
