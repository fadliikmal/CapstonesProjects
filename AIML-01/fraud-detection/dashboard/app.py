import json
import os
import time
import pandas as pd
import requests
import streamlit as st

# Dashboard page configuration
st.set_page_config(
    page_title="Real-Time Fraud Detection Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIGGER_STATUS_PATH = os.path.join(BASE_DIR, "reports", "retrain_trigger_status.json")
DRIFT_HTML_PATH = os.path.join(BASE_DIR, "reports", "evidently_drift_report.html")

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# Store analyst case review dispositions in session state
if "analyst_decisions" not in st.session_state:
    st.session_state.analyst_decisions = {}


def fetch_api_health() -> dict:
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=2)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "OFFLINE", "error": str(e)}


def fetch_recent_transactions(limit: int = 200) -> list:
    try:
        resp = requests.get(f"{API_BASE_URL}/recent", params={"limit": limit}, timeout=3)
        resp.raise_for_status()
        return resp.json().get("transactions", [])
    except Exception:
        return []


def load_drift_status() -> dict:
    if os.path.exists(TRIGGER_STATUS_PATH):
        with open(TRIGGER_STATUS_PATH, "r") as f:
            return json.load(f)
    return {}


# --- SIDEBAR: SYSTEM HEALTH & CONTROLS ---
st.sidebar.title("Fraud Ops Control")
health = fetch_api_health()

if health.get("status") == "HEALTHY":
    st.sidebar.success(f"FastAPI Engine: **ONLINE** ({health.get('total_features', 0)} features)")
else:
    st.sidebar.error("FastAPI Engine: **OFFLINE**")
    st.sidebar.caption("Start the server first: `uvicorn app.main:app --port 8000`")

st.sidebar.markdown("---")
st.sidebar.subheader("3-Tier Risk Thresholds")
st.sidebar.info(
    "- **LOW (`< 0.10`)**: `APPROVE`\n"
    "- **MEDIUM (`0.10 – 0.69`)**: `CHALLENGE_REVIEW`\n"
    "- **HIGH (`>= 0.70`)**: `AUTO_BLOCK`"
)

auto_refresh = st.sidebar.checkbox("Auto-refresh stream (2s)", value=False)
if st.sidebar.button("Refresh Data Now", use_container_width=True):
    st.rerun()

# --- MAIN HEADER ---
st.title("Real-Time Financial Fraud Detection Platform")
st.caption("End-to-End Streaming Risk Scoring • XGBoost Engine • 3-Tier Decisioning • Evidently MLOps")

tab_live, tab_review, tab_sim, tab_mlops = st.tabs([
    "Live Stream Monitor",
    "Analyst Case Review Queue",
    "Real-Time Transaction Simulator",
    "MLOps & Data Drift Status",
])

transactions = fetch_recent_transactions(limit=200)
df_tx = pd.DataFrame(transactions) if transactions else pd.DataFrame()

# ==========================================
# TAB 1: LIVE STREAM MONITOR
# ==========================================
with tab_live:
    if df_tx.empty:
        st.warning(
            "No transactions in the FastAPI buffer yet. Run `python streaming/producer.py --events 100 --delay 0.15` "
            "in a separate terminal to start streaming transactions."
        )
    else:
        total_cnt = len(df_tx)
        approve_cnt = int((df_tx["action"] == "APPROVE").sum())
        review_cnt = int((df_tx["action"] == "CHALLENGE_REVIEW").sum())
        block_cnt = int((df_tx["action"] == "AUTO_BLOCK").sum())
        flagged_vol = float(df_tx[df_tx["action"] != "APPROVE"]["transaction_amt"].sum())
        avg_lat = float(df_tx["latency_ms"].mean())

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Processed Stream", f"{total_cnt:,} tx")
        col2.metric("Approved (Low Risk)", f"{approve_cnt:,}", f"{approve_cnt/total_cnt*100:.1f}%")
        col3.metric("Challenge / Review", f"{review_cnt:,}", f"{review_cnt/total_cnt*100:.1f}%")
        col4.metric("Auto-Blocked (High Risk)", f"{block_cnt:,}", f"${flagged_vol:,.2f} protected")
        col5.metric("Avg Inference Latency", f"{avg_lat:.2f} ms", "SLA < 100 ms")

        st.markdown("---")
        c_left, c_right = st.columns([1, 2])

        with c_left:
            st.subheader("Decision Breakdown")
            action_summary = df_tx["action"].value_counts().reindex(
                ["APPROVE", "CHALLENGE_REVIEW", "AUTO_BLOCK"], fill_value=0
            )
            st.bar_chart(action_summary)

        with c_right:
            st.subheader("Fraud Probability Trend (Recent Stream)")
            chart_df = df_tx.iloc[::-1].reset_index(drop=True)[["fraud_probability"]]
            st.area_chart(chart_df)

        st.subheader("Recent Scored Transactions")
        st.dataframe(
            df_tx[[
                "timestamp", "transaction_id", "transaction_amt",
                "fraud_probability", "risk_tier", "action", "latency_ms"
            ]],
            use_container_width=True,
            hide_index=True,
        )

# ==========================================
# TAB 2: ANALYST CASE REVIEW QUEUE
# ==========================================
with tab_review:
    st.subheader("Flagged Transactions Requiring Investigation")
    if df_tx.empty:
        st.info("No incoming transaction data available yet.")
    else:
        flagged_df = df_tx[df_tx["action"].isin(["CHALLENGE_REVIEW", "AUTO_BLOCK"])].copy()
        if flagged_df.empty:
            st.success("No medium- or high-risk transactions detected in the current batch.")
        else:
            flagged_df["analyst_status"] = flagged_df["transaction_id"].map(
                lambda x: st.session_state.analyst_decisions.get(x, "PENDING_REVIEW")
            )
            st.dataframe(
                flagged_df[[
                    "timestamp", "transaction_id", "transaction_amt",
                    "fraud_probability", "risk_tier", "action", "analyst_status"
                ]],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Quick Case Disposition")
            selected_tx = st.selectbox(
                "Select Transaction ID for case disposition:",
                options=flagged_df["transaction_id"].tolist(),
            )
            b1, b2, _ = st.columns([1, 1, 3])
            if b1.button("Confirm Fraud (Block Card)", type="primary"):
                st.session_state.analyst_decisions[selected_tx] = "CONFIRMED_FRAUD"
                st.rerun()
            if b2.button("False Alarm (Approve Tx)"):
                st.session_state.analyst_decisions[selected_tx] = "CLEARED_LEGITIMATE"
                st.rerun()

# ==========================================
# TAB 3: REAL-TIME TRANSACTION SIMULATOR
# ==========================================
with tab_sim:
    st.subheader("Test Live Scoring Endpoint (`POST /predict`)")
    st.caption("Submit a custom simulated transaction to test the 3-tier risk engine response in real time.")

    with st.form("tx_simulator_form"):
        s1, s2, s3 = st.columns(3)
        sim_tx_id = s1.number_input("TransactionID", value=9900001, step=1)
        sim_amt = s2.number_input("TransactionAmt ($)", value=450.00, min_value=0.1, step=10.0)
        sim_product = s3.selectbox("ProductCD", ["W", "H", "C", "S", "R"])

        s4, s5, s6 = st.columns(3)
        sim_c1 = s4.number_input("C1 (Entity Count)", value=1.0, min_value=0.0, step=1.0)
        sim_c11 = s5.number_input("C11 (Device/Addr Count)", value=1.0, min_value=0.0, step=1.0)
        sim_v258 = s6.number_input("V258 (Vesta Risk Signal)", value=1.0, min_value=0.0, step=1.0)

        submitted = st.form_submit_button("Send to FastAPI /predict", type="primary")

    if submitted:
        payload = {
            "TransactionID": int(sim_tx_id),
            "TransactionAmt": float(sim_amt),
            "features": {
                "TransactionAmt": float(sim_amt),
                "ProductCD": sim_product,
                "C1": float(sim_c1),
                "C11": float(sim_c11),
                "V258": float(sim_v258),
            },
        }
        try:
            res = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=5)
            res.raise_for_status()
            out = res.json()
            st.success(
                f"**Prediction Result:** Action = `{out['action']}` | "
                f"Risk Tier = `{out['risk_tier']}` | "
                f"Probability = `{out['fraud_probability']:.4f}` | "
                f"Latency = `{out['latency_ms']} ms`"
            )
        except Exception as err:
            st.error(f"Failed to reach FastAPI service: {err}")

# ==========================================
# TAB 4: MLOPS & EVIDENTLY DRIFT STATUS
# ==========================================
with tab_mlops:
    st.subheader("Evidently AI Data & Prediction Drift Telemetry")
    drift_info = load_drift_status()

    if not drift_info:
        st.info("Run `python monitoring/drift_report.py` first to generate the drift monitoring report.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Drifted Features",
            f"{drift_info.get('drifted_columns_count', 0)} / {drift_info.get('monitored_columns', 0)}",
            f"{drift_info.get('drift_share', 0)*100:.2f}% share",
        )
        m2.metric(
            "Prediction Drift (PSI)",
            f"{drift_info.get('prediction_psi', 0):.4f}",
            f"Threshold < {drift_info.get('thresholds', {}).get('max_prediction_psi', 0.2)}",
        )
        triggered = drift_info.get("retrain_triggered", False)
        m3.metric(
            "Retrain Trigger Status",
            "TRIGGERED" if triggered else "STABLE",
            drift_info.get("pipeline_action", ""),
        )
        m4.metric("Evaluated Samples", f"{drift_info.get('current_samples', 0):,} rows")

        st.json(drift_info)
        st.caption(f"Full interactive HTML drift report saved at: `{DRIFT_HTML_PATH}`")

# Auto-refresh loop for live streaming demo
if auto_refresh:
    time.sleep(2)
    st.rerun()