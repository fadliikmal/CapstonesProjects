# Real-Time Financial Fraud Detection Platform

A real-time financial fraud detection system powered by **XGBoost**, **FastAPI**, **Streamlit Dashboard**, **Evidently AI MLOps**, **Docker Compose**, and **Kubernetes**.

---

## 📌 Project Overview

This platform provides low-latency (< 100 ms) fraud scoring for streaming financial transactions. The system utilizes an **XGBoost** machine learning model trained on transaction aggregation and velocity features, integrated with a **3-Tier Risk Engine** to execute operational actions automatically.

---

## 🚀 Key Features

- **Low-Latency Inference Engine (FastAPI)**: High-performance REST API for real-time fraud scoring with automatic categorical handling & missing value imputation.
- **3-Tier Risk Decision Engine**:
  - 🟢 **LOW Risk (`< 0.10`)** $\rightarrow$ `APPROVE` (Automatically approved)
  - 🟡 **MEDIUM Risk (`0.10 – 0.69`)** $\rightarrow$ `CHALLENGE_REVIEW` (Sent to analyst case review queue)
  - 🔴 **HIGH Risk (`>= 0.70`)** $\rightarrow$ `AUTO_BLOCK` (Automatically blocked)
- **Real-Time Stream Producer**: Streaming transaction replay script reading from Parquet files with minimal memory footprint (~2 MB RAM).
- **Interactive Analyst Dashboard (Streamlit)**:
  1. **Live Stream Monitor**: Real-time KPI metrics, fraud probability trend charts, and decision breakdowns.
  2. **Analyst Case Review Queue**: Case management interface for suspicious transactions with quick disposition (*Confirm Fraud / False Alarm*).
  3. **Real-Time Transaction Simulator**: Test the `/predict` endpoint directly using a custom transaction form.
  4. **MLOps & Data Drift Status**: Visualizations of feature drift metrics and prediction population drift (*PSI*).
- **Evidently AI Data & Prediction Drift Monitoring**: Scheduled automated data drift detection and Population Stability Index (PSI) tracking to trigger model retraining workflows.
- **Multi-Environment Deployment**: Support for local Python environment, **Docker Compose**, and **Kubernetes (Minikube)** cluster execution.

---

## 📁 Directory Structure

```text
fraud-detection/
├── app/
│   └── main.py                     # FastAPI backend application & 3-Tier Risk Engine
├── artifacts/
│   ├── category_mappings.json      # Numeric/Categorical feature dictionary mapping
│   ├── model_metadata.json         # Model feature list and threshold metadata
│   └── xgb_fraud_model.json        # Trained XGBoost Booster model
├── dashboard/
│   └── app.py                      # Streamlit Analyst & Operations Dashboard
├── data/
│   ├── test_engineered.parquet     # Test dataset for streaming replay (Parquet)
│   └── train_engineered.parquet    # Training dataset (Parquet)
├── k8s/
│   └── deployment.yaml             # Kubernetes Deployment & Service manifests
├── monitoring/
│   └── drift_report.py             # Evidently AI Data & Prediction Drift Monitoring script
├── reports/
│   ├── evidently_drift_report.html # Interactive Evidently AI HTML report
│   ├── evidently_drift_report.json # Detailed drift metrics JSON
│   └── retrain_trigger_status.json # Automated retraining trigger log
├── streaming/
│   └── producer.py                 # Real-time transaction replay producer simulator
├── .dockerignore                   # Docker build exclusion rules
├── Dockerfile                      # Python 3.11-slim container image specification
├── dockercompose.yml               # Multi-container service setup (API & Dashboard)
├── README.md                       # Main project documentation
├── requirements.txt                # Python project dependencies
└── runk8s.ps1                      # Minikube automated deployment script
```

---

## ⚙️ Thresholds & Risk Engine

| Risk Tier | Probability Range | Default Action | Operational Meaning |
| :--- | :--- | :--- | :--- |
| **LOW** | `< 0.10` | `APPROVE` | Transaction cleared and processed without friction |
| **MEDIUM** | `0.10 – 0.69` | `CHALLENGE_REVIEW` | Requires step-up authentication or analyst review |
| **HIGH** | `>= 0.70` | `AUTO_BLOCK` | Transaction immediately blocked automatically |

---

## 🛠️ How to Run

### 1. Run Locally (Virtual Environment)

#### A. Environment Setup
```bash
python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

#### B. Start FastAPI Backend
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
API running at: `http://127.0.0.1:8000`  
Swagger OpenAPI Docs: `http://127.0.0.1:8000/docs`

#### C. Start Streamlit Dashboard
Open a new terminal (with venv activated):
```bash
streamlit run dashboard/app.py --server.port 8501
```
Dashboard running at: `http://localhost:8501`

#### D. Start Stream Producer Simulator
Open another terminal to stream transaction data:
```bash
python streaming/producer.py --events 200 --delay 0.10
```

---

### 2. Run with Docker Compose

Multi-container orchestration setup linking the API engine and Dashboard with automatic health checks:

```bash
# Build and run containers in background
docker compose up --build -d

# Check container status
docker compose ps

# Stop containers
docker compose down
```
- **API Engine**: `http://localhost:8000`
- **Dashboard**: `http://localhost:8501`

---

### 3. Run on Kubernetes (Minikube)

Ensure Minikube and `kubectl` are installed.

#### Option A: Using PowerShell Script (`runk8s.ps1`)
```powershell
.\runk8s.ps1
```

#### Option B: Manual Commands
```bash
# 1. Start Minikube cluster
minikube start --driver=docker --memory=2500mb --cpus=2

# 2. Apply Kubernetes manifests
kubectl apply -f k8s/deployment.yaml

# 3. Wait for all pods to be ready
kubectl wait --for=condition=Ready pods --all --timeout=180s

# 4. Expose Services via Port Forwarding
kubectl port-forward svc/fraud-api-service 8000:8000
minikube service fraud-dashboard-service
```

---

## 📊 Data Drift & MLOps Monitoring (Evidently AI)

Evaluate distribution shifts (*Data Drift*) and score prediction drift (*PSI*) between reference training data and live streaming data:

```bash
python monitoring/drift_report.py
```

### Monitoring Artifacts Generated:
1. **Interactive HTML Report**: `reports/evidently_drift_report.html` (Viewable in any web browser).
2. **Drift Metrics Summary**: `reports/evidently_drift_report.json`.
3. **Automated Retrain Log**: `reports/retrain_trigger_status.json`.

*Automated Retraining Criteria:*
- Feature Drift Share `>= 30%` of monitored features, **or**
- Prediction PSI (Population Stability Index) `>= 0.20`.

---

## 🔌 API Endpoint Specifications

| Method | Endpoint | Description | Example Payload / Response |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Check service health and configured risk thresholds | `{"status": "HEALTHY", "model_loaded": true}` |
| `POST` | `/predict` | Score a single incoming transaction | JSON payload (`TransactionID`, `TransactionAmt`, `features`) |
| `GET` | `/recent` | Fetch recent prediction telemetry buffer (up to 500) | `{"count": 50, "transactions": [...]}` |

### Request Payload Example (`POST /predict`):
```json
{
  "TransactionID": 3000001,
  "TransactionAmt": 150.50,
  "features": {
    "TransactionAmt": 150.50,
    "ProductCD": "W",
    "card1": 13926,
    "card4": "visa",
    "C1": 1.0,
    "C11": 1.0,
    "V258": 1.0,
    "P_emaildomain": "gmail.com"
  }
}
```

### Response Example (`POST /predict`):
```json
{
  "transaction_id": 3000001,
  "transaction_amt": 150.5,
  "fraud_probability": 0.0245,
  "risk_tier": "LOW",
  "action": "APPROVE",
  "latency_ms": 8.42,
  "timestamp": "2026-11-20 20:00:00"
}
```
