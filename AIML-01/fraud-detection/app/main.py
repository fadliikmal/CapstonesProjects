import os
import json
import time
import traceback
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Base paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "xgb_fraud_model.json")
METADATA_PATH = os.path.join(BASE_DIR, "artifacts", "model_metadata.json")
CAT_MAPPING_PATH = os.path.join(BASE_DIR, "artifacts", "category_mappings.json")
TRAIN_PARQUET_PATH = os.path.join(BASE_DIR, "data", "train_engineered.parquet")

# Global state for loaded artifacts and rolling telemetry buffer
ml_artifacts: Dict[str, Any] = {}
recent_predictions: deque = deque(maxlen=500)


def load_or_build_category_mappings(cat_cols: List[str]) -> Dict[str, List[Any]]:
    """Load cached training category dictionaries or extract them once from train_engineered.parquet."""
    if os.path.exists(CAT_MAPPING_PATH):
        with open(CAT_MAPPING_PATH, "r") as f:
            return json.load(f)

    if not os.path.exists(TRAIN_PARQUET_PATH):
        raise RuntimeError(
            f"Neither {CAT_MAPPING_PATH} nor {TRAIN_PARQUET_PATH} found. "
            "Place train_engineered.parquet in data/ to build category mappings."
        )

    print("Building category mappings from 80% temporal training split (one-time setup)...")
    df_train_cats = pd.read_parquet(TRAIN_PARQUET_PATH, columns=cat_cols + ["TransactionDT"])
    df_train_cats = df_train_cats.sort_values("TransactionDT").reset_index(drop=True)
    split_idx = int(len(df_train_cats) * 0.80)
    df_tr_split = df_train_cats.iloc[:split_idx]

    mappings: Dict[str, List[Any]] = {}
    for col in cat_cols:
        cats = df_tr_split[col].astype("category").cat.categories.tolist()
        mappings[col] = cats if len(cats) > 0 else ["missing"]

    with open(CAT_MAPPING_PATH, "w") as f:
        json.dump(mappings, f)

    print(f"Category mappings cached to: {CAT_MAPPING_PATH}")
    return mappings


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        raise RuntimeError("Model artifacts missing in artifacts/ directory.")

    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    cat_mappings = load_or_build_category_mappings(metadata["categorical_features"])

    ml_artifacts["model"] = booster
    ml_artifacts["metadata"] = metadata
    ml_artifacts["cat_mappings"] = cat_mappings
    print("FastAPI Fraud Detection Service initialized and ready!")
    yield
    ml_artifacts.clear()


app = FastAPI(
    title="Real-Time Financial Fraud Detection API",
    description="Low-latency fraud scoring service powered by XGBoost & 3-Tier Risk Engine",
    version="1.0.0",
    lifespan=lifespan,
)


class TransactionRequest(BaseModel):
    TransactionID: int
    TransactionAmt: float
    features: Dict[str, Any]


def evaluate_risk_tier(prob: float, low_thresh: float = 0.10, high_thresh: float = 0.70) -> Dict[str, str]:
    """Map predicted fraud probability into operational 3-Tier Risk Actions."""
    if prob >= high_thresh:
        return {"risk_tier": "HIGH", "action": "AUTO_BLOCK"}
    elif prob >= low_thresh:
        return {"risk_tier": "MEDIUM", "action": "CHALLENGE_REVIEW"}
    return {"risk_tier": "LOW", "action": "APPROVE"}


@app.get("/health")
def health_check():
    return {
        "status": "HEALTHY",
        "model_loaded": "model" in ml_artifacts,
        "total_features": len(ml_artifacts["metadata"]["feature_names"]) if "metadata" in ml_artifacts else 0,
        "thresholds": {"low_screening": 0.10, "high_autoblock": 0.70},
    }


@app.post("/predict")
def predict_fraud(payload: TransactionRequest):
    start_time = time.perf_counter()

    try:
        booster: xgb.Booster = ml_artifacts["model"]
        metadata: Dict[str, Any] = ml_artifacts["metadata"]
        cat_mappings: Dict[str, List[Any]] = ml_artifacts["cat_mappings"]

        feature_names: List[str] = metadata["feature_names"]
        cat_features_set = set(metadata["categorical_features"])

        # Build single-row DataFrame aligned strictly to feature_names
        row_data = {col: payload.features.get(col, np.nan) for col in feature_names}
        df_single = pd.DataFrame([row_data], columns=feature_names)

        # Enforce strict dtypes: Categorical for cat_features, float32 for all numeric features
        for col in feature_names:
            if col in cat_features_set:
                cats = cat_mappings.get(col) or ["missing"]
                # Match numeric category type if training categories are float/int
                if isinstance(cats[0], (int, float)) and not isinstance(cats[0], bool):
                    df_single[col] = pd.to_numeric(df_single[col], errors="coerce")
                df_single[col] = pd.Categorical(df_single[col], categories=cats)
            else:
                df_single[col] = pd.to_numeric(df_single[col], errors="coerce").astype(np.float32)

        dmatrix = xgb.DMatrix(df_single, enable_categorical=True)
        fraud_prob = float(booster.predict(dmatrix)[0])

        decision = evaluate_risk_tier(fraud_prob, low_thresh=0.10, high_thresh=0.70)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        result = {
            "transaction_id": payload.TransactionID,
            "transaction_amt": round(payload.TransactionAmt, 2),
            "fraud_probability": round(fraud_prob, 4),
            "risk_tier": decision["risk_tier"],
            "action": decision["action"],
            "latency_ms": round(latency_ms, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        recent_predictions.appendleft(result)
        return result

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recent")
def get_recent_predictions(limit: int = 50):
    """Return recent scored transactions for live monitoring dashboard."""
    items = list(recent_predictions)[:limit]
    return {"count": len(items), "transactions": items}