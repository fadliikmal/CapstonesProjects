import gc
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pandas.core.dtypes.common as _pd_common
import pyarrow.parquet as pq

# Compatibility shim for Evidently on Pandas 3.0+ where infer_dtype_from_object was removed
if not hasattr(_pd_common, "infer_dtype_from_object"):
    def _infer_dtype_from_object(dtype):
        if isinstance(dtype, type) and issubclass(dtype, np.generic):
            return dtype
        if hasattr(dtype, "type"):
            return dtype.type
        try:
            return np.dtype(dtype).type
        except Exception:
            return object

    _pd_common.infer_dtype_from_object = _infer_dtype_from_object

import xgboost as xgb
from evidently import Report
from evidently.presets import DataDriftPreset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "xgb_fraud_model.json")
METADATA_PATH = os.path.join(BASE_DIR, "artifacts", "model_metadata.json")
CAT_MAPPING_PATH = os.path.join(BASE_DIR, "artifacts", "category_mappings.json")
TRAIN_PARQUET_PATH = os.path.join(BASE_DIR, "data", "train_engineered.parquet")
TEST_PARQUET_PATH = os.path.join(BASE_DIR, "data", "test_engineered.parquet")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Top SHAP features & key business columns to monitor
MONITORED_FEATURES = [
    "TransactionAmt",
    "C1",
    "C11",
    "C13",
    "C14",
    "V258",
    "card1_fq_enc",
    "tx_count_24h",
    "tx_count_7d",
    "tx_sum_amt_1h",
    "tx_sum_amt_7d",
    "velocity_7d_count",
    "hour_cos",
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
]

DRIFT_SHARE_THRESHOLD = 0.30  # Trigger retrain if >= 30% of monitored features drift
PSI_THRESHOLD = 0.20          # Trigger retrain if Prediction PSI >= 0.20


def read_parquet_slice(path: str, target_rows: int = 5000, skip_ratio: float = 0.0) -> pd.DataFrame:
    """Stream only a lightweight slice from Parquet without loading the full 1GB+ table into RAM."""
    pf = pq.ParquetFile(path)
    total_rows = pf.metadata.num_rows
    rows_to_skip = int(total_rows * skip_ratio)
    skipped = 0

    for batch in pf.iter_batches(batch_size=target_rows):
        if skipped + len(batch) <= rows_to_skip:
            skipped += len(batch)
            continue
        return batch.to_pandas().copy()

    first_batch = next(pf.iter_batches(batch_size=target_rows))
    return first_batch.to_pandas().copy()


def prepare_for_inference(
    df: pd.DataFrame,
    feature_names: List[str],
    cat_features: List[str],
    cat_mappings: Dict[str, List[Any]],
) -> pd.DataFrame:
    """Align DataFrame schema and dtypes strictly with XGBoost model requirements."""
    X = df.reindex(columns=feature_names).copy()
    cat_set = set(cat_features)

    for col in feature_names:
        if col in cat_set:
            cats = cat_mappings.get(col) or ["missing"]
            if isinstance(cats[0], (int, float)) and not isinstance(cats[0], bool):
                X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = pd.Categorical(X[col], categories=cats)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(np.float32)

    return X


def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Calculate Population Stability Index (PSI) for prediction drift tracking."""
    breakpoints = np.linspace(0, 1, buckets + 1)
    expected_counts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_counts = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    expected_counts = np.where(expected_counts == 0, 1e-4, expected_counts)
    actual_counts = np.where(actual_counts == 0, 1e-4, actual_counts)

    psi_val = np.sum((actual_counts - expected_counts) * np.log(actual_counts / expected_counts))
    return float(psi_val)


def parse_evidently_drift_summary(report_dict: Dict[str, Any], total_cols: int) -> Tuple[int, int, float]:
    """Extract drifted column count and share cleanly from Evidently JSON output."""
    metrics = report_dict.get("metrics", [])
    if not metrics:
        return total_cols, 0, 0.0

    first_metric = metrics[0]

    # Evidently v0.7+ schema: {"value": {"count": ..., "share": ...}}
    if "value" in first_metric and isinstance(first_metric["value"], dict):
        val = first_metric["value"]
        n_drifted = int(val.get("count", 0))
        drift_share = float(val.get("share", n_drifted / max(total_cols, 1)))
        return total_cols, n_drifted, drift_share

    # Legacy Evidently schema fallback
    if "result" in first_metric and isinstance(first_metric["result"], dict):
        res = first_metric["result"]
        n_cols = int(res.get("number_of_columns", total_cols))
        n_drifted = int(res.get("number_of_drifted_columns", 0))
        drift_share = float(res.get("share_of_drifted_columns", n_drifted / max(n_cols, 1)))
        return n_cols, n_drifted, drift_share

    return total_cols, 0, 0.0


def run_drift_monitoring(sample_size: int = 5000):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    print("Loading model artifacts and lightweight dataset slices (~30 MB RAM)...")

    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    with open(CAT_MAPPING_PATH, "r") as f:
        cat_mappings = json.load(f)

    feature_names = metadata["feature_names"]
    cat_features = metadata["categorical_features"]

    # Stream 5,000 validation rows (80% offset) & 5,000 test stream rows without loading full files
    ref_df = read_parquet_slice(TRAIN_PARQUET_PATH, target_rows=sample_size, skip_ratio=0.80)
    curr_df = read_parquet_slice(TEST_PARQUET_PATH, target_rows=sample_size, skip_ratio=0.0)

    print(f"Scoring {len(ref_df):,} reference rows and {len(curr_df):,} current stream rows...")
    X_ref = prepare_for_inference(ref_df, feature_names, cat_features, cat_mappings)
    ref_probs = booster.predict(xgb.DMatrix(X_ref, enable_categorical=True))
    del X_ref
    gc.collect()

    X_curr = prepare_for_inference(curr_df, feature_names, cat_features, cat_mappings)
    curr_probs = booster.predict(xgb.DMatrix(X_curr, enable_categorical=True))
    del X_curr
    gc.collect()

    # Keep only monitored columns + prediction score to minimize memory footprint
    cols_to_eval = [c for c in MONITORED_FEATURES if c in ref_df.columns and c in curr_df.columns]
    ref_monitor = ref_df[cols_to_eval].copy()
    curr_monitor = curr_df[cols_to_eval].copy()

    del ref_df, curr_df
    gc.collect()

    ref_monitor["fraud_probability"] = ref_probs
    curr_monitor["fraud_probability"] = curr_probs

    for col in cols_to_eval:
        if col in cat_features:
            ref_monitor[col] = ref_monitor[col].astype(str)
            curr_monitor[col] = curr_monitor[col].astype(str)

    # Run Evidently Data & Prediction Drift Report (Task T6.1)
    print("Generating Evidently AI Data & Prediction Drift Report...")
    drift_report = Report(metrics=[DataDriftPreset()])
    snapshot = drift_report.run(reference_data=ref_monitor, current_data=curr_monitor)

    target_report = snapshot if snapshot is not None else drift_report
    html_path = os.path.join(REPORTS_DIR, "evidently_drift_report.html")
    json_path = os.path.join(REPORTS_DIR, "evidently_drift_report.json")
    target_report.save_html(html_path)
    target_report.save_json(json_path)

    # Parse drift summary & evaluate Automated Retraining Trigger (Task T6.2)
    with open(json_path, "r") as f:
        report_dict = json.load(f)

    total_monitored = len(cols_to_eval) + 1
    n_cols, n_drifted, drift_share = parse_evidently_drift_summary(report_dict, total_monitored)
    pred_psi = calculate_psi(ref_probs, curr_probs)

    retrain_needed = bool(drift_share >= DRIFT_SHARE_THRESHOLD or pred_psi >= PSI_THRESHOLD)
    trigger_status = {
        "timestamp": datetime.now().isoformat(),
        "reference_samples": len(ref_monitor),
        "current_samples": len(curr_monitor),
        "monitored_columns": n_cols,
        "drifted_columns_count": n_drifted,
        "drift_share": round(float(drift_share), 4),
        "prediction_psi": round(pred_psi, 4),
        "thresholds": {
            "max_drift_share": DRIFT_SHARE_THRESHOLD,
            "max_prediction_psi": PSI_THRESHOLD,
        },
        "retrain_triggered": retrain_needed,
        "pipeline_action": "TRIGGER_RETRAIN_WEBHOOK" if retrain_needed else "NO_ACTION_MODEL_STABLE",
    }

    trigger_path = os.path.join(REPORTS_DIR, "retrain_trigger_status.json")
    with open(trigger_path, "w") as f:
        json.dump(trigger_status, f, indent=2)

    print("\n=== EVIDENTLY AI DRIFT & RETRAIN TRIGGER SUMMARY ===")
    print(f"Drifted Columns     : {n_drifted} / {n_cols} ({drift_share * 100:.2f}%)")
    print(f"Prediction Drift PSI: {pred_psi:.4f} (Threshold: {PSI_THRESHOLD})")
    print(f"Retrain Triggered   : {retrain_needed} -> [{trigger_status['pipeline_action']}]")
    print(f"HTML Report Saved   : {html_path}")
    print(f"Trigger Log Saved   : {trigger_path}")


if __name__ == "__main__":
    run_drift_monitoring(sample_size=5000)