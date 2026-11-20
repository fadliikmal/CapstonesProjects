import argparse
import os
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_PARQUET_PATH = os.path.join(BASE_DIR, "data", "test_engineered.parquet")
API_URL = "http://127.0.0.1:8000/predict"


def load_stream_slice(path: str, num_events: int = 100, offset: int = 0) -> pd.DataFrame:
    """Read only the requested number of rows via PyArrow iter_batches (~2 MB RAM)."""
    pf = pq.ParquetFile(path)
    collected = []
    rows_seen = 0
    rows_needed = num_events

    for batch in pf.iter_batches(batch_size=max(num_events, 500)):
        batch_len = len(batch)
        if rows_seen + batch_len <= offset:
            rows_seen += batch_len
            continue

        start_in_batch = max(0, offset - rows_seen)
        slice_batch = batch.slice(start_in_batch, rows_needed)
        collected.append(slice_batch.to_pandas())
        rows_needed -= len(slice_batch)
        rows_seen += batch_len

        if rows_needed <= 0:
            break

    if not collected:
        first_batch = next(pf.iter_batches(batch_size=num_events))
        return first_batch.to_pandas()

    return pd.concat(collected, ignore_index=True)


def clean_record_for_json(row_dict: dict) -> dict:
    """Convert pandas/numpy NaN and scalar types into standard Python JSON-safe values."""
    cleaned = {}
    for k, v in row_dict.items():
        if pd.isna(v):
            cleaned[k] = None
        elif isinstance(v, (np.integer,)):
            cleaned[k] = int(v)
        elif isinstance(v, (np.floating,)):
            cleaned[k] = float(v)
        elif isinstance(v, (np.bool_,)):
            cleaned[k] = bool(v)
        else:
            cleaned[k] = v
    return cleaned


def run_http_replay(num_events: int = 100, delay_sec: float = 0.15, offset: int = 0):
    if not os.path.exists(TEST_PARQUET_PATH):
        raise FileNotFoundError(f"Test dataset not found at: {TEST_PARQUET_PATH}")

    print(f"Streaming {num_events} events (offset={offset}, ~2 MB RAM) from {TEST_PARQUET_PATH}...")
    stream_df = load_stream_slice(TEST_PARQUET_PATH, num_events=num_events, offset=offset)
    print(f"Starting real-time transaction replay ({len(stream_df)} events -> {API_URL})...\n")

    latencies = []
    action_counts = {"APPROVE": 0, "CHALLENGE_REVIEW": 0, "AUTO_BLOCK": 0}

    for _, row in stream_df.iterrows():
        tx_id = int(row["TransactionID"])
        tx_amt = float(row["TransactionAmt"])
        features_dict = clean_record_for_json(row.to_dict())

        payload = {
            "TransactionID": tx_id,
            "TransactionAmt": tx_amt,
            "features": features_dict,
        }

        try:
            resp = requests.post(API_URL, json=payload, timeout=5)
            resp.raise_for_status()
            data = resp.json()

            action = data["action"]
            prob = data["fraud_probability"]
            lat = data["latency_ms"]

            latencies.append(lat)
            action_counts[action] = action_counts.get(action, 0) + 1

            tag = "[BLOCK]" if action == "AUTO_BLOCK" else ("[REVIEW]" if action == "CHALLENGE_REVIEW" else "[OK]   ")
            print(
                f"{tag} TxID: {tx_id} | Amt: ${tx_amt:8.2f} | "
                f"Prob: {prob:.4f} | Action: {action:<16} | Latency: {lat:5.2f} ms"
            )
        except requests.exceptions.RequestException as err:
            print(f"[ERROR] Failed sending TxID {tx_id}: {err}")

        time.sleep(delay_sec)

    if latencies:
        print("\n=== STREAMING REPLAY SUMMARY ===")
        print(f"Total Processed   : {len(latencies)} transactions")
        print(f"Average Latency   : {np.mean(latencies):.2f} ms (P95: {np.percentile(latencies, 95):.2f} ms)")
        print(f"Decision Breakdown: {action_counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Transaction Stream Producer")
    parser.add_argument("--events", type=int, default=100, help="Number of transactions to replay")
    parser.add_argument("--delay", type=float, default=0.15, help="Delay in seconds between events")
    parser.add_argument("--offset", type=int, default=0, help="Row offset to start replay from")
    args = parser.parse_args()

    run_http_replay(num_events=args.events, delay_sec=args.delay, offset=args.offset)