"""
Run DT-IDS model in real time:

- Loads models/model.pkl (Pipeline with scaler + IsolationForest)
- Uses simulation_stream(...) as data source
- For each sample: compute anomaly_flag + anomaly_score
- Append all data to data/realtime_stream.csv

This is the "online IDS" process you run before opening the dashboard.
"""

import os
import sys
import time
import joblib
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)  # .../water_tank_simulation
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)

sys.path.append(BASE_DIR)

from realtime_simulation import simulation_stream  # noqa: E402

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
STREAM_CSV = os.path.join(DATA_DIR, "realtime_stream.csv")


def run_realtime_inference(
    duration: float = 300.0,
    dt: float = 0.1,
    with_faults: bool = True,
    speed_multiplier: float = 1.0,
):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train_anomaly_model.py first."
        )

    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    feature_cols = bundle["features"]

    print("=== REAL-TIME DT-IDS STARTED ===")
    print(f"Model: {MODEL_PATH}")
    print(f"Writing stream to: {STREAM_CSV}")
    print("--------------------------------\n")

    # Remove previous stream file (optional)
    if os.path.exists(STREAM_CSV):
        os.remove(STREAM_CSV)

    # Stream simulation + inference
    t0 = time.time()
    header_written = False

    for row in simulation_stream(
        duration=duration,
        dt=dt,
        with_faults=with_faults,
        speed_multiplier=speed_multiplier,
    ):
        # Build feature vector
        X = pd.DataFrame([[row[col] for col in feature_cols]], columns=feature_cols)

        # IsolationForest: predict + score
        pred_if = pipeline.predict(X)[0]  # 1=normal, -1=anomaly
        score = pipeline.decision_function(X)[0]

        anomaly_flag = 1 if pred_if == -1 else 0

        row["anomaly_flag"] = anomaly_flag
        row["anomaly_score"] = float(score)

        # Append to CSV for the dashboard
        df_row = pd.DataFrame([row])
        df_row.to_csv(
            STREAM_CSV,
            mode="a",
            index=False,
            header=not header_written,
        )
        header_written = True

        # Console log
        status = "ANOMALY" if anomaly_flag == 1 else "NORMAL "
        print(
            f"[t={row['timestamp']:6.2f}s] "
            f"level={row['level_real']:.3f} m  "
            f"flow_in={row['flow_in_real']:.3f}  "
            f"label={row['label']}  "
            f"pred={status}  score={score:.3f}"
        )

    dt_wall = time.time() - t0
    print("\n=== REAL-TIME DT-IDS FINISHED ===")
    print(f"Simulated {duration:.1f}s in {dt_wall:.1f}s wall time.")


if __name__ == "__main__":
    # Example: 5 minutes simulation at real time
    run_realtime_inference(
        duration=300.0, dt=0.1, with_faults=True, speed_multiplier=1.0
    )
