"""
Run DT-IDS model in real time using the trained RandomForest.

- Loads artifacts from ../artifacts:
    rf_model.pkl, scaler.pkl, feature_cols.pkl
- Uses simulation_stream(...) as data source
- For each sample: compute predicted class, class name, confidence,
  and an anomaly_flag (0 = normal, 1 = any fault)
- Appends all data to data/realtime_stream.csv

Run from water_tank_simulation/src:

    python run_realtime_inference.py
"""

import os
import sys
import time
import joblib
import pandas as pd

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)  # .../water_tank_simulation
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "src/artifacts")

os.makedirs(DATA_DIR, exist_ok=True)

# Paths to your trained artifacts
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "rf_model.pkl")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "feature_cols.pkl")

STREAM_CSV = os.path.join(DATA_DIR, "realtime_stream.csv")

# Make sure we can import simulation_stream
sys.path.append(BASE_DIR)
from realtime_simulation import simulation_stream  # noqa: E402


# Same mapping as in your AI-anomaly-detector
CLASS_NAMES = {
    0: "normal",
    5: "fault_both",
    6: "fault_clogged",
    7: "fault_filling",
}


def load_artifacts():
    """Load scaler, model and feature column list from artifacts/."""
    if not (
        os.path.exists(SCALER_PATH)
        and os.path.exists(MODEL_PATH)
        and os.path.exists(FEATURES_PATH)
    ):
        raise FileNotFoundError(
            "Artifacts missing: expected scaler.pkl, rf_model.pkl, "
            "feature_cols.pkl under 'artifacts/'"
        )

    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    return scaler, model, feature_cols


def run_realtime_inference(
    duration: float = 300.0,
    dt: float = 0.1,
    with_faults: bool = True,
    speed_multiplier: float = 1.0,
):
    scaler, model, feature_cols = load_artifacts()

    print("=== REAL-TIME DT-IDS (RandomForest) STARTED ===")
    print(f"Artifacts loaded from: {ARTIFACTS_DIR}")
    print(f"Writing stream to: {STREAM_CSV}")
    print("----------------------------------------------\n")

    # Remove old file if exists
    if os.path.exists(STREAM_CSV):
        os.remove(STREAM_CSV)

    t0 = time.time()
    header_written = False

    for row in simulation_stream(
        duration=duration,
        dt=dt,
        with_faults=with_faults,
        speed_multiplier=speed_multiplier,
    ):
        # ----------------------------------------
        # Build feature vector from current row
        # ----------------------------------------
        # Make sure all needed keys exist in the row
        features = []
        for f in feature_cols:
            if f not in row:
                # if missing, treat as 0 – should not happen if simulation matches training
                value = 0.0
            else:
                value = row[f]
            features.append(float(value))

        X = pd.DataFrame([features], columns=feature_cols)
        X_scaled = scaler.transform(X)

        # ----------------------------------------
        # Predict with RandomForest
        # ----------------------------------------
        pred = int(model.predict(X_scaled)[0])  # 0,5,6,7
        proba = model.predict_proba(X_scaled)[0]
        confidence = float(proba.max())  # best class prob

        # anomaly_flag: anything different from 0 = anomaly
        anomaly_flag = 0 if pred == 0 else 1

        # anomaly_score: you can define it as confidence of "being faulty"
        # For simplicity: if it's a fault, score = confidence, else score = 1 - confidence
        anomaly_score = confidence if anomaly_flag == 1 else (1.0 - confidence)

        row["predicted_class"] = pred
        row["class_name"] = CLASS_NAMES.get(pred, str(pred))
        row["confidence"] = confidence
        row["anomaly_flag"] = anomaly_flag
        row["anomaly_score"] = anomaly_score

        # ----------------------------------------
        # Append to CSV for dashboard
        # ----------------------------------------
        df_row = pd.DataFrame([row])
        df_row.to_csv(
            STREAM_CSV,
            mode="a",
            index=False,
            header=not header_written,
        )
        header_written = True

        # Console log
        status_str = f"ANOMALY ({row['class_name']})" if anomaly_flag == 1 else "NORMAL"
        print(
            f"[t={row['timestamp']:6.2f}s] "
            f"level={row['level_real']:.3f} m  "
            f"flow_in={row['flow_in_real']:.3f}  "
            f"label={row['label']}  "
            f"pred={status_str:<20} "
            f"conf={confidence:.3f}"
        )

    dt_wall = time.time() - t0
    print("\n=== REAL-TIME DT-IDS FINISHED ===")
    print(f"Simulated {duration:.1f}s in {dt_wall:.1f}s (wall time).")


if __name__ == "__main__":
    # Example run: 5 minutes at real-time
    run_realtime_inference(
        duration=300.0,
        dt=0.1,
        with_faults=True,
        speed_multiplier=1.0,
    )
