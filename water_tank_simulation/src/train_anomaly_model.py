"""
Train anomaly detection model (IsolationForest) for DT-IDS.

- Trains only on NORMAL data (normal_operation.csv)
- Evaluates on RANDOM FAULTS (random_faults.csv) if available
- Saves a sklearn Pipeline (StandardScaler + IsolationForest) as models/model.pkl

Requires:
    pip install scikit-learn pandas numpy joblib
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)  # .../water_tank_simulation
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# If you ever need to import your tank/controller here:
sys.path.append(BASE_DIR)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

# These features must match what you will use at runtime
FEATURE_COLUMNS = [
    "level_real",
    "flow_in_real",
    "flow_out_real",
    "pressure_real",
    "pump_current",
    "valve_position",
    "pump_state",
]

NORMAL_FILE = os.path.join(DATA_DIR, "normal_operation.csv")
FAULT_FILE = os.path.join(DATA_DIR, "random_faults.csv")


def load_normal_data() -> pd.DataFrame:
    if not os.path.exists(NORMAL_FILE):
        raise FileNotFoundError(
            f"Normal data not found at {NORMAL_FILE}. " f"Run generate_normal.py first."
        )

    df = pd.read_csv(NORMAL_FILE)
    # Some security columns may exist; we don't need them as features for IF.
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in normal data: {missing}")

    return df


def load_fault_data() -> pd.DataFrame | None:
    if not os.path.exists(FAULT_FILE):
        print("[INFO] random_faults.csv not found – skipping evaluation.")
        return None

    df = pd.read_csv(FAULT_FILE)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in fault data: {missing}")

    return df


def train_model():
    print("=== TRAINING ANOMALY DETECTION MODEL (IsolationForest) ===")

    df_normal = load_normal_data()
    X_normal = df_normal[FEATURE_COLUMNS].values

    # IsolationForest: train on normal only
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                IsolationForest(
                    n_estimators=200,
                    contamination=0.05,  # expected anomaly ratio in future data
                    random_state=42,
                    bootstrap=True,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print(f"[INFO] Training on {len(df_normal)} normal samples ...")
    pipeline.fit(X_normal)
    print("[OK] Model trained.")

    # -----------------------------------------------------------------
    # Optional evaluation on random_faults dataset
    # -----------------------------------------------------------------
    df_fault = load_fault_data()
    if df_fault is not None:
        df_eval = df_fault.copy()
        X_eval = df_eval[FEATURE_COLUMNS].values

        # Ground-truth: 0 = normal, 1 = anomaly/fault/hack
        y_true = (df_eval["label"] != 0).astype(int)

        # IsolationForest: predict → 1 (normal), -1 (anomaly)
        y_pred_if = pipeline.predict(X_eval)
        y_pred = (y_pred_if == -1).astype(int)

        print("\n=== EVALUATION ON random_faults.csv ===")
        print("Confusion matrix [rows=true, cols=pred]:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, digits=3))

    # -----------------------------------------------------------------
    # Save model
    # -----------------------------------------------------------------
    joblib.dump({"pipeline": pipeline, "features": FEATURE_COLUMNS}, MODEL_PATH)
    print(f"\n[OK] Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
