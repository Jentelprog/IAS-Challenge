"""
Run DT-IDS model in real time using the trained RandomForest, with optional
cyber-attack injection and basic security monitoring.

- Loads artifacts from ../artifacts:
    rf_model.pkl, scaler.pkl, feature_cols.pkl
- Uses simulation_stream(...) as data source
- For each sample:
    * optionally inject an attack on the sensor / command data
    * run security checks (bounds, inconsistency, replay, cmd-injection)
    * run the RandomForest DT-IDS classifier
    * compute predicted class, confidence, anomaly flag/score
- Appends everything to data/realtime_stream.csv

Run from water_tank_simulation/src:

    python run_realtime_inference.py
"""

import os
import sys
import time
import copy
import joblib
import pandas as pd

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)  # .../water_tank_simulation
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "src", "artifacts")
SECURITY_DIR = os.path.join(BASE_DIR, "security")

os.makedirs(DATA_DIR, exist_ok=True)

# Paths to your trained artifacts
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "rf_model.pkl")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "feature_cols.pkl")

STREAM_CSV = os.path.join(DATA_DIR, "realtime_stream.csv")

# ---------------------------------------------------------------------
# Imports from other modules
# ---------------------------------------------------------------------

# Make sure we can import simulation_stream
sys.path.append(BASE_DIR)
from realtime_simulation import simulation_stream  # noqa: E402

# Security helpers (anti-replay, integrity, physical checks, logging)
sys.path.append(SECURITY_DIR)
from anti_replay import ReplayDetector  # noqa: E402
from security_monitor import SecurityMonitor  # noqa: E402
from integrity import compute_hash  # noqa: E402
from logger import log as sec_log  # noqa: E402

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

# Same mapping as in your AI-anomaly-detector
CLASS_NAMES = {
    0: "normal",
    5: "fault_both",
    6: "fault_clogged",
    7: "fault_filling",
}


# ---------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------
def load_artifacts():
    """Load scaler, model and feature column list from artifacts/."""
    if not (
        os.path.exists(SCALER_PATH)
        and os.path.exists(MODEL_PATH)
        and os.path.exists(FEATURES_PATH)
    ):
        raise FileNotFoundError(
            "Artifacts missing: expected scaler.pkl, rf_model.pkl, "
            "feature_cols.pkl under 'src/artifacts/'"
        )

    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    return scaler, model, feature_cols


# ---------------------------------------------------------------------
# Attack injection helpers
# ---------------------------------------------------------------------
def inject_attack_into_row(
    row,
    attack_type: str | None,
    t: float,
    attack_start: float,
    attack_end: float,
    replay_template: dict | None,
):
    """
    Modify the current 'row' in place to simulate a cyber attack.

    attack_type:
        None             -> no attack
        "spoofing"       -> fake sensor readings (constant bogus values)
        "replay"         -> keep replaying an earlier benign sample
        "cmd"            -> malicious pump / valve commands pattern

    attack window: attack_start <= t <= attack_end

    Returns:
        row, replay_template, is_under_attack (0/1), attack_name (str)
    """
    # Default metadata
    is_under_attack = 0
    attack_name = "none"

    if attack_type is None:
        return row, replay_template, is_under_attack, attack_name

    # If we're before the attack window, optionally prepare replay template
    if t < attack_start:
        if attack_type == "replay":
            # keep updating template with *clean* traffic
            replay_template = copy.deepcopy(row)
        return row, replay_template, is_under_attack, attack_name

    # After attack window -> no more attacks
    if t > attack_end:
        return row, replay_template, is_under_attack, attack_name

    # Inside the attack window
    is_under_attack = 1
    attack_name = attack_type

    # ------------------------------------------------------------------
    # SPOOFING ATTACK:
    #      Send "nice-looking" but WRONG sensor values: constant level,
    #      no flow, etc.
    # ------------------------------------------------------------------
    if attack_type == "spoofing":
        # Example: claim tank is perfectly stable at mid-level
        spoof_level = 1.0  # meters
        row["level_real"] = spoof_level
        row["flow_in_real"] = 0.0
        row["flow_out_real"] = 0.0
        # Pressure & current spoofed as well (optional)
        if "pressure_real" in row:
            row["pressure_real"] = 9.8  # around 1m water column
        if "pump_current" in row:
            row["pump_current"] = 0.5  # arbitrary “normal-looking” value

    # ------------------------------------------------------------------
    # REPLAY ATTACK:
    #      Re-send exactly the same sensor tuple as an old benign sample.
    # ------------------------------------------------------------------
    elif attack_type == "replay":
        if replay_template is not None:
            for key in [
                "level_real",
                "flow_in_real",
                "flow_out_real",
                "pressure_real",
                "pump_current",
            ]:
                if key in replay_template:
                    row[key] = replay_template[key]

    # ------------------------------------------------------------------
    # COMMAND INJECTION ATTACK:
    #      Force pump ON with valve closed -> dangerous combination.
    # ------------------------------------------------------------------
    elif attack_type in ("cmd", "cmd_injection", "command"):
        # Force pump on and valve nearly closed
        row["pump_state"] = 1
        # valve_position is in %, model feature; set close to 0
        row["valve_position"] = 1.0

    # Any other string -> no modification, but still flagged as attack
    return row, replay_template, is_under_attack, attack_name


# ---------------------------------------------------------------------
# Main realtime loop
# ---------------------------------------------------------------------
def run_realtime_inference(
    duration: float = 300.0,
    dt: float = 0.1,
    with_faults: bool = True,
    speed_multiplier: float = 1.0,
    attack_type: str | None = None,
    attack_start: float = 60.0,
    attack_end: float = 120.0,
):
    """
    Run the realtime DT-IDS and optionally simulate a cyber attack.

    attack_type:
        None / "spoofing" / "replay" / "cmd"
    attack_start / attack_end:
        time window in seconds where the attack is active
    """
    scaler, model, feature_cols = load_artifacts()

    monitor = SecurityMonitor(level_min=0.0, level_max=2.0)
    replay_detector = ReplayDetector(window=5, round_digits=3)

    print("=== REAL-TIME DT-IDS (RandomForest) STARTED ===")
    print(f"Artifacts loaded from: {ARTIFACTS_DIR}")
    print(f"Writing stream to: {STREAM_CSV}")
    if attack_type is not None:
        print(
            f"Attack simulation ENABLED: {attack_type} "
            f"(t in [{attack_start:.1f}s, {attack_end:.1f}s])"
        )
    else:
        print("Attack simulation DISABLED (normal cyber conditions).")
    print("----------------------------------------------\n")

    # Remove old file if exists
    if os.path.exists(STREAM_CSV):
        os.remove(STREAM_CSV)

    t0 = time.time()
    header_written = False
    replay_template = None  # for replay attack payload

    for row in simulation_stream(
        duration=duration,
        dt=dt,
        with_faults=with_faults,
        speed_multiplier=speed_multiplier,
    ):
        t = float(row.get("timestamp", 0.0))

        # --------------------------------------------------------------
        # 1) Inject cyber attack (modifies 'row' in place)
        # --------------------------------------------------------------
        row, replay_template, is_under_attack, attack_name = inject_attack_into_row(
            row=row,
            attack_type=attack_type,
            t=t,
            attack_start=attack_start,
            attack_end=attack_end,
            replay_template=replay_template,
        )
        row["is_under_attack"] = int(is_under_attack)
        row["attack_type"] = attack_name

        # --------------------------------------------------------------
        # 2) Security monitoring & integrity
        # --------------------------------------------------------------
        level = float(row.get("level_real", 0.0))
        flow_in = float(row.get("flow_in_real", 0.0))
        flow_out = float(row.get("flow_out_real", 0.0))
        pressure = float(row.get("pressure_real", 0.0))
        pump_current = float(row.get("pump_current", 0.0))

        sec_alerts: list[str] = []

        # Physical bounds / sanity
        msg = monitor.detect_physical_bounds(level, flow_in, flow_out)
        if msg:
            sec_alerts.append(msg)

        # Simple physical consistency
        msg = monitor.detect_inconsistency(level, flow_in, flow_out)
        if msg:
            sec_alerts.append(msg)

        # Command injection pattern (pump high + valve closed)
        pump_cmd = float(row.get("pump_state", 0.0))  # 0 or 1
        valve_cmd = float(row.get("valve_position", 0.0)) / 100.0  # convert % -> 0-1
        msg = monitor.detect_command_injection(pump_cmd, valve_cmd)
        if msg:
            sec_alerts.append(msg)

        # Replay detection (based on sensor tuple)
        is_replay_suspected = replay_detector.check(
            (level, flow_in, flow_out, pressure, pump_current)
        )
        if is_replay_suspected:
            sec_alerts.append("Replay pattern detected (repeated samples)")

        # Add security metadata
        sec_alert_msg = " | ".join(sec_alerts)
        row["sec_alert"] = 1 if sec_alerts else 0
        row["sec_alert_message"] = sec_alert_msg
        row["is_replay_suspected"] = int(is_replay_suspected)

        if sec_alerts:
            sec_log(f"[t={t:.2f}s] {sec_alert_msg}", level="WARN")

        # Integrity hash over key process values (after attack injection)
        integrity_payload = {
            "timestamp": t,
            "level_real": level,
            "flow_in_real": flow_in,
            "flow_out_real": flow_out,
            "pressure_real": pressure,
            "pump_current": pump_current,
            "attack_type": attack_name,
            "is_under_attack": int(is_under_attack),
        }
        row["integrity_hash"] = compute_hash(integrity_payload)

        # --------------------------------------------------------------
        # 3) Build feature vector and run RandomForest DT-IDS
        # --------------------------------------------------------------
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

        pred = int(model.predict(X_scaled)[0])  # 0,5,6,7
        proba = model.predict_proba(X_scaled)[0]
        confidence = float(proba.max())  # best class prob

        # anomaly_flag: anything different from 0 = anomaly
        anomaly_flag = 0 if pred == 0 else 1

        # anomaly_score: confidence of "being faulty"
        anomaly_score = confidence if anomaly_flag == 1 else (1.0 - confidence)

        row["predicted_class"] = pred
        row["class_name"] = CLASS_NAMES.get(pred, str(pred))
        row["confidence"] = confidence
        row["anomaly_flag"] = anomaly_flag
        row["anomaly_score"] = anomaly_score

        # --------------------------------------------------------------
        # 4) Append to CSV for dashboard / plotting
        # --------------------------------------------------------------
        df_row = pd.DataFrame([row])
        df_row.to_csv(
            STREAM_CSV,
            mode="a",
            index=False,
            header=not header_written,
        )
        header_written = True

        # --------------------------------------------------------------
        # 5) Console log
        # --------------------------------------------------------------
        status_str = (
            f"ANOMALY ({row['class_name']})" if anomaly_flag == 1 else "NORMAL"
        )
        attack_str = (
            f"ATTACK={attack_name}" if is_under_attack else "ATTACK=none"
        )
        sec_str = "SEC-OK"
        if sec_alerts:
            sec_str = f"SEC-ALERT ({len(sec_alerts)})"

        print(
            f"[t={t:6.2f}s] "
            f"level={row['level_real']:.3f} m  "
            f"flow_in={row['flow_in_real']:.3f}  "
            f"label={row['label']}  "
            f"pred={status_str:<20} "
            f"conf={confidence:.3f}  "
            f"{attack_str:<16} "
            f"{sec_str}"
        )

    dt_wall = time.time() - t0
    print("\n=== REAL-TIME DT-IDS FINISHED ===")
    print(f"Simulated {duration:.1f}s in {dt_wall:.1f}s (wall time).")
    print(f"CSV written to: {STREAM_CSV}")


# ---------------------------------------------------------------------
# Example CLI run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Example: 5 minutes, real-time, with a spoofing attack from 60s to 120s
    run_realtime_inference(
        duration=300.0,
        dt=0.1,
        with_faults=True,
        speed_multiplier=1.0,
        attack_type="spoofing",   # None / "spoofing" / "replay" / "cmd"
        attack_start=60.0,
        attack_end=120.0,
    )
