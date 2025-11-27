"""
Generate Normal Operation Dataset
Week 1 Deliverable
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tank_model import WaterTank
from controller import OnOffController
from sensor_model import SensorSuite
from config.parameters import *

# --- Security Imports ---
from security.authentication import check_key, SECRET_KEY
from security.integrity import compute_hash
from security.anti_replay import ReplayDetector
from security.security_monitor import SecurityMonitor
from security.logger import log


def generate_normal_data(duration=600, dt=0.1):
    """Generate normal operation dataset"""

    print(f" Generating normal operation data...")
    print(f" Duration: {duration}s ({duration/60:.1f} minutes)")

    # Initialize system
    tank = WaterTank(initial_level=50, area=50)
    controller = OnOffController()
    sensors = SensorSuite(seed=42)
    valve_opening = VALVE_POSITION_DEFAULT

    # Data storage
    data = []
    t = 0.0
    steps = int(duration / dt)

    # --- Initialize Security Components ---
    replay_detector = ReplayDetector(window=5, round_digits=3)
    monitor = SecurityMonitor(level_min=0.0, level_max=TANK_HEIGHT_MAX)

    # Simulation loop
    for step in range(steps):
        # Controller decides
        pump_command, pump_state = controller.update(tank.h)

        # Update physics
        true_level, q_in, q_out = tank.update(pump_command, valve_opening, dt)
        true_pressure = tank.calculate_pressure()

        # Measure with sensors
        level_measured = sensors.measure_level(true_level)
        flow_in_measured = sensors.measure_flow(q_in)
        flow_out_measured = sensors.measure_flow(q_out)
        pressure_measured = sensors.measure_pressure(true_pressure)
        current_measured = sensors.measure_current(pump_command, pump_state)

        # -------------------------------
        #    SECURITY LAYERS
        # -------------------------------
        sensor_payload = {
            "level": float(level_measured),
            "flow_in": float(flow_in_measured),
            "flow_out": float(flow_out_measured),
            "pressure": float(pressure_measured),
            "pump_current": float(current_measured),
        }

        # Authentication
        provided_key = SECRET_KEY
        if not check_key(provided_key):
            log("Authentication failed: invalid source key", level="WARN")
            auth_ok = 0
        else:
            auth_ok = 1

        # Integrity hash
        row_hash = compute_hash(sensor_payload)

        # Replay detection
        sample_tuple = (
            sensor_payload["level"],
            sensor_payload["flow_in"],
            sensor_payload["flow_out"],
            sensor_payload["pressure"],
            sensor_payload["pump_current"],
        )
        is_replay = replay_detector.check(sample_tuple)
        if is_replay:
            log(f"Replay detected at t={t:.2f}s: {sample_tuple}", level="WARN")

        # Physical bounds check
        bounds_issue = monitor.detect_physical_bounds(
            sensor_payload["level"],
            sensor_payload["flow_in"],
            sensor_payload["flow_out"],
        )
        if bounds_issue:
            log(f"Physical bounds issue at t={t:.2f}s: {bounds_issue}", level="WARN")

        # Physical inconsistency check
        inconsistency_issue = monitor.detect_inconsistency(
            sensor_payload["level"],
            sensor_payload["flow_in"],
            sensor_payload["flow_out"],
        )
        if inconsistency_issue:
            log(
                f"Physical inconsistency at t={t:.2f}s: {inconsistency_issue}",
                level="WARN",
            )

        # -------------------------------
        # Store Data Row
        # -------------------------------
        row = {
            "timestamp": t,
            "scenario_id": "normal_001",
            "label": 0,
            "level_real": level_measured,
            "flow_in_real": flow_in_measured,
            "flow_out_real": flow_out_measured,
            "pressure_real": pressure_measured,
            "pump_current": current_measured,
            "valve_position": valve_opening * 100,
            "pump_state": pump_state,
            "controller_setpoint": controller.setpoint,
            # --- Security Metadata ---
            "auth_ok": auth_ok,
            "hash": row_hash,
            "replay_flag": int(is_replay),
            "bounds_issue": bounds_issue or "",
            "inconsistency_issue": inconsistency_issue or "",
        }

        data.append(row)

        # Progress
        if step % 1000 == 0:
            print(f"   Progress: {100*step/steps:.0f}%")

        t += dt

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    output_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "normal_operation.csv"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f" Saved to: {output_path}")
    print(f" Rows: {len(df)}")

    # Create plot
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(df["timestamp"], df["level_real"], linewidth=0.8)
    plt.axhline(y=1.0, color="r", linestyle="--", label="Setpoint")
    plt.ylabel("Water Level [m]")
    plt.title("Normal Operation - Water Level")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(df["timestamp"], df["flow_in_real"], label="Flow In", linewidth=0.8)
    plt.plot(df["timestamp"], df["flow_out_real"], label="Flow Out", linewidth=0.8)
    plt.ylabel("Flow [m³/s]")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = os.path.join(
        os.path.dirname(__file__), "..", "plots", "normal_operation.png"
    )
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f" Plot saved to: {plot_path}")
    plt.show()

    return df


if __name__ == "__main__":
    # Generate data
    df = generate_normal_data(duration=600)

    print("\n Data Summary:")
    print(df.describe())
