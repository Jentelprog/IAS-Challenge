import os
import numpy as np
import pandas as pd

from tank_model import WaterTank
from controller import OnOffController
from sensor_model import SensorSuite
from config.parameters import *  # provides SIMULATION_DT, INITIAL_WATER_LEVEL, TANK_AREA, VALVE_POSITION_DEFAULT, etc.

# Output path (one folder up from this script)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "anomalous_cmd_injection.csv")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


def run_cmd_injection(duration=120.0, dt=SIMULATION_DT, attack_window=(30.0, 60.0)):
    """
    Runs a simulation where actuator commands are overridden during attack_window.
    Produces a CSV with real and measured values; marks rows during attack with a label.
    """

    # Safe fallbacks if some constants are missing from config.parameters
    pump_full_power = globals().get("PUMP_POWER_ON", 1.0)           # default: full power = 1.0
    label_valve_stuck = globals().get("LABEL_VALVE_STUCK", 3)      # default attack label (3)

    # Initialize models
    tank = WaterTank(initial_level=INITIAL_WATER_LEVEL, area=TANK_AREA)
    controller = OnOffController()
    sensors = SensorSuite(seed=42)

    rows = []
    steps = int(duration / dt)

    for step in range(steps):
        current_time = step * dt

        # Normal control
        pump_cmd, pump_state = controller.update(tank.h)
        valve_cmd = VALVE_POSITION_DEFAULT  # expected in range [0,1] (check your config)

        # Attack window: override commands
        if attack_window[0] <= current_time <= attack_window[1]:
            # Force pump to full power and close valve
            pump_cmd = pump_full_power
            valve_cmd = 0.0
            attack_active = 1
        else:
            attack_active = 0

        # Update physical model with (possibly) overridden commands
        # Expect tank.update to return (level, q_in, q_out) or similar
        true_level, q_in, q_out = tank.update(pump_cmd, valve_cmd, dt)

        # Compute true physical values
        true_pressure = tank.calculate_pressure()
        true_current = sensors.measure_current(pump_cmd, pump_state)  # physical current measured by sensor model

        # Sensor measurements (simulate real sensors)
        level_meas = sensors.measure_level(true_level)
        flow_in_meas = sensors.measure_flow(q_in)
        flow_out_meas = sensors.measure_flow(q_out)
        pressure_meas = sensors.measure_pressure(true_pressure)
        current_meas = sensors.measure_current(pump_cmd, pump_state)

        # Compose CSV row
        row = {
            "timestamp": round(current_time, 4),
            "scenario_id": "attack_cmd_injection_001",
            # label for rows inside attack window, else 0 (normal)
            "label": label_valve_stuck if attack_active else 0,

            # Ground truth (real) values
            "level_real": float(true_level),
            "flow_in_real": float(q_in),
            "flow_out_real": float(q_out),
            "pressure_real": float(true_pressure),
            "pump_current_real": float(true_current),

            # Measured (sensor) values
            "level_meas": float(level_meas),
            "flow_in_meas": float(flow_in_meas),
            "flow_out_meas": float(flow_out_meas),
            "pressure_meas": float(pressure_meas),
            "pump_current_meas": float(current_meas),

            # Actuator / controller info
            "valve_position": float(valve_cmd),
            "valve_commanded": float(VALVE_POSITION_DEFAULT),
            "pump_state": int(pump_state),
            "pump_command": float(pump_cmd),
            "controller_setpoint": getattr(controller, "setpoint", None)
        }

        rows.append(row)

    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Command injection dataset saved to: {OUTPUT_PATH} (rows: {len(df)})")


if __name__ == "__main__":
    # Example run
    run_cmd_injection(duration=120.0, dt=SIMULATION_DT, attack_window=(20.0, 50.0))
