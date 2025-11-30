import os
import numpy as np
import pandas as pd

from tank_model import WaterTank
from controller import OnOffController
from sensor_model import SensorSuite
from config.parameters import *  # LABEL_SPOOFING, SIMULATION_DT, etc.

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "anomalous_spoofing.csv"
)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


def spoof_value(true_val, mode="offset"):
    """Return a spoofed measurement."""
    if mode == "offset":
        return true_val + np.random.uniform(0.5, 1.5)
    else:
        return np.random.uniform(TANK_HEIGHT_MIN, TANK_HEIGHT_MAX)


def run_spoofing(duration=120.0, dt=SIMULATION_DT,
                 spoof_mode="offset", spoof_start=20.0, spoof_end=80.0):

    tank = WaterTank(initial_level=INITIAL_WATER_LEVEL, area=TANK_AREA)
    controller = OnOffController()
    sensors = SensorSuite(seed=42)

    data = []
    t = 0.0
    steps = int(duration / dt)

    for step in range(steps):

        # Controller decides
        pump_cmd, pump_state = controller.update(tank.h)

        # Update physics
        true_level, q_in, q_out = tank.update(
            pump_cmd, VALVE_POSITION_DEFAULT, dt
        )
        true_pressure = tank.calculate_pressure()
        true_current = sensors.measure_current(pump_cmd, pump_state)

        # Apply spoofing
        if spoof_start <= t <= spoof_end:
            level_meas = spoof_value(true_level, mode=spoof_mode)
            flow_in_meas = spoof_value(q_in, mode=spoof_mode)
            flow_out_meas = spoof_value(q_out, mode=spoof_mode)
            pressure_meas = spoof_value(true_pressure, mode=spoof_mode)
            current_meas = spoof_value(true_current, mode=spoof_mode)
        else:
            level_meas = sensors.measure_level(true_level)
            flow_in_meas = sensors.measure_flow(q_in)
            flow_out_meas = sensors.measure_flow(q_out)
            pressure_meas = sensors.measure_pressure(true_pressure)
            current_meas = sensors.measure_current(pump_cmd, pump_state)

        # CSV row (correct fields)
        row = {
            "timestamp": t,
            "scenario_id": "attack_spoofing_001",
            "label": LABEL_SPOOFING,

            # REAL values (ground truth)
            "level_real": true_level,
            "flow_in_real": q_in,
            "flow_out_real": q_out,
            "pressure_real": true_pressure,
            "pump_current_real": true_current,

            # MEASURED (possibly spoofed) values
            "level_meas": level_meas,
            "flow_in_meas": flow_in_meas,
            "flow_out_meas": flow_out_meas,
            "pressure_meas": pressure_meas,
            "pump_current_meas": current_meas,

            # Actuators
            "valve_position": VALVE_POSITION_DEFAULT,
            "valve_commanded": VALVE_POSITION_DEFAULT,
            "pump_state": pump_state,
            "controller_setpoint": controller.setpoint,
        }

        data.append(row)
        t += dt

    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Spoofing dataset saved to: {OUTPUT_PATH} (rows: {len(df)})")


if __name__ == "__main__":
    run_spoofing(
        duration=120.0,
        dt=SIMULATION_DT,
        spoof_mode="offset",
        spoof_start=10.0,
        spoof_end=70.0
    )
