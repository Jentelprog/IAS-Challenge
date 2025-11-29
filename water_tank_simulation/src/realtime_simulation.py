"""
Real-time digital twin simulation stream for DT-IDS.

Provides a generator `simulation_stream(...)` that yields one sample (dict)
at a time, including:
    - Physical signals (level, flows, pressure, etc.)
    - Fault indicators (valve clogging, random filling)
    - Security metadata (auth_ok, hash, replay_flag, bounds_issue, inconsistency_issue)

You can use this generator in:
    - run_realtime_inference.py (to feed the ML model)
    - tests, or logging scripts.

Requires your existing modules:
    tank_model.py, controller.py, sensor_model.py, generate_random_faults.py,
    security/*, config/parameters.py
"""

import os
import sys
import time
from typing import Dict, Iterator, Optional

# ---------------------------------------------------------------------
# Path & imports (aligned with your existing files)
# ---------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)  # .../water_tank_simulation
sys.path.append(BASE_DIR)

from tank_model import WaterTank
from controller import OnOffController
from sensor_model import SensorSuite
from generate_random_faults import RandomFaultGenerator
from config.parameters import *  # noqa: F401, brings in constants

from security.authentication import check_key, SECRET_KEY
from security.integrity import compute_hash
from security.anti_replay import ReplayDetector
from security.security_monitor import SecurityMonitor
from security.logger import log


def simulation_stream(
    duration: float = 600.0,
    dt: float = 0.1,
    with_faults: bool = True,
    speed_multiplier: float = 1.0,
    seed: int = 42,
) -> Iterator[Dict]:
    """
    Yield simulation samples in (approximate) real time.

    Args:
        duration: total simulated time [s]
        dt: simulation time step [s]
        with_faults: if True, enable valve clogging + random filling
        speed_multiplier: >1 = faster than real-time, <1 = slower
        seed: random seed for sensors & faults

    Yields:
        row: dict with physical + security fields (similar to your CSV structure)
    """

    # ----------------- Initialize plant + controller + sensors -----------------
    tank = WaterTank(initial_level=INITIAL_WATER_LEVEL)
    controller = OnOffController()
    sensors = SensorSuite(seed=seed)

    # Fault generator
    fault_gen: Optional[RandomFaultGenerator] = (
        RandomFaultGenerator(seed=seed) if with_faults else None
    )

    # Base valve opening
    base_valve_opening = VALVE_POSITION_DEFAULT

    # Security components
    replay_detector = ReplayDetector(window=5, round_digits=3)
    monitor = SecurityMonitor(level_min=TANK_HEIGHT_MIN, level_max=TANK_HEIGHT_MAX)

    # Timing
    steps = int(duration / dt)
    t = 0.0

    # For random filling logic (same thresholds as your offline generator)
    filling_start_capacitie = TANK_HEIGHT_MAX * 0.2
    filling_end_capacitie = TANK_HEIGHT_MAX * 0.8

    for step in range(steps):
        # -------------- Faults --------------
        valve_clog_factor = 0.0
        random_filling_rate = 0.0

        if with_faults and fault_gen is not None:
            valve_clog_factor = fault_gen.check_valve_clogging(t, dt)
            random_filling_rate = fault_gen.check_random_filling(
                filling_start_capacitie,
                filling_end_capacitie,
                tank.h,
                dt,
            )

        # Effective valve opening
        if valve_clog_factor > 0:
            effective_valve = base_valve_opening * valve_clog_factor
        else:
            effective_valve = base_valve_opening

        # -------------- Control + physics --------------
        pump_command, pump_state = controller.update(tank.h)

        # Disturbance: random filling adds water → negative in tank.update()
        true_level, q_in, q_out = tank.update(
            pump_command,
            effective_valve,
            dt,
            disturbance=-random_filling_rate,
        )
        true_pressure = tank.calculate_pressure()

        # -------------- Sensors --------------
        level_measured = sensors.measure_level(true_level)
        flow_in_measured = sensors.measure_flow(q_in + random_filling_rate)
        flow_out_measured = sensors.measure_flow(q_out)
        pressure_measured = sensors.measure_pressure(true_pressure)
        current_measured = sensors.measure_current(pump_command, pump_state)

        # -------------- Security payload --------------
        sensor_payload = {
            "level": float(level_measured),
            "flow_in": float(flow_in_measured),
            "flow_out": float(flow_out_measured),
            "pressure": float(pressure_measured),
            "pump_current": float(current_measured),
        }

        # Authentication
        provided_key = SECRET_KEY  # in real deployment, comes externally
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

        # Physical checks
        bounds_issue = monitor.detect_physical_bounds(
            sensor_payload["level"],
            sensor_payload["flow_in"],
            sensor_payload["flow_out"],
        )
        if bounds_issue:
            log(f"Physical bounds issue at t={t:.2f}s: {bounds_issue}", level="WARN")

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

        cmd_issue = monitor.detect_command_injection(
            pump_cmd=pump_command, valve_cmd=effective_valve
        )
        if cmd_issue:
            log(
                f"Command injection suspicion at t={t:.2f}s: {cmd_issue}",
                level="WARN",
            )

        # -------------- Build row --------------
        label = 0
        if with_faults and fault_gen is not None:
            # Use your previous convention
            if fault_gen.valve_is_clogged and fault_gen.is_filling:
                label = 5
            elif fault_gen.valve_is_clogged:
                label = 6
            elif fault_gen.is_filling:
                label = 7
            else:
                label = 0

        row = {
            "timestamp": t,
            "scenario_id": "realtime_stream",
            "label": label,
            "level_real": level_measured,
            "flow_in_real": flow_in_measured,
            "flow_out_real": flow_out_measured,
            "pressure_real": pressure_measured,
            "pump_current": current_measured,
            "valve_position": effective_valve * 100,
            "valve_commanded": base_valve_opening * 100,
            "pump_state": pump_state,
            "controller_setpoint": controller.setpoint,
            "is_valve_clogged": (
                1 if (with_faults and fault_gen and fault_gen.valve_is_clogged) else 0
            ),
            "is_filling": (
                1 if (with_faults and fault_gen and fault_gen.is_filling) else 0
            ),
            "filling_rate": random_filling_rate,
            # Security metadata
            "auth_ok": auth_ok,
            "hash": row_hash,
            "replay_flag": int(is_replay),
            "bounds_issue": bounds_issue or "",
            "inconsistency_issue": inconsistency_issue or "",
        }

        yield row

        # Real-time pacing
        if speed_multiplier > 0:
            time.sleep(dt / speed_multiplier)

        t += dt


# Small demo if you run this file directly
if __name__ == "__main__":
    print("Starting demo real-time simulation (no ML, just printing)...")
    for i, row in enumerate(
        simulation_stream(duration=10.0, dt=0.2, with_faults=True, speed_multiplier=2.0)
    ):
        print(
            f"[t={row['timestamp']:5.2f}s] level={row['level_real']:.3f} m | "
            f"flow_in={row['flow_in_real']:.3f} | flow_out={row['flow_out_real']:.3f} | "
            f"label={row['label']}"
        )
