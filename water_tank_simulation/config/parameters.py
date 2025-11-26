"""
Water Tank Simulation Parameters
All physical constants and configuration in one place
"""

# Tank Physical Parameters
TANK_AREA = 1.0
TANK_HEIGHT_MAX = 2.0
TANK_HEIGHT_MIN = 0.0

# Flow Parameters
K_PUMP = 0.3
K_VALVE = 0.5
GRAVITY = 9.81

# Fluid Properties
WATER_DENSITY = 1000.0

# Controller Parameters
SETPOINT_DEFAULT = 1.0
CONTROLLER_DEADBAND = 0.1
PUMP_POWER_ON = 0.8
PUMP_POWER_OFF = 0.0

# Valve Parameters
VALVE_POSITION_DEFAULT = 0.5

# Sensor Noise Levels
NOISE_LEVEL = 0.01
NOISE_FLOW = 0.005
NOISE_PRESSURE = 0.5
NOISE_CURRENT = 0.1

# Pump Current Parameters
PUMP_CURRENT_IDLE = 1.0
PUMP_CURRENT_MAX = 10.0

# Simulation Parameters
SIMULATION_DT = 0.1
SIMULATION_DURATION = 600

# Initial Conditions
INITIAL_WATER_LEVEL = 3.0

# Scenario Labels
LABEL_NORMAL = 0
LABEL_SPOOFING = 1
LABEL_REPLAY = 2
LABEL_VALVE_STUCK = 3
LABEL_SENSOR_DRIFT = 4

# Label Names
LABEL_NAMES = {
    0: "normal",
    1: "cyber_attack_spoofing",
    2: "cyber_attack_replay",
    3: "process_fault_valve_stuck",
    4: "sensor_fault_drift",
}
