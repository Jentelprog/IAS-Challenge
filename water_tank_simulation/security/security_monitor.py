# security/security_monitor.py
from typing import Optional, Tuple


class SecurityMonitor:
    def __init__(self, level_min: float = 0.0, level_max: float = 2.0):
        self.level_min = level_min
        self.level_max = level_max

    def detect_physical_bounds(
        self, level: float, flow_in: float, flow_out: float
    ) -> Optional[str]:
        if level < self.level_min or level > self.level_max:
            return f"Level out of bounds: {level:.3f} m"

        if flow_in < 0 or flow_out < 0:
            return f"Negative flow detected (in={flow_in:.3f}, " f"out={flow_out:.3f})"

        # Additional pressure & current sanity checks could follow
        return None

    def detect_inconsistency(
        self, level: float, flow_in: float, flow_out: float, threshold: float = 0.5
    ) -> Optional[str]:
        """
        Very simple physical consistency check: detect large mismatches
        over a short dt. Threshold is domain-specific.
        """

        imbalance = abs(flow_in - flow_out)

        # Example: large inflow while tank appears empty
        if imbalance > threshold and level < 0.05:
            return "Physical inconsistency: large imbalance while tank " "nearly empty"

        return None

    def detect_command_injection(
        self, pump_cmd: float, valve_cmd: float
    ) -> Optional[str]:
        # Example suspicious pattern: pump at high power while valve is closed
        if pump_cmd > 0.95 and valve_cmd < 0.05:
            return "Suspicious command pattern: pump full power while " "valve closed"
        return None
