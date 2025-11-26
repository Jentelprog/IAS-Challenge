"""
Sensor Models with Realistic Noise
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.parameters import *


class SensorSuite:
    """Collection of sensors with realistic noise"""
    
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
    
    def measure_level(self, true_level):
        """Level sensor with Gaussian noise"""
        noise = np.random.normal(0, NOISE_LEVEL)
        measured = true_level + noise
        return np.clip(measured, TANK_HEIGHT_MIN, TANK_HEIGHT_MAX)
    
    def measure_flow(self, true_flow):
        """Flow sensor with Gaussian noise"""
        noise = np.random.normal(0, NOISE_FLOW)
        measured = true_flow + noise
        return max(0.0, measured)
    
    def measure_pressure(self, true_pressure):
        """Pressure sensor with Gaussian noise"""
        noise = np.random.normal(0, NOISE_PRESSURE)
        measured = true_pressure + noise
        return max(0.0, measured)
    
    def measure_current(self, pump_command, pump_state):
        """Current sensor (proportional to pump power)"""
        if pump_state == 0:
            true_current = PUMP_CURRENT_IDLE
        else:
            current_range = PUMP_CURRENT_MAX - PUMP_CURRENT_IDLE
            true_current = PUMP_CURRENT_IDLE + current_range * pump_command
        
        noise = np.random.normal(0, NOISE_CURRENT)
        measured = true_current + noise
        return np.clip(measured, 0.0, PUMP_CURRENT_MAX * 1.1)


# Test sensors
if __name__ == "__main__":
    print("Testing Sensors...")
    sensors = SensorSuite(seed=42)
    
    # Test level sensor
    true_level = 1.0
    for i in range(5):
        measured = sensors.measure_level(true_level)
        print(f"Level: True={true_level}m, Measured={measured:.4f}m")
    
    # Test current sensor
    print("\nCurrent sensor:")
    print(f"Pump OFF: {sensors.measure_current(0.0, 0):.2f}A")
    print(f"Pump ON (80%): {sensors.measure_current(0.8, 1):.2f}A")
    
    print("\n Sensors work!")