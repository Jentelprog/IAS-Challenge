"""
Water Tank Physical Model
Implements the core physics equations
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.parameters import *


class WaterTank:
    """
    Single water tank with inlet pump and outlet valve
    """
    
    def __init__(self, initial_level=INITIAL_WATER_LEVEL, area=TANK_AREA):
        self.h = initial_level
        self.A = area
        self.k_pump = K_PUMP
        self.k_valve = K_VALVE
        self.g = GRAVITY
        
    def calculate_inlet_flow(self, pump_command):
        """Calculate inlet flow from pump"""
        return self.k_pump * pump_command
    
    def calculate_outlet_flow(self, valve_opening):
        """Calculate outlet flow (gravity-driven with valve control)"""
        if self.h <= 0:
            return 0.0
        return valve_opening * self.k_valve * np.sqrt(2 * self.g * self.h)
    
    def calculate_pressure(self):
        """Calculate pressure at tank bottom"""
        pressure_pa = WATER_DENSITY * self.g * self.h
        pressure_kpa = pressure_pa / 1000.0
        return pressure_kpa
    
    def update(self, pump_command, valve_opening, dt, disturbance=0.0):
        """Update water level using Euler integration"""
        q_in = self.calculate_inlet_flow(pump_command)
        q_out = self.calculate_outlet_flow(valve_opening)
        
        dh_dt = (1.0 / self.A) * (q_in - q_out - disturbance)
        
        self.h += dh_dt * dt
        self.h = np.clip(self.h, TANK_HEIGHT_MIN, TANK_HEIGHT_MAX)
        
        return self.h, q_in, q_out
    
    def get_state(self):
        """Get current tank state"""
        return {
            'level': self.h,
            'pressure': self.calculate_pressure()
        }


# Test the model
if __name__ == "__main__":
    print("Testing Water Tank Model...")
    tank = WaterTank()
    print(f"Initial level: {tank.h} m")
    
    # Simulate 10 steps
    for i in range(10):
        h, q_in, q_out = tank.update(pump_command=0.8, valve_opening=0.5, dt=0.1)
        print(f"Step {i+1}: Level={h:.3f}m, Flow_in={q_in:.3f}, Flow_out={q_out:.3f}")
    
    print("Tank model works!")
