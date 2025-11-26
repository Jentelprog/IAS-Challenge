"""
Simple On-Off Controller for Water Tank
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.parameters import *


class OnOffController:
    """Simple bang-bang controller"""
    
    def __init__(self, setpoint=SETPOINT_DEFAULT, deadband=CONTROLLER_DEADBAND):
        self.setpoint = setpoint
        self.deadband = deadband
        self.pump_on = False
        
    def update(self, current_level):
        """Calculate pump command based on current level"""
        if current_level < self.setpoint - self.deadband:
            self.pump_on = True
        elif current_level > self.setpoint + self.deadband:
            self.pump_on = False
        
        if self.pump_on:
            return PUMP_POWER_ON, 1
        else:
            return PUMP_POWER_OFF, 0
    
    def set_setpoint(self, new_setpoint):
        """Update controller setpoint"""
        self.setpoint = new_setpoint


# Test the controller
if __name__ == "__main__":
    print("Testing Controller...")
    controller = OnOffController()
    
    # Test different levels
    test_levels = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for level in test_levels:
        cmd, state = controller.update(level)
        print(f"Level={level}m → Pump={'ON' if state==1 else 'OFF'} (command={cmd})")
    
    print("Controller works!")
