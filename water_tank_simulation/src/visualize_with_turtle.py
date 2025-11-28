"""
Live Water Tank Visualization using Turtle Graphics
Shows real-time animation of the water tank system
"""

import turtle
import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tank_model import WaterTank
from controller import OnOffController
from sensor_model import SensorSuite
from config.parameters import *


class WaterTankVisualizer:
    """
    Real-time visualization of water tank using Turtle graphics
    """

    def __init__(self):
        # Setup screen
        self.screen = turtle.Screen()
        self.screen.setup(width=1000, height=800)
        self.screen.bgcolor("lightblue")
        self.screen.title(" Water Tank Simulation - Live View")
        self.screen.tracer(0)  # Turn off auto-update for smooth animation

        # Tank dimensions (in pixels)
        self.tank_width = 200
        self.tank_height = 350
        self.tank_x = -100
        self.tank_y = -200

        # Create drawing turtles
        self.tank_drawer = turtle.Turtle()
        self.water_drawer = turtle.Turtle()
        self.pump_drawer = turtle.Turtle()
        self.valve_drawer = turtle.Turtle()
        self.text_drawer = turtle.Turtle()

        # Hide turtles
        for t in [
            self.tank_drawer,
            self.water_drawer,
            self.pump_drawer,
            self.valve_drawer,
            self.text_drawer,
        ]:
            t.hideturtle()
            t.speed(0)

        # Draw static elements
        self.draw_tank_structure()
        self.draw_labels()

    def draw_tank_structure(self):
        """Draw the tank outline and structure"""
        t = self.tank_drawer
        t.penup()

        # Draw tank walls
        t.goto(self.tank_x, self.tank_y)
        t.pendown()
        t.pensize(5)
        t.color("darkblue")

        # Bottom
        t.goto(self.tank_x + self.tank_width, self.tank_y)
        # Right wall
        t.goto(self.tank_x + self.tank_width, self.tank_y + self.tank_height)
        # Top
        t.goto(self.tank_x, self.tank_y + self.tank_height)
        # Left wall
        t.goto(self.tank_x, self.tank_y)

        # Draw setpoint line
        setpoint_y = (
            self.tank_y + (SETPOINT_DEFAULT / TANK_HEIGHT_MAX) * self.tank_height
        )
        t.penup()
        t.goto(self.tank_x - 30, setpoint_y)
        t.pendown()
        t.pensize(2)
        t.color("red")
        t.goto(self.tank_x + self.tank_width + 30, setpoint_y)

        # Draw level markers
        t.color("gray")
        t.pensize(1)
        for level in [0.5, 1.0, 1.5]:
            y = self.tank_y + (level / TANK_HEIGHT_MAX) * self.tank_height
            t.penup()
            t.goto(self.tank_x - 10, y)
            t.pendown()
            t.goto(self.tank_x, y)
            t.penup()
            t.goto(self.tank_x + self.tank_width, y)
            t.pendown()
            t.goto(self.tank_x + self.tank_width + 10, y)

            # Label
            t.penup()
            t.goto(self.tank_x - 40, y - 10)
            t.write(f"{level}m", font=("Arial", 10, "normal"))

    def draw_labels(self):
        """Draw static labels"""
        t = self.text_drawer
        t.penup()
        t.color("black")

        # Title
        t.goto(0, 350)
        t.write("WATER TANK SIMULATION", align="center", font=("Arial", 24, "bold"))

        # Pump label
        t.goto(-300, 100)
        t.write("INLET PUMP", align="center", font=("Arial", 12, "bold"))

        # Valve label
        t.goto(300, -100)
        t.write("OUTLET VALVE", align="center", font=("Arial", 12, "bold"))

    def draw_water(self, level):
        """Draw water inside tank"""
        self.water_drawer.clear()

        if level <= 0:
            return

        # Calculate water height in pixels
        water_height = (level / TANK_HEIGHT_MAX) * self.tank_height

        # Draw water rectangle
        t = self.water_drawer
        t.penup()
        t.goto(self.tank_x + 2, self.tank_y + 2)
        t.setheading(0)
        t.color("cyan")
        t.fillcolor("cyan")
        t.begin_fill()

        # Draw water shape
        for _ in range(2):
            t.forward(self.tank_width - 4)
            t.left(90)
            t.forward(water_height - 2)
            t.left(90)

        t.end_fill()

        # Draw water surface waves
        t.penup()
        t.color("darkblue")
        t.pensize(2)
        wave_y = self.tank_y + water_height
        t.goto(self.tank_x, wave_y)
        t.pendown()

        # Simple wave pattern
        for x in range(0, int(self.tank_width), 20):
            t.goto(self.tank_x + x, wave_y + 3)
            t.goto(self.tank_x + x + 10, wave_y - 3)
            t.goto(self.tank_x + x + 20, wave_y)

    def draw_pump(self, is_on, flow_rate):
        """Draw pump and water flowing in"""
        self.pump_drawer.clear()
        t = self.pump_drawer

        # Pump position
        pump_x = -300
        pump_y = 150

        # Draw pump box
        t.penup()
        t.goto(pump_x - 30, pump_y - 30)
        t.pendown()
        t.pensize(3)

        if is_on:
            t.color("green")
            t.fillcolor("lightgreen")
        else:
            t.color("gray")
            t.fillcolor("lightgray")

        t.begin_fill()
        for _ in range(4):
            t.forward(60)
            t.left(90)
        t.end_fill()

        # Draw pump symbol (circle in middle)
        t.penup()
        t.goto(pump_x, pump_y - 15)
        t.pendown()
        t.circle(15)

        # Draw flow if pump is on
        if is_on and flow_rate > 0:
            # Pipe from pump to tank
            t.penup()
            t.goto(pump_x + 30, pump_y)
            t.pendown()
            t.pensize(5)
            t.color("blue")
            t.goto(self.tank_x, self.tank_y + self.tank_height - 20)

            # Animated water drops
            for i in range(3):
                drop_x = pump_x + 30 + i * 40
                drop_y = pump_y - i * 30
                t.penup()
                t.goto(drop_x, drop_y)
                t.dot(10, "blue")

        # Flow rate text
        t.penup()
        t.goto(pump_x, pump_y - 80)
        t.color("black")
        status = "ON" if is_on else "OFF"
        t.write(
            f"Status: {status}\nFlow: {flow_rate*1000:.1f} L/s",
            align="center",
            font=("Arial", 10, "normal"),
        )

    def draw_valve(self, position, flow_rate):
        """Draw valve and water flowing out"""
        self.valve_drawer.clear()
        t = self.valve_drawer

        # Valve position
        valve_x = 300
        valve_y = self.tank_y + 50

        # Draw valve
        t.penup()
        t.goto(valve_x - 20, valve_y)
        t.pendown()
        t.pensize(3)

        # Color based on position
        if position > 40:
            t.color("green")
            t.fillcolor("lightgreen")
        elif position > 20:
            t.color("orange")
            t.fillcolor("lightyellow")
        else:
            t.color("red")
            t.fillcolor("lightcoral")

        # Draw valve triangle
        t.begin_fill()
        t.goto(valve_x, valve_y + 30)
        t.goto(valve_x + 20, valve_y)
        t.goto(valve_x - 20, valve_y)
        t.end_fill()

        # Draw pipe from tank to valve
        t.penup()
        t.goto(self.tank_x + self.tank_width, self.tank_y + 50)
        t.pendown()
        t.pensize(5)
        t.color("darkblue")
        t.goto(valve_x - 20, valve_y)

        # Draw outflow if valve is open
        if position > 10 and flow_rate > 0:
            # Water stream going down
            t.penup()
            t.goto(valve_x, valve_y - 5)
            t.pendown()
            t.pensize(int(position / 5))  # Width proportional to opening
            t.color("blue")
            t.goto(valve_x, valve_y - 100)

            # Splash at bottom
            t.penup()
            t.goto(valve_x, valve_y - 100)
            t.dot(20, "lightblue")

        # Valve info text
        t.penup()
        t.goto(valve_x, valve_y - 140)
        t.color("black")
        t.write(
            f"Opening: {position:.1f}%\nFlow: {flow_rate*1000:.1f} L/s",
            align="center",
            font=("Arial", 10, "normal"),
        )

    def draw_info_panel(self, time_sec, level, pressure, pump_state, setpoint):
        """Draw information panel"""
        t = self.text_drawer

        # Clear previous text
        t.clear()

        # Redraw title
        t.penup()
        t.color("black")
        t.goto(0, 350)
        t.write("WATER TANK SIMULATION", align="center", font=("Arial", 24, "bold"))

        # Info panel
        info_x = 0
        info_y = 280

        t.goto(info_x, info_y)
        t.write(
            f"  Time: {time_sec:.1f}s ({time_sec/60:.1f} min)",
            align="center",
            font=("Arial", 14, "normal"),
        )

        t.goto(info_x, info_y - 30)
        level_color = "green" if abs(level - setpoint) < 0.15 else "red"
        t.color(level_color)
        t.write(
            f" Water Level: {level:.3f} m", align="center", font=("Arial", 16, "bold")
        )

        t.color("black")
        t.goto(info_x, info_y - 60)
        t.write(
            f" Target: {setpoint:.1f} m", align="center", font=("Arial", 12, "normal")
        )

        t.goto(info_x, info_y - 85)
        t.write(
            f" Pressure: {pressure:.2f} kPa",
            align="center",
            font=("Arial", 12, "normal"),
        )

        t.goto(info_x, info_y - 110)
        pump_status = "ON" if pump_state == 1 else " OFF"
        t.write(f"Pump: {pump_status}", align="center", font=("Arial", 12, "bold"))

        # Redraw static labels
        t.goto(-300, 200)
        t.write("INLET PUMP", align="center", font=("Arial", 12, "bold"))

        t.goto(300, -100)
        t.write("OUTLET VALVE", align="center", font=("Arial", 12, "bold"))

    def update_display(self):
        """Update the screen"""
        self.screen.update()

    def close(self):
        """Close the visualization"""
        self.screen.bye()


def run_simulation_with_visualization(duration=120, speed_multiplier=1.0):
    """
    Run simulation with live turtle visualization

    Args:
        duration: Simulation duration in seconds
        speed_multiplier: 1.0 = real-time, 2.0 = 2x speed, 0.5 = half speed
    """

    print("=" * 70)
    print(" STARTING TURTLE VISUALIZATION".center(70))
    print("=" * 70)
    print(f"Duration: {duration}s ({duration/60:.1f} minutes)")
    print(f"Speed: {speed_multiplier}x")
    print()
    print(" Close the turtle window to stop simulation")
    print("=" * 70)
    print()

    # Initialize visualization
    viz = WaterTankVisualizer()

    # Initialize system
    tank = WaterTank()
    controller = OnOffController()
    sensors = SensorSuite(seed=42)

    valve_opening = VALVE_POSITION_DEFAULT

    # Simulation parameters
    dt = 0.1  # Time step
    steps = int(duration / dt)

    # Animation delay (adjust based on speed_multiplier)
    # Lower delay = faster animation
    base_delay = 0.05  # 50ms per frame
    animation_delay = base_delay / speed_multiplier

    try:
        t = 0.0
        for step in range(steps):
            # Controller decides
            pump_command, pump_state = controller.update(tank.h)

            # Update physics
            true_level, q_in, q_out = tank.update(pump_command, valve_opening, dt)
            true_pressure = tank.calculate_pressure()

            # Measure with sensors
            level_measured = sensors.measure_level(true_level)

            # Update visualization every 5 steps (every 0.5 seconds)
            if step % 5 == 0:
                viz.draw_water(level_measured)
                viz.draw_pump(pump_state == 1, q_in)
                viz.draw_valve(valve_opening * 100, q_out)
                viz.draw_info_panel(
                    t, level_measured, true_pressure, pump_state, controller.setpoint
                )
                viz.update_display()

                # Console output
                if step % 50 == 0:  # Every 5 seconds
                    pump_str = "ON " if pump_state == 1 else "OFF"
                    print(
                        f"[{t:6.1f}s] Level: {level_measured:.3f}m | "
                        f"Pump: {pump_str} | "
                        f"Flow In: {q_in*1000:5.1f} L/s | "
                        f"Flow Out: {q_out*1000:5.1f} L/s"
                    )

                # Small delay for animation
                time.sleep(animation_delay)

            t += dt

    except turtle.Terminator:
        print("\n  Visualization closed by user")

    except KeyboardInterrupt:
        print("\n  Simulation interrupted by user")

    finally:
        print("\n" + "=" * 70)
        print(" SIMULATION COMPLETE".center(70))
        print("=" * 70)
        print(f"Total time simulated: {t:.1f}s")
        print(f"Final water level: {tank.h:.3f}m")

        # Keep window open for a moment
        try:
            time.sleep(2)
            viz.close()
        except:
            pass


def run_with_random_faults(duration=180):
    """
    Run simulation with random faults AND visualization
    """

    print("=" * 70)
    print(" RANDOM FAULTS + TURTLE VISUALIZATION".center(70))
    print("=" * 70)
    print(f"Duration: {duration}s ({duration/60:.1f} minutes)")
    print("Includes: Valve Clogging + Random Filling")
    print()
    print(" Close the turtle window to stop simulation")
    print("=" * 70)
    print()

    # Import fault generator
    from generate_random_faults import RandomFaultGenerator

    # Initialize visualization
    viz = WaterTankVisualizer()

    # Initialize system
    tank = WaterTank()
    controller = OnOffController()
    sensors = SensorSuite(seed=42)
    fault_gen = RandomFaultGenerator(seed=42)

    base_valve_opening = VALVE_POSITION_DEFAULT

    dt = 0.1
    steps = int(duration / dt)
    animation_delay = 0.05

    try:
        t = 0.0
        filling_start_capacitie = TANK_HEIGHT_MAX * 0.2
        filling_end_capacitie = TANK_HEIGHT_MAX * 0.8

        for step in range(steps):
            # Check for faults
            valve_clog_factor = fault_gen.check_valve_clogging(t, dt)
            random_filling = fault_gen.check_random_filling(
                filling_start_capacitie, filling_end_capacitie, tank.h, dt
            )

            # Effective valve opening
            if valve_clog_factor > 0:
                effective_valve = base_valve_opening * valve_clog_factor
            else:
                effective_valve = base_valve_opening

            # Controller decides
            pump_command, pump_state = controller.update(tank.h)

            # Update physics
            true_level, q_in, q_out = tank.update(
                pump_command, effective_valve, dt, disturbance=-random_filling
            )
            true_pressure = tank.calculate_pressure()

            # Measure
            level_measured = sensors.measure_level(true_level)
            flow_in_measured = q_in + random_filling

            # Update visualization
            if step % 5 == 0:
                viz.draw_water(level_measured)
                viz.draw_pump(pump_state == 1, flow_in_measured)
                viz.draw_valve(effective_valve * 100, q_out)
                viz.draw_info_panel(
                    t, level_measured, true_pressure, pump_state, controller.setpoint
                )
                viz.update_display()

                # Console with fault info
                if step % 50 == 0:
                    pump_str = "ON " if pump_state == 1 else "OFF"
                    fault_str = ""
                    if fault_gen.valve_is_clogged:
                        fault_str += " CLOGGED "
                    if fault_gen.is_filling:
                        fault_str += " FILLING "

                    print(
                        f"[{t:6.1f}s] Level: {level_measured:.3f}m | "
                        f"Pump: {pump_str} | "
                        f"Valve: {effective_valve*100:4.1f}% | {fault_str}"
                    )

                time.sleep(animation_delay)

            t += dt

    except turtle.Terminator:
        print("\n  Visualization closed by user")

    except KeyboardInterrupt:
        print("\n Simulation interrupted by user")

    finally:
        print("\n" + "=" * 70)
        print(" SIMULATION COMPLETE".center(70))
        print("=" * 70)

        try:
            time.sleep(2)
            viz.close()
        except:
            pass


if __name__ == "__main__":
    print("\n WATER TANK TURTLE VISUALIZATION\n")
    print("Choose simulation mode:")
    print("1. Normal operation (2 minutes)")
    print("2. Normal operation - FAST (2 minutes at 2x speed)")
    print("3. Long simulation (5 minutes)")
    print("4. With random faults (3 minutes)")
    print()

    choice = input("Enter choice (1-4) or press Enter for default: ").strip()

    if choice == "2":
        run_simulation_with_visualization(duration=120, speed_multiplier=2.0)
    elif choice == "3":
        run_simulation_with_visualization(duration=300, speed_multiplier=1.0)
    elif choice == "4":
        run_with_random_faults(duration=180)
    else:
        # Default: normal operation
        run_simulation_with_visualization(duration=120, speed_multiplier=1.0)
