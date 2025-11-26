"""
Generate Random Fault Scenarios
Two faults that can occur simultaneously:
1. Valve Clogging (random)
2. Random Water Filling (random)
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


class RandomFaultGenerator:
    """
    Manages random faults during simulation
    """

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Valve clogging state
        self.valve_is_clogged = False
        self.valve_clog_level = 0.0  # 0 = normal, 1 = fully clogged
        self.valve_clog_start_time = None
        self.valve_clog_end_time = None

        # Random filling state
        self.is_filling = False
        self.filling_rate = 0.0  # m³/s
        self.filling_start_time = None
        self.filling_end_time = None

        # Timing parameters (seconds)
        self.min_time_between_events = 60.0  # Minimum 60s between faults
        self.last_event_time = 0.0

    def check_valve_clogging(self, current_time, dt):
        """
        Randomly trigger valve clogging
        Returns: effective valve opening (0-1)
        """

        # If already clogged, check if it's time to unclog
        if self.valve_is_clogged:
            if current_time >= self.valve_clog_end_time:
                print(f" [{current_time:.1f}s] Valve UNCLOGGED (cleared)")
                self.valve_is_clogged = False
                self.valve_clog_level = 0.0
                self.last_event_time = current_time
            return self.valve_clog_level

        # Random chance to start clogging (0.1% per time step when not recently happened)
        time_since_last = current_time - self.last_event_time
        if time_since_last > self.min_time_between_events:
            if np.random.random() < 0.001:  # 0.1% chance per step
                # Start clogging
                self.valve_is_clogged = True
                self.valve_clog_start_time = current_time

                # Random duration: 30-120 seconds
                clog_duration = np.random.uniform(30, 120)
                self.valve_clog_end_time = current_time + clog_duration

                # Random clog severity: 20-40% (valve opens only 20-40% of normal)
                self.valve_clog_level = np.random.uniform(0.2, 0.4)

                print(
                    f" [{current_time:.1f}s] Valve CLOGGED! "
                    f"Opening reduced to {self.valve_clog_level*100:.0f}% "
                    f"for {clog_duration:.0f}s"
                )

        return self.valve_clog_level if self.valve_is_clogged else 0.0

    def check_random_filling(self, current_time, dt):
        """
        Randomly trigger external water filling
        Returns: additional flow rate (m³/s)
        """

        # If already filling, check if it's time to stop
        if self.is_filling:
            if current_time >= self.filling_end_time:
                print(f" [{current_time:.1f}s] Random filling STOPPED")
                self.is_filling = False
                self.filling_rate = 0.0
                self.last_event_time = current_time
            return self.filling_rate

        # Random chance to start filling (0.08% per time step when not recently happened)
        time_since_last = current_time - self.last_event_time
        if time_since_last > self.min_time_between_events:
            if np.random.random() < 0.0008:  # 0.08% chance per step
                # Start filling
                self.is_filling = True
                self.filling_start_time = current_time

                # Random duration: 20-90 seconds
                fill_duration = np.random.uniform(20, 90)
                self.filling_end_time = current_time + fill_duration

                # Random filling rate: 0.05-0.15 m³/s (significant but not overwhelming)
                self.filling_rate = np.random.uniform(0.05, 0.15)

                print(
                    f" [{current_time:.1f}s] Random filling STARTED! "
                    f"Rate: {self.filling_rate*1000:.0f} L/s "
                    f"for {fill_duration:.0f}s"
                )

        return self.filling_rate if self.is_filling else 0.0

    def get_current_label(self):
        """
        Return label based on current faults
        Priority: Both > Clogged > Filling > Normal
        """
        if self.valve_is_clogged and self.is_filling:
            return 5  # Both faults (new label!)
        elif self.valve_is_clogged:
            return 6  # Valve clogged only (new label!)
        elif self.is_filling:
            return 7  # Random filling only (new label!)
        else:
            return 0  # Normal


def generate_random_faults_data(duration=600, dt=0.1, seed=42):
    """
    Generate dataset with random valve clogging and random filling
    Both faults can occur simultaneously
    """

    print("=" * 70)
    print(" GENERATING RANDOM FAULTS SIMULATION".center(70))
    print("=" * 70)
    print(f"Duration: {duration}s ({duration/60:.1f} minutes)")
    print(f"Faults: Valve Clogging + Random Water Filling")
    print(f"Both can happen simultaneously!")
    print("=" * 70)
    print()

    # Initialize system
    tank = WaterTank(initial_level=INITIAL_WATER_LEVEL)
    controller = OnOffController()
    sensors = SensorSuite(seed=seed)
    fault_gen = RandomFaultGenerator(seed=seed)

    # Base valve position
    base_valve_opening = VALVE_POSITION_DEFAULT

    # Data storage
    data = []
    t = 0.0
    steps = int(duration / dt)

    # Statistics
    event_log = []

    # Simulation loop
    for step in range(steps):
        # Check for random faults
        valve_clog_factor = fault_gen.check_valve_clogging(t, dt)
        random_filling_rate = fault_gen.check_random_filling(t, dt)

        # Calculate effective valve opening
        if valve_clog_factor > 0:
            # Valve is clogged - reduce opening
            effective_valve = base_valve_opening * valve_clog_factor
        else:
            # Normal valve operation
            effective_valve = base_valve_opening

        # Controller decides pump command
        pump_command, pump_state = controller.update(tank.h)

        # Update tank physics with random filling as disturbance
        # Note: negative disturbance = leak, positive = extra filling
        true_level, q_in, q_out = tank.update(
            pump_command,
            effective_valve,
            dt,
            disturbance=-random_filling_rate,  # Negative because it's ADDING water
        )

        true_pressure = tank.calculate_pressure()

        # Measure with sensors
        level_measured = sensors.measure_level(true_level)
        flow_in_measured = sensors.measure_flow(
            q_in + random_filling_rate
        )  # Total inflow
        flow_out_measured = sensors.measure_flow(q_out)
        pressure_measured = sensors.measure_pressure(true_pressure)
        current_measured = sensors.measure_current(pump_command, pump_state)

        # Get current label
        label = fault_gen.get_current_label()

        # Store data
        data.append(
            {
                "timestamp": t,
                "scenario_id": "random_faults_001",
                "label": label,
                "level_real": level_measured,
                "flow_in_real": flow_in_measured,
                "flow_out_real": flow_out_measured,
                "pressure_real": pressure_measured,
                "pump_current": current_measured,
                "valve_position": effective_valve * 100,  # Actual valve opening %
                "valve_commanded": base_valve_opening * 100,  # What controller wanted
                "pump_state": pump_state,
                "controller_setpoint": controller.setpoint,
                "is_valve_clogged": 1 if fault_gen.valve_is_clogged else 0,
                "is_filling": 1 if fault_gen.is_filling else 0,
                "filling_rate": random_filling_rate,
            }
        )

        # Log events
        if label != 0 and (step == 0 or data[step - 1]["label"] != label):
            event_log.append(
                {
                    "time": t,
                    "event": (
                        "Valve Clogged"
                        if fault_gen.valve_is_clogged
                        else "Random Filling"
                    ),
                    "both": fault_gen.valve_is_clogged and fault_gen.is_filling,
                }
            )

        # Progress indicator
        if step % 1000 == 0 and step > 0:
            print(
                f"Progress: {100*step/steps:.0f}% | "
                f"Level: {true_level:.2f}m | "
                f"Events: {len(event_log)}"
            )

        t += dt

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    output_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "random_faults.csv"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print()
    print("=" * 70)
    print(" SIMULATION COMPLETE".center(70))
    print("=" * 70)
    print(f"Saved to: {output_path}")
    print(f"Total rows: {len(df)}")
    print(f"Total events: {len(event_log)}")
    print()

    # Print label distribution
    print(" Label Distribution:")
    label_names = {
        0: "Normal",
        5: "Both Faults (Clogged + Filling)",
        6: "Valve Clogged Only",
        7: "Random Filling Only",
    }

    for label, name in label_names.items():
        count = (df["label"] == label).sum()
        percent = 100 * count / len(df)
        if count > 0:
            print(f"   Label {label} ({name}): {count} rows ({percent:.1f}%)")

    print()
    print(" Event Log:")
    for event in event_log:
        both_marker = " + BOTH!" if event["both"] else ""
        print(f"   [{event['time']:.1f}s] {event['event']}{both_marker}")

    # Create visualization
    create_visualization(df, output_path)

    return df


def create_visualization(df, csv_path):
    """
    Create comprehensive visualization of the simulation
    """
    print("\n Creating visualization...")

    fig = plt.figure(figsize=(16, 12))

    # Color mapping for labels
    label_colors = {
        0: "green",
        5: "red",  # Both faults
        6: "orange",  # Valve clogged
        7: "blue",  # Random filling
    }

    colors = df["label"].map(label_colors)

    # Plot 1: Water Level
    ax1 = plt.subplot(4, 2, 1)
    scatter = ax1.scatter(df["timestamp"], df["level_real"], c=colors, s=2, alpha=0.6)
    ax1.axhline(
        y=SETPOINT_DEFAULT, color="black", linestyle="--", linewidth=2, label="Setpoint"
    )
    ax1.axhline(
        y=SETPOINT_DEFAULT + CONTROLLER_DEADBAND, color="gray", linestyle=":", alpha=0.5
    )
    ax1.axhline(
        y=SETPOINT_DEFAULT - CONTROLLER_DEADBAND, color="gray", linestyle=":", alpha=0.5
    )
    ax1.set_ylabel("Water Level [m]", fontsize=10)
    ax1.set_title("Water Level (color = fault type)", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Flows
    ax2 = plt.subplot(4, 2, 2)
    ax2.plot(
        df["timestamp"], df["flow_in_real"], label="Flow In", linewidth=1, alpha=0.8
    )
    ax2.plot(
        df["timestamp"], df["flow_out_real"], label="Flow Out", linewidth=1, alpha=0.8
    )
    ax2.set_ylabel("Flow [m³/s]", fontsize=10)
    ax2.set_title("Flow Rates", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Valve Position (Commanded vs Actual)
    ax3 = plt.subplot(4, 2, 3)
    ax3.plot(
        df["timestamp"],
        df["valve_commanded"],
        label="Commanded",
        linewidth=2,
        linestyle="--",
        color="blue",
        alpha=0.7,
    )
    ax3.plot(
        df["timestamp"],
        df["valve_position"],
        label="Actual",
        linewidth=1,
        color="red",
        alpha=0.8,
    )
    ax3.set_ylabel("Valve Position [%]", fontsize=10)
    ax3.set_title("Valve Opening (Clogging Detection)", fontsize=12, fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Filling Rate
    ax4 = plt.subplot(4, 2, 4)
    ax4.fill_between(
        df["timestamp"],
        0,
        df["filling_rate"] * 1000,
        color="cyan",
        alpha=0.5,
        label="Random Filling",
    )
    ax4.set_ylabel("Filling Rate [L/s]", fontsize=10)
    ax4.set_title("Random Water Filling", fontsize=12, fontweight="bold")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Fault Indicators
    ax5 = plt.subplot(4, 2, 5)
    ax5.fill_between(
        df["timestamp"],
        0,
        df["is_valve_clogged"],
        color="orange",
        alpha=0.5,
        label="Valve Clogged",
        step="post",
    )
    ax5.fill_between(
        df["timestamp"],
        0,
        df["is_filling"],
        color="blue",
        alpha=0.3,
        label="Random Filling",
        step="post",
    )
    ax5.set_ylabel("Fault Active", fontsize=10)
    ax5.set_ylim([-0.1, 1.3])
    ax5.set_title("Fault Timeline", fontsize=12, fontweight="bold")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Pressure
    ax6 = plt.subplot(4, 2, 6)
    ax6.scatter(df["timestamp"], df["pressure_real"], c=colors, s=2, alpha=0.6)
    ax6.set_ylabel("Pressure [kPa]", fontsize=10)
    ax6.set_title("Bottom Pressure", fontsize=12, fontweight="bold")
    ax6.grid(True, alpha=0.3)

    # Plot 7: Pump State and Current
    ax7 = plt.subplot(4, 2, 7)
    ax7_twin = ax7.twinx()
    ax7.plot(
        df["timestamp"],
        df["pump_state"],
        color="red",
        linewidth=1,
        label="Pump State",
        alpha=0.7,
    )
    ax7_twin.plot(
        df["timestamp"],
        df["pump_current"],
        color="purple",
        linewidth=1,
        label="Current",
        alpha=0.7,
    )
    ax7.set_ylabel("Pump State [0/1]", color="red", fontsize=10)
    ax7_twin.set_ylabel("Current [A]", color="purple", fontsize=10)
    ax7.set_xlabel("Time [s]", fontsize=10)
    ax7.set_title("Pump Status", fontsize=12, fontweight="bold")
    ax7.grid(True, alpha=0.3)

    # Plot 8: Label Distribution
    ax8 = plt.subplot(4, 2, 8)
    label_counts = df["label"].value_counts().sort_index()
    label_names = {0: "Normal", 5: "Both", 6: "Clogged", 7: "Filling"}
    bar_colors = [label_colors.get(label, "gray") for label in label_counts.index]
    bars = ax8.bar(
        [label_names.get(l, str(l)) for l in label_counts.index],
        label_counts.values,
        color=bar_colors,
        alpha=0.7,
    )
    ax8.set_ylabel("Count", fontsize=10)
    ax8.set_title("Label Distribution", fontsize=12, fontweight="bold")
    ax8.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax8.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.suptitle(
        "Random Faults Simulation: Valve Clogging + Random Filling",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()

    # Save plot
    plot_path = csv_path.replace(".csv", ".png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f" Plot saved to: {plot_path}")

    # Show plot
    plt.show()


# Legend for labels
LABEL_LEGEND = """
 Label Legend:
   0 = Normal operation
   5 = BOTH faults (Valve Clogged + Random Filling)
   6 = Valve Clogged only
   7 = Random Filling only
"""


if __name__ == "__main__":
    print(LABEL_LEGEND)

    # Generate simulation
    df = generate_random_faults_data(
        duration=6000,  # 100 minutes
        dt=0.1,  # 10 Hz sampling
        seed=41,  # For reproducibility
    )

    print("\n" + "=" * 70)
    print("ALL DONE!".center(70))
    print("=" * 70)
    print("\n Files created:")
    print("    data/random_faults.csv")
    print("    data/random_faults.png")
    print("\n Share these with your ML team!")
    print("\n" + LABEL_LEGEND)
