# 🚰 Water Tank Simulation System

### Fault Injection • Sensor Modeling • Control Logic • Dataset Generation • Visualization

This project is a **modular and extensible water tank simulation framework**.  
It models tank physics, pump control logic, sensor noise, and fault behaviors.  
The system can generate **clean datasets**, **faulty datasets**, and **live visualizations**.

It is ideal for:

- 🧠 Machine learning dataset generation
- 🔧 Fault detection & diagnosis research
- 🎓 Educational control system simulations
- 🧪 Controller testing / benchmarking

---

# 📁 Project Structure

```
project/
│
├── config/
│   └── parameters.py
│
├── controller.py
├── tank_model.py
├── sensor_model.py
│
├── generate_normal.py
├── generate_random_faults.py
│
├── visualize_with_turtle.py
│
└── data/
    ├── normal.csv
    ├── random_faults.csv
    └── random_faults.png
```

---

# ⚙️ System Overview

The simulation consists of **5 main subsystems**:

---

## 1. 🌊 WaterTank — Physical Tank Model

**File:** `tank_model.py`

Simulates realistic tank dynamics:

- Water inflow (pump)
- Water outflow (valve-controlled)
- Disturbances (leaks, extra filling)
- Safety level clamping
- Pressure computation (`ρ g h`)

**Main equation:**

```
dh/dt = (q_in - q_out ± disturbance) / A
```

Where:

- `A` = tank cross-sectional area
- `q_in` = pump flow
- `q_out` = valve-controlled outflow

---

## 2. ⚡ OnOffController — Pump Controller

**File:** `controller.py`

Implements a simple ON/OFF logic:

- Pump **ON** if level < (setpoint − deadband)
- Pump **OFF** if level > (setpoint + deadband)

---

## 3. 🎛 SensorSuite — Measurement Model

**File:** `sensor_model.py`

Simulates imperfect sensors with:

- Gaussian noise
- Drift
- Bias
- Non-linearity
- Pump current estimation
- Spoofing / attack modes

---

## 4. ⚠️ Fault Injection System

**File:** `generate_random_faults.py`

Injects two **process-level faults**:

### **Valve Clogging (label 6)**

- Random trigger
- Duration: **30–120 s**
- Severity: **20–40% open**

### **Random Filling (label 7)**

- Level-based trigger (<20%)
- Stops at (~80%)
- Filling rate: **0.05–0.15 m³/s**

### **Both Faults (label 5)**

Occurs when both fault mechanisms overlap.

---

## 5. 📊 Data Generators

### Normal Operation

**File:** `generate_normal.py`  
Output: `data/normal.csv`

### Fault Injection

**File:** `generate_random_faults.py`  
Output:

- `data/random_faults.csv`
- `data/random_faults.png`

---

# 📈 Visualization Tools

## Matplotlib Plot

Automatically generated fault visualization saved as PNG.

## Turtle Animation

Real-time dynamic tank simulation:

```
python visualize_with_turtle.py
```

---

# 🧰 Technologies Used

- Python 3.10+
- numpy
- pandas
- matplotlib
- turtle
- time / os / sys

---

# 🔧 Configuration System

Centralized config file:

```
config/parameters.py
```

---

# ▶️ Running the Simulation

### Normal:

```
python generate_normal.py
```

### Random Faults:

```
python generate_random_faults.py
```

### Live Turtle Animation:

```
python visualize_with_turtle.py
```

---

# 📄 Dataset Format

Each CSV contains:

| Column           | Description                  |
| ---------------- | ---------------------------- |
| timestamp        | simulation time              |
| level_real       | noisy level                  |
| flow_in_real     | inflow                       |
| flow_out_real    | outflow                      |
| pressure_real    | pressure                     |
| pump_state       | 0/1                          |
| pump_current     | estimated current            |
| valve_position   | actual valve position (%)    |
| valve_commanded  | commanded valve position (%) |
| is_valve_clogged | 1/0                          |
| is_filling       | 1/0                          |
| filling_rate     | m³/s                         |
| label            | ML class                     |
| scenario_id      | scenario tag                 |

---

# 🛠 Future Improvements

- PID controller
- Multi-tank system
- Additional sensor/actuator faults
- Reinforcement learning environment
- Streaming (MQTT/SocketIO)

---

# 🏁 Summary

This is a complete simulation environment for:

✔ Fault detection  
✔ Predictive maintenance  
✔ Control system teaching  
✔ ML dataset generation  
✔ SCADA/ICS security research

A robust and modular platform for testing intelligent control and fault diagnosis algorithms.
