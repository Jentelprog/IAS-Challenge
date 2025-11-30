# Water Tank Digital Twin with Intrusion Detection System (IDS)

This repository contains a Python-based digital twin simulation of an industrial water tank system, integrated with an Intrusion Detection System (IDS) based on Machine Learning (ML) and security metadata analysis.

The project is structured to simulate a Cyber-Physical System (CPS) environment, generating realistic sensor data, applying control logic, introducing physical faults, and implementing security mechanisms to detect anomalies and attacks.

## Features

- **Digital Twin Simulation:** Realistic simulation of a water tank's physical dynamics (level, flow, pressure) using a time-step model.
- **Control System:** An On/Off controller maintains the water level at a defined setpoint.
- **Fault Generation:** Scripts to generate datasets for normal operation and various fault scenarios (e.g., valve clogging, unauthorized filling).
- **Machine Learning IDS:** A **Random Forest Classifier** trained to detect physical anomalies/faults in the sensor data stream.
- **Security Metadata Integration:** The simulation stream includes security checks for:
  - **Authentication**
  - **Integrity** (data hashing)
  - **Anti-Replay** (detection of repeated sensor values)
  - **Physical Bounds & Inconsistency Monitoring**
- **Real-Time Dashboard:** A Streamlit application for visualizing the live simulation data and ML-based anomaly predictions.

## Project Structure

The project is organized into three main components: the core simulation, the machine learning IDS, and the visualization dashboard.

| Directory                             | Purpose                                                  | Key Files                                                                    |
| :------------------------------------ | :------------------------------------------------------- | :--------------------------------------------------------------------------- |
| `water_tank_simulation/src`           | Core simulation logic and data generation.               | `tank_model.py`, `controller.py`, `realtime_simulation.py`                   |
| `water_tank_simulation/security`      | Security and integrity monitoring components.            | `authentication.py`, `integrity.py`, `anti_replay.py`, `security_monitor.py` |
| `water_tank_simulation/config`        | System configuration parameters.                         | `parameters.py`                                                              |
| `water_tank_simulation/data`          | Generated datasets for training and real-time inference. | `random_faults.csv`, `realtime_stream.csv`                                   |
| `water_tank_simulation/src/artifacts` | Trained ML model and evaluation results.                 | `rf_model.pkl`, `scaler.pkl`, `training_report.txt`                          |
| `dashboarding`                        | Streamlit web application for visualization.             | `vise.py`                                                                    |

## Installation

The project requires Python 3.10+ and the dependencies listed in `requirements.txt`.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Jentelprog/IAS-Challenge
    cd IAS-Challenge
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The workflow involves three main steps: data generation (optional), model training, and real-time execution with the dashboard.

### 1. Train the Anomaly Detector

The `AI-anomaly-detector.py` script handles the training, evaluation, and artifact saving for the Random Forest classifier.

```bash
# Train the model using the provided random_faults.csv
python water_tank_simulation/src/AI-anomaly-detector.py --train
```

This command will save the trained model (`rf_model.pkl`), the data scaler (`scaler.pkl`), and evaluation reports to the `water_tank_simulation/src/artifacts` directory.

### 2. Run Real-Time Inference

The `run_realtime_inference.py` script starts the digital twin simulation and continuously feeds the live sensor data (including security metadata) into the trained ML model for real-time anomaly detection.

```bash
# Start the live simulation and inference loop
python water_tank_simulation/src/run_realtime_inference.py
```

This script writes the latest sensor data and ML predictions to `water_tank_simulation/data/realtime_stream.csv`, which is then read by the dashboard.

### 3. Launch the Dashboard

The Streamlit dashboard provides a visual interface to monitor the system state and the IDS output.

```bash
# Launch the Streamlit application
streamlit run dashboarding/vise.py
```

Access the dashboard in your web browser, typically at `http://localhost:8501`.

## Suggested Improvements

The current implementation is a solid foundation for a Cyber-Physical IDS challenge. Here are the detailed suggestions for enhancing the project:

| Category                | Suggestion                       | Rationale                                                                                                                                                                                                                                                                     |
| :---------------------- | :------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Security**            | **Secure Key Management**        | The `SECRET_KEY` is currently hardcoded in `security/authentication.py`. For production readiness, it should be loaded from a secure source like an environment variable or a secret manager to prevent exposure.                                                             |
| **ML Model**            | **Feature Engineering**          | The current model uses raw sensor values. Incorporating **time-series features** (e.g., rolling averages, standard deviations, time-lagged values) would significantly improve the model's ability to detect temporal anomalies and subtle changes in system dynamics.        |
| **ML Model**            | **Model Selection**              | While Random Forest is effective, exploring other time-series-aware models like **LSTMs** or **Isolation Forest** could provide better performance or a more robust baseline for comparison.                                                                                  |
| **Code Structure**      | **Configuration Centralization** | Centralize all configuration, including ML constants (`CLASS_NAMES`, artifact paths), into `config/parameters.py`. This avoids hardcoding paths and values in the ML script (`AI-anomaly-detector.py`), making the project easier to configure and maintain.                  |
| **Security Monitoring** | **Advanced Checks**              | Enhance the `SecurityMonitor` to include more sophisticated checks, such as **Rate Limiting** (detecting an unusually high rate of data transmission) and **Statistical Process Control (SPC)** (monitoring for shifts in the mean or variance of sensor readings over time). |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for any bugs or feature requests.
