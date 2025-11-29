"""
DT-IDS Dashboard (Streamlit)

Shows:
- Live real-time data from realtime_stream.csv
- Historical datasets (normal_operation.csv, random_faults.csv)
- Current anomaly status and anomaly timeline
- Simple future forecast of water level (linear trend extrapolation)

Requires:
    pip install streamlit pandas numpy matplotlib

Run from project root:
    cd dashboarding
    streamlit run vise.py
"""

import os
import time
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(THIS_DIR, "..", "water_tank_simulation")

DATA_DIR = os.path.join(BASE_DIR, "data")
REALTIME_FILE = os.path.join(DATA_DIR, "realtime_stream.csv")
NORMAL_FILE = os.path.join(DATA_DIR, "normal_operation.csv")
FAULT_FILE = os.path.join(DATA_DIR, "random_faults.csv")

# ---------------------------------------------------------------------
# Streamlit config
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="DT-IDS Dashboard",
    layout="wide",
)

st.title("AI-Driven Digital Twin IDS – Dashboard")
st.caption("Live simulation, anomaly detection, and historical analysis")


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------


@st.cache_data(ttl=2.0)
def load_realtime_data() -> pd.DataFrame:
    if not os.path.exists(REALTIME_FILE):
        return pd.DataFrame()
    df = pd.read_csv(REALTIME_FILE)
    return df


@st.cache_data(ttl=60.0)
def load_historical_data(which: str) -> pd.DataFrame:
    if which == "Normal operation":
        path = NORMAL_FILE
    else:
        path = FAULT_FILE

    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def simple_forecast(df: pd.DataFrame, horizon_seconds: float = 30.0) -> pd.DataFrame:
    """Very simple linear trend extrapolation for water level."""
    if df.empty or "timestamp" not in df.columns or "level_real" not in df.columns:
        return pd.DataFrame()

    if len(df) < 5:
        return pd.DataFrame()

    t = df["timestamp"].values
    y = df["level_real"].values

    # Estimate slope from first and last points
    dt = t[-1] - t[0]
    if dt <= 0:
        return pd.DataFrame()

    slope = (y[-1] - y[0]) / dt

    # Estimate time step
    if len(t) >= 2:
        dt_step = np.median(np.diff(t))
    else:
        dt_step = 0.1

    n_steps = max(1, int(horizon_seconds / dt_step))

    future_t = t[-1] + dt_step * np.arange(1, n_steps + 1)
    future_y = y[-1] + slope * (future_t - t[-1])

    return pd.DataFrame({"timestamp": future_t, "level_pred": future_y})


def status_badge(flag: int) -> str:
    return "🟥 ANOMALY" if flag == 1 else "🟩 NORMAL"


# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------

tab_live, tab_history = st.tabs(["🔴 Live Monitoring", "📚 Historical Data"])

# ---------------------------------------------------------------------
# LIVE TAB
# ---------------------------------------------------------------------
with tab_live:
    st.subheader("Live DT-IDS Monitoring")

    placeholder = st.empty()  # used for soft auto-refresh

    # You can use Streamlit's auto-refresh-like behaviour with a loop
    # BUT for simplicity we just reload once when user refreshes the page.
    # If you want auto-refresh, uncomment the small loop below.

    df_live = load_realtime_data()

    if df_live.empty:
        st.info(
            "No real-time data yet. "
            "Make sure `run_realtime_inference.py` is running and generating "
            "`realtime_stream.csv`."
        )
    else:
        df_live = df_live.sort_values("timestamp")

        # Last row = current status
        last = df_live.iloc[-1]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Time [s]", f"{last['timestamp']:.1f}")
        col2.metric("Water Level [m]", f"{last['level_real']:.3f}")
        col3.metric("Pump State", "ON" if last["pump_state"] == 1 else "OFF")
        col4.metric("IDS Status", status_badge(int(last.get("anomaly_flag", 0))))

        st.markdown("---")

        # Charts
        c1, c2 = st.columns((2, 1))

        with c1:
            st.markdown("#### Water Level & Anomaly Flags")

            chart_df = df_live[["timestamp", "level_real", "anomaly_flag"]].copy()
            chart_df["Anomaly"] = chart_df["anomaly_flag"].astype(bool)

            # Plot level
            st.line_chart(
                data=chart_df.set_index("timestamp")[["level_real"]],
                height=300,
            )

            # Mark anomalies as dots (simple textual table)
            anomalies = df_live[df_live["anomaly_flag"] == 1]
            if not anomalies.empty:
                st.write("Detected anomalies (latest 10):")
                st.dataframe(
                    anomalies.tail(10)[
                        [
                            "timestamp",
                            "level_real",
                            "flow_in_real",
                            "flow_out_real",
                            "label",
                            "anomaly_score",
                        ]
                    ],
                    use_container_width=True,
                )
            else:
                st.write("✅ No anomalies detected yet in this stream.")

        with c2:
            st.markdown("#### Flows & Pump Current")

            flows_df = df_live[["timestamp", "flow_in_real", "flow_out_real"]]
            st.line_chart(
                data=flows_df.set_index("timestamp"),
                height=150,
            )

            current_df = df_live[["timestamp", "pump_current"]]
            st.line_chart(
                data=current_df.set_index("timestamp"),
                height=150,
            )

        st.markdown("---")

        # Future prediction
        st.markdown("#### Simple Future Forecast (Water Level)")

        horizon = st.slider("Forecast horizon [seconds]", 10, 120, 30, step=10)
        forecast_df = simple_forecast(df_live, horizon_seconds=horizon)

        if forecast_df.empty:
            st.info("Not enough data yet for forecasting.")
        else:
            # Merge last part of history + forecast
            hist_tail = df_live[["timestamp", "level_real"]].tail(200)
            hist_tail = hist_tail.rename(columns={"level_real": "level"})

            forecast_plot_df = pd.DataFrame(
                {
                    "timestamp": hist_tail["timestamp"].tolist()
                    + forecast_df["timestamp"].tolist(),
                    "value": hist_tail["level"].tolist()
                    + forecast_df["level_pred"].tolist(),
                    "type": ["History"] * len(hist_tail)
                    + ["Forecast"] * len(forecast_df),
                }
            )

            st.line_chart(
                data=forecast_plot_df.pivot_table(
                    index="timestamp",
                    columns="type",
                    values="value",
                ),
                height=250,
            )

# ---------------------------------------------------------------------
# HISTORICAL TAB
# ---------------------------------------------------------------------
with tab_history:
    st.subheader("Historical Datasets Exploration")

    dataset_choice = st.radio(
        "Select dataset:",
        ["Normal operation", "Random faults"],
        horizontal=True,
    )

    df_hist = load_historical_data(dataset_choice)

    if df_hist.empty:
        st.info(f"No {dataset_choice.lower()} dataset found yet.")
    else:
        st.write(f"Loaded {len(df_hist)} rows.")

        # Basic filters
        max_rows = st.slider("Rows to preview", 100, min(5000, len(df_hist)), 500)
        st.dataframe(df_hist.head(max_rows), use_container_width=True)

        st.markdown("---")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Water Level")
            st.line_chart(
                data=df_hist.set_index("timestamp")[["level_real"]],
                height=300,
            )

        with col_b:
            st.markdown("#### Flow In / Flow Out")
            st.line_chart(
                data=df_hist.set_index("timestamp")[["flow_in_real", "flow_out_real"]],
                height=300,
            )

        if "label" in df_hist.columns:
            st.markdown("#### Label Distribution")
            label_counts = df_hist["label"].value_counts().sort_index()
            st.bar_chart(label_counts)
