"""
DT-IDS Dashboard (Streamlit)

Live dashboard for:
- Real-time data from realtime_stream.csv
- Historical datasets (normal_operation.csv, random_faults.csv)
- Current anomaly status & fault type
- Simple future forecast of water level

Requires:
    pip install streamlit pandas numpy altair streamlit-autorefresh

Run from project root:
    cd dashboarding
    streamlit run vise.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from streamlit_autorefresh import st_autorefresh

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
# Streamlit base config
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="DT-IDS Dashboard",
    layout="wide",
)

st.title("🛰️ AI-Driven Digital Twin IDS – Dashboard")
st.caption("Live simulation • Anomaly detection • Industrial cybersecurity")

# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Controls")

    refresh_interval = st.slider(
        "Auto-refresh (seconds)",
        min_value=1.0,
        max_value=10.0,
        value=2.0,
        step=1.0,
        key="refresh_interval",
    )

    history_window = st.slider(
        "History window (seconds)",
        min_value=30,
        max_value=600,
        value=180,
        step=30,
        key="history_window",
    )

    st.markdown("---")
    st.markdown(
        "💡 Run:\n\n"
        "`python run_realtime_inference.py`\n\n"
        "from `water_tank_simulation/src` to start the live stream."
    )

# Auto-refresh the whole app every N seconds (no manual rerun needed)
st_autorefresh(interval=int(refresh_interval * 1000), key="dt_ids_autorefresh")

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------


@st.cache_data(ttl=1.0)
def load_realtime_data() -> pd.DataFrame:
    """Load the latest real-time stream produced by run_realtime_inference.py."""
    if not os.path.exists(REALTIME_FILE):
        return pd.DataFrame()
    return pd.read_csv(REALTIME_FILE)


@st.cache_data(ttl=60.0)
def load_historical_data(which: str):
    """
    Return (df, path, exists)
    Tries a couple of common locations and reports which one worked.
    """
    if which == "Normal operation":
        candidates = [
            NORMAL_FILE,
            os.path.join(BASE_DIR, "normal_operation.csv"),
        ]
    else:
        candidates = [
            FAULT_FILE,
            os.path.join(BASE_DIR, "random_faults.csv"),
        ]

    for path in candidates:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                return df, path, True
            except Exception:
                return pd.DataFrame(), path, False

    return pd.DataFrame(), candidates[0], False


def simple_forecast(df: pd.DataFrame, horizon_seconds: float = 30.0) -> pd.DataFrame:
    """Very simple linear trend extrapolation for water level."""
    if df.empty or "timestamp" not in df.columns or "level_real" not in df.columns:
        return pd.DataFrame()
    if len(df) < 5:
        return pd.DataFrame()

    t = df["timestamp"].values
    y = df["level_real"].values

    dt = t[-1] - t[0]
    if dt <= 0:
        return pd.DataFrame()

    slope = (y[-1] - y[0]) / dt

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


def attack_badge(is_under_attack: int, attack_type: str) -> str:
    if is_under_attack:
        return f"🧨 {attack_type}"
    return "🟩 none"


def sec_badge(sec_alert: int) -> str:
    return "🟥 ALERT" if sec_alert else "🟩 OK"


# ---------------------------------------------------------------------
# Tabs: Live + Historical
# ---------------------------------------------------------------------

tab_live, tab_history = st.tabs([" Live Monitoring", " Historical Data"])

# ---------------------------------------------------------------------
# LIVE TAB
# ---------------------------------------------------------------------
with tab_live:
    st.subheader("Live DT-IDS Monitoring")

    df_live = load_realtime_data()

    if df_live.empty:
        st.info(
            "No real-time data yet.\n\n"
            "In another terminal, run `python run_realtime_inference.py` "
            "from `water_tank_simulation/src` to start the simulation + IDS."
        )
    else:
        df_live = df_live.sort_values("timestamp")

        # Restrict to last N seconds to avoid overcrowded plots
        t_max = df_live["timestamp"].max()
        t_min = max(df_live["timestamp"].min(), t_max - history_window)
        df_view = df_live[df_live["timestamp"].between(t_min, t_max)]

        last = df_live.iloc[-1]

        anomaly_flag = int(last.get("anomaly_flag", 0))
        class_name = str(last.get("class_name", "unknown"))
        confidence = float(last.get("confidence", 0.0))

        is_under_attack = int(last.get("is_under_attack", 0))
        attack_type = str(last.get("attack_type", "none"))
        sec_alert = int(last.get("sec_alert", 0))
        sec_msg = str(last.get("sec_alert_message", "")).strip()

        total_samples = len(df_live)
        anomalies_count = int(df_live.get("anomaly_flag", 0).sum())
        anomaly_rate = anomalies_count / total_samples if total_samples > 0 else 0.0

        # -----------------------------------------------------------------
        # KPI section
        # -----------------------------------------------------------------
        col_snapshot, col_stats, col_model = st.columns(3)

        with col_snapshot:
            st.markdown("### System Snapshot")
            r1c1, r1c2, r1c3 = st.columns(3)
            r1c1.metric("Time [s]", f"{last['timestamp']:.1f}")
            r1c2.metric("Level [m]", f"{last['level_real']:.3f}")
            r1c3.metric("Pump", "ON" if last["pump_state"] == 1 else "OFF")

            r2c1, r2c2, r2c3 = st.columns(3)
            r2c1.metric("IDS Status", status_badge(anomaly_flag))
            r2c2.metric("Predicted", class_name)
            r2c3.metric("Confidence", f"{confidence*100:.1f}%")

            r3c1, r3c2 = st.columns(2)
            r3c1.metric("Cyber Attack", attack_badge(is_under_attack, attack_type))
            r3c2.metric("Security Monitor", sec_badge(sec_alert))

            if sec_alert and sec_msg:
                st.caption(f"Security alert: {sec_msg}")

        with col_stats:
            st.markdown("### Anomaly Statistics")
            c1, c2, c3 = st.columns(3)
            c1.metric("Samples", f"{total_samples}")
            c2.metric("Anomalies", f"{anomalies_count}")
            c3.metric("Rate", f"{anomaly_rate*100:.2f}%")

        with col_model:
            st.markdown("### Model Info")
            st.write(
                "- **Model**: RandomForestClassifier\n"
                "- **Classes**: `normal`, `fault_both`, `fault_clogged`, `fault_filling`\n"
                "- **Stream file**: `realtime_stream.csv`\n"
                "- **Extra fields**: `is_under_attack`, `attack_type`, `sec_alert`, "
                "`sec_alert_message`, `is_replay_suspected`, `integrity_hash`"
            )

        # Status banner
        if anomaly_flag == 1:
            st.warning(
                f" Detected **{class_name}** "
                f"(confidence {confidence*100:.1f}%) at t = {last['timestamp']:.1f} s."
            )
        else:
            st.success("IDS reports **normal** behaviour for the latest sample.")

        if is_under_attack:
            st.info(
                f"Cyber attack **{attack_type}** is active in the simulation "
                f"around t = {last['timestamp']:.1f} s."
            )

        st.markdown("---")

        # -----------------------------------------------------------------
        # Charts row – main graph left, extras right
        # -----------------------------------------------------------------
        main_col, side_col = st.columns([2.2, 1.8])

        # Main chart: water level + anomaly markers + attack shading
        with main_col:
            st.markdown("#### Water Level with Anomaly & Attack Markers")

            base = alt.Chart(df_view).encode(x=alt.X("timestamp:Q", title="Time [s]"))

            level_line = base.mark_line(color="#1f77b4").encode(
                y=alt.Y("level_real:Q", title="Water level [m]")
            )

            anomaly_points = (
                base.transform_filter("datum.anomaly_flag == 1")
                .mark_circle(size=70, color="red")
                .encode(
                    y="level_real:Q",
                    tooltip=[
                        alt.Tooltip("timestamp:Q", format=".2f", title="Time [s]"),
                        alt.Tooltip("level_real:Q", format=".3f", title="Level [m]"),
                        alt.Tooltip("class_name:N", title="Predicted"),
                        alt.Tooltip("confidence:Q", format=".2f", title="Confidence"),
                        alt.Tooltip("attack_type:N", title="Attack"),
                    ],
                )
            )

            # Attack shading (light orange) where is_under_attack == 1
            attack_layer = None
            if "is_under_attack" in df_view.columns:
                attack_layer = (
                    base.transform_filter("datum.is_under_attack == 1")
                    .mark_rect(opacity=0.15, color="orange")
                    .encode(
                        y=alt.value(0),
                        y2=alt.value(1),
                    )
                )

            if attack_layer is not None:
                chart = (attack_layer + level_line + anomaly_points).properties(
                    height=280
                )
            else:
                chart = (level_line + anomaly_points).properties(height=280)

            st.altair_chart(chart, use_container_width=True)

            # Latest anomalies table
            anomalies = df_live[df_live.get("anomaly_flag", 0) == 1]
            if not anomalies.empty:
                st.markdown("##### Latest anomalies")
                cols_to_show = [
                    "timestamp",
                    "level_real",
                    "flow_in_real",
                    "flow_out_real",
                    "label",
                ]
                for extra in [
                    "class_name",
                    "confidence",
                    "attack_type",
                    "is_under_attack",
                    "sec_alert",
                ]:
                    if extra in anomalies.columns:
                        cols_to_show.append(extra)
                st.dataframe(
                    anomalies.tail(8)[cols_to_show],
                    use_container_width=True,
                )
            else:
                st.write("No anomalies detected yet in this stream.")

        # Side charts: flows, current, predicted class timeline, attack & sec flags
        with side_col:
            st.markdown("#### Flows")

            if {"timestamp", "flow_in_real", "flow_out_real"}.issubset(df_view.columns):
                flows_df = df_view[["timestamp", "flow_in_real", "flow_out_real"]]
                flows_df = flows_df.set_index("timestamp")
                st.line_chart(flows_df, height=150)
            else:
                st.info("Flow columns missing in real-time stream.")

            st.markdown("#### Pump Current")
            if {"timestamp", "pump_current"}.issubset(df_view.columns):
                current_df = df_view[["timestamp", "pump_current"]]
                current_df = current_df.set_index("timestamp")
                st.line_chart(current_df, height=150)
            else:
                st.info("`pump_current` column missing in real-time stream.")

            if "class_name" in df_view.columns:
                st.markdown("#### Predicted Class Timeline")
                class_chart = (
                    alt.Chart(df_view)
                    .mark_rect()
                    .encode(
                        x=alt.X("timestamp:Q", title="Time [s]"),
                        y=alt.Y("class_name:N", title="Class"),
                        color=alt.Color("class_name:N", legend=None),
                        tooltip=[
                            alt.Tooltip("timestamp:Q", format=".2f", title="Time [s]"),
                            "class_name:N",
                            "attack_type:N",
                        ],
                    )
                    .properties(height=80)
                )
                st.altair_chart(class_chart, use_container_width=True)

            # Attack & security flags over time
            if {
                "timestamp",
                "is_under_attack",
                "sec_alert",
            }.issubset(df_view.columns):
                st.markdown("#### Cyber Attack & Security Flags")
                flags_df = df_view[["timestamp", "is_under_attack", "sec_alert"]].copy()
                flags_df = flags_df.melt("timestamp", var_name="flag", value_name="val")
                flag_chart = (
                    alt.Chart(flags_df)
                    .mark_line(step="post")
                    .encode(
                        x=alt.X("timestamp:Q", title="Time [s]"),
                        y=alt.Y(
                            "val:Q", title="Flag (0/1)", scale=alt.Scale(domain=[0, 1])
                        ),
                        color=alt.Color("flag:N", title="Signal"),
                        tooltip=[
                            alt.Tooltip("timestamp:Q", format=".2f", title="Time [s]"),
                            "flag:N",
                            "val:Q",
                        ],
                    )
                    .properties(height=130)
                )
                st.altair_chart(flag_chart, use_container_width=True)

        st.markdown("---")

        # -----------------------------------------------------------------
        # Simple future forecast
        # -----------------------------------------------------------------
        st.markdown("#### Simple Future Forecast (Water Level)")

        horizon = st.slider(
            "Forecast horizon [seconds]",
            10,
            120,
            30,
            step=10,
            key="forecast_horizon",
        )
        forecast_df = simple_forecast(df_live, horizon_seconds=horizon)

        if not forecast_df.empty:
            hist = df_view[["timestamp", "level_real"]].rename(
                columns={"level_real": "value"}
            )
            hist["type"] = "history"
            fut = forecast_df.rename(columns={"level_pred": "value"})
            fut["type"] = "forecast"
            combined = pd.concat([hist, fut], ignore_index=True)

            forecast_chart = (
                alt.Chart(combined)
                .mark_line()
                .encode(
                    x=alt.X("timestamp:Q", title="Time [s]"),
                    y=alt.Y("value:Q", title="Water level [m]"),
                    color=alt.Color("type:N", title="Series"),
                )
                .properties(height=240)
            )
            st.altair_chart(forecast_chart, use_container_width=True)
        else:
            st.info("Not enough data yet to compute a forecast.")

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

    df_hist, hist_path, exists = load_historical_data(dataset_choice)

    st.caption(f"Looking for file: `{os.path.relpath(hist_path)}`")

    if not exists:
        st.error(
            "Could not find the selected dataset file.\n\n"
            f"Make sure this file exists:\n\n`{hist_path}`"
        )
    elif df_hist.empty:
        st.warning(
            "The dataset was loaded but it contains **0 rows**.\n\n"
            "Check that your data-generation scripts actually wrote data "
            "to this file."
        )
    else:
        st.write(
            f"Loaded **{len(df_hist)}** rows from `{os.path.basename(hist_path)}`."
        )

        # Basic filters
        max_rows = st.slider(
            "Rows to preview",
            100,
            min(5000, len(df_hist)),
            500,
            step=100,
            key="hist_rows",
        )
        st.dataframe(df_hist.head(max_rows), use_container_width=True)

        st.markdown("---")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Water Level")
            if "timestamp" in df_hist.columns and "level_real" in df_hist.columns:
                st.line_chart(
                    data=df_hist.set_index("timestamp")[["level_real"]],
                    height=260,
                )
            else:
                st.info("`timestamp` or `level_real` column missing in this dataset.")

        with col_b:
            st.markdown("#### Flow In / Flow Out")
            if (
                "timestamp" in df_hist.columns
                and "flow_in_real" in df_hist.columns
                and "flow_out_real" in df_hist.columns
            ):
                st.line_chart(
                    data=df_hist.set_index("timestamp")[
                        ["flow_in_real", "flow_out_real"]
                    ],
                    height=260,
                )
            else:
                st.info("Flow columns missing in this dataset.")

        if "label" in df_hist.columns:
            st.markdown("#### Label Distribution")
            label_counts = df_hist["label"].value_counts().sort_index()
            st.bar_chart(label_counts)
        else:
            st.info("`label` column not found; cannot show label distribution.")
