import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("../water_tank_simulation/data/random_faults.csv")

st.title("Water Tank Simulation Data Visualization")
st.dataframe(df)


st.subheader("Tank Height Over Time")
st.line_chart(df[["timestamp", "level_real"]].set_index("timestamp"))
