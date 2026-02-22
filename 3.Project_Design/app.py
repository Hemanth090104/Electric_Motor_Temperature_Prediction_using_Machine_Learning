import streamlit as st
import numpy as np
import joblib

model = joblib.load("motor_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Electric Motor Temperature Predictor")

voltage = st.number_input("Voltage")
current = st.number_input("Current")
speed = st.number_input("Speed")
torque = st.number_input("Torque")
ambient = st.number_input("Ambient Temp")
vibration = st.number_input("Vibration")
load = st.number_input("Motor Load")

if st.button("Predict"):
    data = np.array([[voltage,current,speed,torque,ambient,vibration,load]])
    data = scaler.transform(data)
    pred = model.predict(data)
    st.success(f"Predicted Temperature: {pred[0]:.2f} °C")