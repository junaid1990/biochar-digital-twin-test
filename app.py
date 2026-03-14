import streamlit as st  # <--- THIS IS THE FIX
import pandas as pd
import numpy as np
import joblib

# 1. Load the "Digital Twin" Brain
model = joblib.load('biochar_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Biochar Digital Twin", layout="wide")

st.title("🌱 Biochar Carbon Stability Digital Twin")
st.markdown("""
This app predicts the **H/C Ratio** (Carbon Stability) of biochar based on production parameters.
It uses **Gaussian Process Regression** to estimate uncertainty.
""")

# 2. Sidebar for User Inputs
st.sidebar.header("Biochar Recipe")
temp = st.sidebar.slider("Pyrolysis Temperature (°C)", 300, 900, 500)
time = st.sidebar.slider("Residence Time (min)", 30, 120, 60)
feed_c = st.sidebar.slider("Feedstock Carbon (%)", 40, 60, 50)
lignin = st.sidebar.slider("Lignin Content (%)", 10, 35, 25)
moisture = st.sidebar.slider("Moisture Content (%)", 5, 25, 10)
ph = st.sidebar.slider("Soil pH", 4.0, 9.0, 7.0)

# 3. Prediction Logic
features = np.array([[temp, time, feed_c, lignin, moisture, ph]])
features_scaled = scaler.transform(features)
prediction, sigma = model.predict(features_scaled, return_std=True)

# 4. Display Results
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Predicted H/C Ratio", value=f"{prediction[0]:.3f}")
    st.write(f"**Model Uncertainty (Sigma):** {sigma[0]:.4f}")

with col2:
    # Traffic Light System based on your Sigma Analysis
    if sigma[0] < 0.015:
        st.success("✅ High Reliability: This region is well-sampled.")
    elif sigma[0] < 0.021:
        st.warning("⚠️ Moderate Reliability: Results are approximate.")
    else:
        st.error("🚨 Low Reliability: More lab experiments (Active Learning) needed for this range!")

# 5. Sensitivity Visualization
st.divider()
st.subheader("Temperature Sensitivity Scan")
t_range = np.linspace(300, 900, 50)
scan_data = np.tile([temp, time, feed_c, lignin, moisture, ph], (50, 1))
scan_data[:, 0] = t_range
scan_scaled = scaler.transform(scan_data)
p_scan, s_scan = model.predict(scan_scaled, return_std=True)

chart_data = pd.DataFrame({
    'Temperature': t_range,
    'H/C Ratio': p_scan,
    'Lower Bound': p_scan - (1.96 * s_scan),
    'Upper Bound': p_scan + (1.96 * s_scan)
})

st.line_chart(chart_data.set_index('Temperature')[['H/C Ratio']])
st.write("The shaded area in a real plot would represent the 95% confidence interval.")
