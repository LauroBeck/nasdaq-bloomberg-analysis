import numpy as np
import streamlit as st
import plotly.graph_objects as go

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# -----------------------------
# Streamlit Page
# -----------------------------
st.set_page_config(layout="wide")
st.title("Quantum Market Regime Monitor (Live Simulation)")

# -----------------------------
# User Inputs (Dynamic Market Features)
# -----------------------------
colA, colB, colC = st.columns(3)

with colA:
    trend_strength = st.slider("Trend Strength", 0.0, 3.14, 1.2)

with colB:
    coherence_stress = st.slider("Cross-Index Stress", 0.0, 3.14, 2.4)

with colC:
    instability = st.slider("Instability / Volatility", 0.0, 3.14, 0.3)

# -----------------------------
# Build Quantum Circuit
# -----------------------------
qc = QuantumCircuit(3, 3)

qc.ry(trend_strength, 0)      # q0 → Directional bias
qc.ry(coherence_stress, 1)    # q1 → Agreement / rotation
qc.ry(instability, 2)         # q2 → Instability

qc.cx(0, 2)
qc.cx(1, 2)

qc.measure([0, 1, 2], [0, 1, 2])

# -----------------------------
# Execute Circuit (Live)
# -----------------------------
simulator = AerSimulator()
compiled = transpile(qc, simulator)

job = simulator.run(compiled, shots=4000)
result = job.result()
counts = result.get_counts()

shots = sum(counts.values())
probs = {state: counts[state] / shots for state in counts}

# -----------------------------
# Aggregate Factor Probabilities
# -----------------------------
p_q0_1 = sum(p for s, p in probs.items() if s[2] == "1")
p_q1_1 = sum(p for s, p in probs.items() if s[1] == "1")
p_q2_1 = sum(p for s, p in probs.items() if s[0] == "1")

# -----------------------------
# Regime Logic
# -----------------------------
if p_q2_1 > 0.6:
    regime = "HIGH INSTABILITY / TRANSITION"
    regime_color = "red"
elif p_q0_1 < 0.4 and p_q1_1 < 0.4:
    regime = "STABLE RISK-ON TREND"
    regime_color = "green"
elif p_q1_1 > 0.6:
    regime = "ROTATIONAL / DIVERGENCE"
    regime_color = "orange"
else:
    regime = "MIXED / NEUTRAL"
    regime_color = "gray"

# -----------------------------
# Gauge Builder
# -----------------------------
def gauge(title, value, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color}
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Layout
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    gauge("Directional Bias (q0)", p_q0_1, "green")

with col2:
    gauge("Cross-Index Stress (q1)", p_q1_1, "orange")

with col3:
    gauge("Instability / Volatility (q2)", p_q2_1, "red")

st.markdown("---")
st.subheader("Composite Regime Signal")

st.markdown(
    f"<h2 style='color:{regime_color}'>{regime}</h2>",
    unsafe_allow_html=True
)

st.markdown("---")
st.subheader("Quantum State Probabilities")
st.write({k: round(v, 4) for k, v in sorted(probs.items())})
