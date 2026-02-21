import streamlit as st
import plotly.graph_objects as go

# -----------------------------
# Example probabilities (replace with live quantum output)
# -----------------------------
probs = {
    "001": 0.601,
    "110": 0.396,
    "000": 0.002,
    "111": 0.002
}

# Aggregate qubit probabilities
p_q0_1 = sum(p for s, p in probs.items() if s[2] == "1")  # Directional bias
p_q1_1 = sum(p for s, p in probs.items() if s[1] == "1")  # Coherence stress
p_q2_1 = sum(p for s, p in probs.items() if s[0] == "1")  # Instability

# -----------------------------
# Regime Classification Logic
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
# Streamlit Layout
# -----------------------------
st.set_page_config(layout="wide")
st.title("Quantum Market Regime Monitor")

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

# Optional probability table
st.markdown("---")
st.subheader("State Probabilities")
st.write({k: round(v, 4) for k, v in probs.items()})
