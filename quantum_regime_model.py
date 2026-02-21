import numpy as np
import pandas as pd
import yfinance as yf

from qiskit import QuantumCircuit, Aer, transpile

# -----------------------------------
# CONFIGURATION
# -----------------------------------

TICKER = "JPMC34.SA"     # B3 BDR example
PERIOD = "1y"
INTERVAL = "1d"
TRADING_DAYS = 252
SHOTS = 4000

# -----------------------------------
# DATA DOWNLOAD
# -----------------------------------

print(f"Downloading data for {TICKER}...")

df = yf.download(TICKER, period=PERIOD, interval=INTERVAL)
df = df.dropna()

if df.empty:
    raise RuntimeError("No market data retrieved.")

close = df["Close"]

if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]

close = close.astype(float)

# -----------------------------------
# CLASSICAL METRICS (Bloomberg-style)
# -----------------------------------

returns = close.pct_change().dropna()

momentum_3m = float(close.iloc[-1] / close.iloc[-63] - 1)
volatility = float(returns.std() * np.sqrt(TRADING_DAYS))

print("\nCLASSICAL MARKET STATE")
print("-----------------------------------")
print(f"Last Price:    {close.iloc[-1]:.2f} BRL")
print(f"3M Momentum:   {momentum_3m:.2%}")
print(f"Volatility:    {volatility:.2%}")

# -----------------------------------
# NORMALIZE INTO QUANTUM ROTATIONS
# -----------------------------------
# Map financial variables -> [0, π]

def normalize_to_angle(x, clip=1.0):
    x = np.clip(x, -clip, clip)
    return (x + clip) / (2 * clip) * np.pi

theta_regime = normalize_to_angle(volatility, clip=1.0)
theta_momentum = normalize_to_angle(momentum_3m, clip=1.0)

# -----------------------------------
# QUANTUM CIRCUIT
# -----------------------------------

qc = QuantumCircuit(3)

# Encode regime uncertainty (volatility proxy)
qc.ry(theta_regime, 0)

# Encode momentum bias
qc.ry(theta_momentum, 1)

# Entanglement -> dependency structure
qc.cx(0, 1)
qc.cx(0, 2)

qc.measure_all()

# -----------------------------------
# EXECUTION
# -----------------------------------

backend = Aer.get_backend("aer_simulator")
compiled = transpile(qc, backend)

job = backend.run(compiled, shots=SHOTS)
result = job.result()

counts = result.get_counts()

# -----------------------------------
# PROBABILITY INTERPRETATION
# -----------------------------------

total = sum(counts.values())

def prob(state):
    return counts.get(state, 0) / total

p_calm = prob("000")
p_stress = prob("111")

print("\nQUANTUM REGIME DISTRIBUTION")
print("-----------------------------------")
for state, c in sorted(counts.items()):
    print(f"{state}: {c}")

print("\nREGIME SIGNAL")
print("-----------------------------------")
print(f"Calm Regime Probability : {p_calm:.2%}")
print(f"Stress Regime Probability: {p_stress:.2%}")

# -----------------------------------
# SIMPLE DECISION HEURISTIC
# -----------------------------------

if p_stress > 0.6:
    regime_view = "RISK OFF"
elif p_calm > 0.6:
    regime_view = "RISK ON"
else:
    regime_view = "NEUTRAL / MIXED"

print(f"\nQuantum Regime View: {regime_view}")

print("\nDone.")
